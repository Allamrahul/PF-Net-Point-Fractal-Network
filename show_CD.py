#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import numpy as np
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
from utils import PointLoss_test
from utils import distance_squre
import data_utils as d_utils
import ModelNet40Loader
import shapenet_part_loader
from model_PFNet import _netlocalD,_netG

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset',  default='ModelNet40', help='ModelNet10|ModelNet40|ShapeNet')
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int,default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=512,help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default='Checkpoint/point_netG160.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[1536,1024,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.9,help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
parser.add_argument('--translation_delta',type=float,default=0.03,help='translation for pc')
opt = parser.parse_args()
print(opt)

def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 

def translate_pc(points):
    # translation = np.random.uniform(-opt.tran, self.translate_range)
    points[:, :, 0:3] += opt.translation_delta
    return points


test_dset = shapenet_part_loader.PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.pnum, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                         shuffle=False,num_workers = int(opt.workers))
length = len(test_dataloader)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num) 
point_netG = torch.nn.DataParallel(point_netG)
point_netG.to(device)
point_netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])   
point_netG.eval()

criterion_PointLoss = PointLoss_test().to(device)

input_cropped1 = torch.FloatTensor(opt.batchSize, 1, opt.pnum, 3)
errG_min = 100
n = 0
CD = 0
Gt_Pre =0
Pre_Gt = 0
IDX = 1


def save(p, filename):
    np.savetxt(filename + '.csv', torch.squeeze(p), fmt="%f,%f,%f")

# TODO
def tranformation_mtx(p1, p2): # p2 is the new centroid, p1 is the old centroid
    """
    computes the transformation matrix between source point cloud centroid and destiantion point cloud centroid p2
    """
    p = torch.FloatTensor(
        [[1, 0, 0, p2[0] - p1[0]],
         [0, 1, 0, p2[1] - p1[1]],
         [0, 0, 1, p2[2] - p1[2]],
         [0, 0, 0, 1]]
    )
    return p

import copy

# TODO
def final_t(tx, input_cropped1_tmp, real_point_tmp):
    """
    Accepts tx, the transformation mtx, the input_cropped points and the real_points (ground truth)
    """
    # deep copy
    input_cropped1_tmp = copy.deepcopy(input_cropped1_tmp)
    real_point_tmp = copy.deepcopy(real_point_tmp)

    # squeezing both point clouds
    input_cropped1_tmp = torch.squeeze(input_cropped1_tmp)
    real_point_tmp = torch.squeeze(real_point_tmp)

    # pad ones and add it as last column for both point clouds
    temp = torch.ones((input_cropped1_tmp.shape[0], 1))
    input_cropped1_p = torch.cat((input_cropped1_tmp, temp), 1)

    temp = torch.ones((real_point_tmp.shape[0], 1))
    real_point_p = torch.cat((real_point_tmp, temp), 1)

    # applying the matrix transformation for both point clouds, selcting the first 3 rows and taking transpose
    input_cropped1_p_r = torch.matmul(tx, input_cropped1_p.t())[:3, :].t()
    real_point_p_r = torch.matmul(tx, real_point_p.t())[:3, :].t()

    # unsqueezing to get the same shape back
    return torch.unsqueeze(torch.unsqueeze(real_point_p_r,0), 0), torch.unsqueeze(input_cropped1_p_r, 0)

# TODO
def centeroidnp(arr, num=None):
    """
    computes centroid of point cloud
    """
    arr = torch.squeeze(arr).numpy()
    shape = arr.shape[0]
    if num:
        shape = num
    return np.sum(arr[:, 0])/shape, np.sum(arr[:, 1])/shape, np.sum(arr[:, 2])/shape

for i, data in enumerate(test_dataloader, 0):


    real_point, target = data

    # save(real_point, "real_point_wholepc")

    real_point = torch.unsqueeze(real_point, 1)
    batch_size = real_point.size()[0]
    # print("Real point shape", real_point.shape)

    # computing centroid of ground truth
    real_point_centroid = centeroidnp(real_point) # TODO
    # print("Real point centroid ", real_point_centroid)



    real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)  # target
    input_cropped_partial =torch.FloatTensor(batch_size, 1, opt.pnum-opt.crop_point_num, 3)       # input
    input_cropped1.resize_(real_point.size()).copy_(real_point) # same size as real_point, whole pc

    p_origin = [0,0,0]

    choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
    index = choice[IDX-1]
    IDX  = IDX+1
    if IDX%5 == 0:
        IDX = 0
    distance_list = []
#    p_center  = real_point[0,0,index]
    p_center = index

    # sorting points based on proximity to p_center
    for num in range(opt.pnum):
        distance_list.append(distance_squre(real_point[0,0,num],p_center))
    distance_order = sorted(enumerate(distance_list), key = lambda x:x[1])

    # replacing the closest crop_point_num points in input_cropped1 to zeros
    # Adding the closes crop_point_num points to real_center
    for sp in range(opt.crop_point_num):
        input_cropped1.data[0,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0]) # whole pc, replacing the closes points with zeros
        real_center.data[0,0,sp] = real_point[0,0,distance_order[sp][0]]    # target

    # getting the rest of the points in order
    crop_num_list = []
    for num_ in range(opt.pnum-opt.crop_point_num):
        crop_num_list.append(distance_order[num_+opt.crop_point_num][0])
    indices = torch.LongTensor(crop_num_list)

    # input - contains not the closest points
    input_cropped_partial[0,0]=torch.index_select(real_point[0,0],0,indices)
    input_cropped_partial = torch.squeeze(input_cropped_partial,1)

    input_cropped_partial = input_cropped_partial.to(device)

    real_center = torch.squeeze(real_center,1) # target
#    real_center_key_idx = utils.farthest_point_sample(real_center,64,train = False)
#    real_center_key = utils.index_points(real_center,real_center_key_idx)
#    input_cropped1 = input_cropped1.to(device)

    # save(input_cropped1, "input_cropped_before_trans_trials")
    # IFPS of cropped ip
    input_cropped1 = torch.squeeze(input_cropped1, 1)
    # print("shape here ", input_cropped1.shape)

    # input_cropped1 = translate_pc(input_cropped1)

    # centroid of input_cropped1
    # print("HERE ", input_cropped1.shape)
    input_cropped1_centroid = centeroidnp(input_cropped_partial)  # TODO

    # computing the transformation mtx
    tx = tranformation_mtx(real_point_centroid, input_cropped1_centroid) # TODO

    # print(tx)

    # save(input_cropped1, "input_cropped1_before_tran")
    # save(real_center, "real_center_before_tran")
    # print(input_cropped_partial.shape, real_center.shape)
    # getting the transformed points
    real_center, input_cropped1 = final_t(tx, input_cropped_partial, real_center) # TODO

    # save(input_cropped1, "input_cropped1_after_tran")
    # save(real_center, "real_center_after_tran")
    # print("Centroid of input_cropped1 ", input_cropped1_centroid)

    # move the coordinate system to the new centroid

    # save(input_cropped1, "input_cropped_after_trans")

    # tx = tranformation_mtx(input_cropped1_centroid, real_point_centroid) # 4 x 4
    #
    # input_cropped1_trans, real_point_trans = final_t(tx, input_cropped1, real_point)

    input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
    input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)

    input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
    input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)

    input_cropped2 = input_cropped2.to(device)
    input_cropped3 = input_cropped3.to(device)

    input_cropped  = [input_cropped1, input_cropped2, input_cropped3]

    
#    fake,fake_part = point_netG(input_cropped)
    fake_center1, fake_center2, fake = point_netG(input_cropped)  # fakes in 3 scales
    # print("fake shape ", fake.shape)
    # save(fake.detach(), "fake_fine")

    fake_whole = torch.cat((input_cropped_partial, fake), 1)  # putting the whole fake together

    # save(fake_whole.detach(), "fake_whole")

    # save(real_center, "real_center_before")
    # print("real center shape ", real_center.shape)
    # real_center = translate_pc(real_center)

    # save(real_center, "real_center_after")

    fake_whole = fake_whole.to(device)
    real_point = real_point.to(device)  # real whole cloud
    real_center = real_center.to(device)  # target only

    dist_all, dist1, dist2 = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1))#+0.1*criterion_PointLoss(torch.squeeze(fake_part,1),torch.squeeze(real_center,1))
    dist_all=dist_all.cpu().detach().numpy()
    dist1 =dist1.cpu().detach().numpy()
    dist2 = dist2.cpu().detach().numpy()
    CD = CD + dist_all/length
    Gt_Pre = Gt_Pre + dist1/length
    Pre_Gt = Pre_Gt + dist2/length
    print(CD, Gt_Pre, Pre_Gt)



print(CD, Gt_Pre, Pre_Gt)
print("CD:{} , Gt_Pre:{} , Pre_Gt:{}".format(float(CD),float(Gt_Pre),float(Pre_Gt)))
print(length)    
    
    
