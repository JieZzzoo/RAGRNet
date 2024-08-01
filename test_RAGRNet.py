import torch
import torch.nn.functional as F

import numpy as np
import os, argparse
import time
import imageio
from skimage import img_as_ubyte

from model.RAGRNet import RAGRNet
from utils.data import test_dataset

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')  # The test size of SEINet_ResNet50 and SEINet2_ResNet50 is 352
opt = parser.parse_args()

dataset_path = '/Your_File/SOD/test_dataset/'

model = RAGRNet()
model.load_state_dict(torch.load('/Your_pth.pth'))

model.cuda()
model.eval()

# test_datasets = ['ORSSD']
test_datasets = ['EORSSD']

for dataset in test_datasets:
    save_path = './results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/Images/'
    print(dataset)
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        res, s1_sig, e1, s2, s2_sig, e2, s3, s3_sig, e3, s4, s4_sig, e4, edgesal, graphsal, graphsal_sig, g1sal, g1sal_sig = model(image)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imsave(save_path+name, img_as_ubyte(res))
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size/time_sum))
