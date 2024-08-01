import torch
import torch.nn.functional as F
from torch.autograd import Variable
import imageio
import numpy as np
import os, argparse
from datetime import datetime
from skimage import img_as_ubyte
from model.RAGRNet import RAGRNet
from utils.data import get_loader
from utils.data import test_dataset
from utils.func import label_edge_prediction, AvgMeter, clip_gradient, adjust_lr
import time

import pytorch_iou
import pytorch_fm

torch.cuda.set_device(0)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average=True)
floss = pytorch_fm.FLoss()
size_rates = [0.75, 1, 1.25]


def _get_adaptive_threshold(matrix, max_value = 1):
    """
    Return an adaptive threshold, which is equal to twice the mean of ``matrix``.
    :param matrix: a data array
    :param max_value: the upper limit of the threshold
    :return: min(2 * matrix.mean(), max_value)
    """
    return min(2 * matrix.mean(), max_value)

def cal_adaptive_fm(pred, gt):
    """
    Calculate the adaptive F-measure.
    :return: adaptive_fm
    """
    # ``np.count_nonzero`` is faster and better
    beta = 0.3
    adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
    binary_predcition = (pred >= adaptive_threshold).astype(np.float32)
    area_intersection = np.count_nonzero(binary_predcition * gt)
    if area_intersection == 0:
        adaptive_fm = 0
    else:
        pre = area_intersection * 1.0 / np.count_nonzero(binary_predcition)
        rec = area_intersection * 1.0 / np.count_nonzero(gt)
        adaptive_fm = (1 + beta) * pre * rec / (beta * pre + rec)
    return adaptive_fm

def run(train_i):
    best_adp_fm = 0
    best_mae = 1
    best_epoch = 0
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        model.train()
        loss_record1, loss_record2, loss_record3, loss_record4, loss_record5, loss_record6 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # edge prediction
                edges = label_edge_prediction(gts)

                # multi-scale training samples
                trainsize = int(round(opt.trainsize * rate / 32) * 32)

                if rate != 1:
                    images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    edges = F.interpolate(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                s1, s1_sig, edg1, s2, s2_sig, edg2, s3, s3_sig, edg3, s4, s4_sig, edg4, edgesal, graphsal, graphsal_sig, g1sal, g1sal_sig = model(images)
                loss1 = CE(s1, gts) + IOU(s1_sig, gts) + floss(s1_sig, gts) + CE(edg1, edges)
                loss2 = CE(s2, gts) + IOU(s2_sig, gts) + floss(s2_sig, gts) + CE(edg2, edges)
                loss3 = CE(s3, gts) + IOU(s3_sig, gts) + floss(s3_sig, gts) + CE(edg3, edges)
                loss4 = CE(s4, gts) + IOU(s4_sig, gts) + floss(s4_sig, gts) + CE(edg4, edges)
                loss5 = CE(graphsal, gts) + IOU(graphsal_sig, gts) + floss(graphsal_sig, gts) + CE(edgesal, edges)
                loss6 = CE(g1sal, gts) + IOU(g1sal_sig, gts) + floss(g1sal_sig, gts)


                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

                loss.backward()

                clip_gradient(optimizer, opt.clip)
                optimizer.step()

                if rate == 1:
                    loss_record1.update(loss1.data, opt.batchsize)
                    loss_record2.update(loss2.data, opt.batchsize)
                    loss_record3.update(loss3.data, opt.batchsize)
                    loss_record4.update(loss4.data, opt.batchsize)
                    loss_record5.update(loss5.data, opt.batchsize)
                    loss_record6.update(loss6.data, opt.batchsize)

            if i % 100 == 0 or i == total_step:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Lossgraph: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data,
                           loss2.data, loss5.data))

        save_path = 'models/Your_Files/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        res_save_path = save_path + 'salmap/'
        if not os.path.exists(res_save_path):
            os.makedirs(res_save_path)

        # test
        with torch.no_grad():
            model.eval()
            time_sum = 0
            adaptive_fms = 0.0
            mae = 0.0
            for i in range(test_loader.size):
                image, gt, name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                time_start = time.time()
                res, s1_sig, e1, s2, s2_sig, e2, s3, s3_sig, e3, s4, s4_sig, e4, edgesal, graphsal, graphsal_sig, g1sal, g1sal_sig = model(image)
                time_end = time.time()
                time_sum = time_sum + (time_end - time_start)
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                adaptive_fm = cal_adaptive_fm(pred=res, gt=gt)
                # adaptive_fm = 0
                adaptive_fms += adaptive_fm
                mae += np.sum(np.abs(gt - res)) / (gt.shape[0] * gt.shape[1])
                imageio.imsave(res_save_path + name, img_as_ubyte(res))

            print('FPS {:.5f}'.format(test_loader.size / time_sum))
            torch.save(model.state_dict(), save_path + 'SEI_ORSSD.pth' + '.%d' % epoch)
            adp_fm = adaptive_fms / test_loader.size
            mae_mean = mae / test_loader.size
            if mae_mean < best_mae:
                best_adp_fm = adp_fm
                best_mae = mae_mean
                best_epoch = epoch
                print('Epoch [{:03d}], best_adp_fm {:.4f}, best_mae {:.4f}'.format(epoch, best_adp_fm, best_mae))
            print('Current_epoch [{:03d}], adp_fm {:.4f}, mae {:.4f}'.format(epoch, adp_fm, mae_mean))
            print('Best_epoch [{:03d}], best_adp_fm {:.4f}, best_mae {:.4f}'.format(best_epoch, best_adp_fm, best_mae))

print("Let's go!")
for train_i in range(0, 1):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    opt = parser.parse_args()

    # build train_models
    model = RAGRNet()
    model.cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)
    image_root = '/Your_File/SOD/train_dataset/EORSSD/Images/'
    gt_root = '/Your_File/SOD/train_dataset/EORSSD/GT/'
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    # build test_models
    test_dataset_path = '/Your_File/SOD/test_dataset/'
    # test_datasets = 'ORSSD'
    test_datasets = 'EORSSD'
    test_image_root = test_dataset_path + test_datasets + '/Images/'
    test_gt_root = test_dataset_path + test_datasets + '/GT/'
    test_loader = test_dataset(test_image_root, test_gt_root, opt.trainsize)

    print('Statr {}-th training!!!'.format(train_i))
    print('Learning Rate: {}'.format(opt.lr))

    run(train_i)

