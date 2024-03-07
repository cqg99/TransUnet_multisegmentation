import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable as V
import cv2
import os
import math
import warnings
from tqdm import tqdm
import numpy as np
from data import ImageFolder
from models.networks.EESNet import get_transNet
import ttach as tta

BATCHSIZE_PER_CARD = 8

class TTAFrame():
    def __init__(self, net):
        # self.net = net(out_planes=1).cuda()
        self.net = net.cuda()
        # self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        # self.net = torch.nn.DataParallel(self.net, device_ids=[0])
    def test_one_img_from_path(self, path, evalmode = True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, img):
        # img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 - 1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask1 = np.argmax(mask1, axis=1)
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
    
        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        return mask2
    
    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 -1.6
        img6 = V(torch.Tensor(img6).cuda())
        
        maska = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3
    
    def test_one_img_from_path_1(self, img):
        # img = cv2.imread(path)#.transpose(2,0,1)[None]
        
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        
        mask = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        # self.net.load_state_dict(torch.load(path,map_location={'cuda:4':'cuda:0'}))

    def tta_use(self,img):
        #1 
        tta_model = tta.SegmentationTTAWrapper(self.net, tta.aliases.flip_transform(), merge_mode='mean')

        #2 time long
        # transforms = tta.Compose(
        #     [
        #         tta.HorizontalFlip(),
        #         tta.Rotate90(angles=[0, 180]),
        #         # tta.Scale(scales=[1, 2, 4]),
        #         # tta.Multiply(factors=[0.9, 1, 1.1]),
        #     ]
        # )
        # tta_model = tta.SegmentationTTAWrapper(self.net, transforms, merge_mode='gmean') #mean/gmean/max/min/sum/tsharpen

        img = img.transpose(2,0,1)
        img = np.array(img, np.float32)/255.0 * 3.2 -1.6
        img = V(torch.Tensor(img).cuda())
        # print(img.shape)
        mask = tta_model.forward(img.unsqueeze(0)).squeeze().cpu().data.numpy()
        return mask


if __name__ == "__main__":    
    
    test_path = 'D:/segmentation/TransUnet_multi/dataset/PV/test/images'
    save_path = 'D:/segmentation/TransUnet_multi/dataset/PV/test/pre/'
    imgs = os.listdir(test_path)


    model_path = './weights/trans_cam_iou_best.th'
    net = get_transNet(4)
    solver = TTAFrame(net)
    solver.load(model_path)

    for img in tqdm(imgs,ncols=20,total=len(imgs)):
        img_path = os.path.join(test_path, img)
        im = cv2.imread(img_path)
        im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_NEAREST)
        # pre = solver.test_one_img_from_path_8(im)
        pre = solver.tta_use(im)
        pre = np.argmax(pre, axis=0)
        pre = cv2.resize(pre, (480,360), interpolation=cv2.INTER_NEAREST)
        save_out = os.path.join(save_path, img)
        cv2.imwrite(save_out, pre)