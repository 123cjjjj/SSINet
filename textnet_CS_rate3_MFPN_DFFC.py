
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.vgg import VggNet
from network.resnet import ResNet
from util.config import config as cfg
import time
from collections import OrderedDict
import torchvision
import math

from network.textnet_merge_fpn_DFFC import FPN

from network.textnet_cs_rate3_TASPP_add_pcon1 import RRGN



class TextNet(nn.Module):
    def __init__(self, backbone='vgg', is_training=True):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN(self.backbone_name, pre_train=is_training)
        self.rrgn = RRGN(16)

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict['model'])

    def forward(self, x):
        end = time.time()
        # up1, up2, up3, up4, up5 = self.fpn(x)
        feature_total, attention = self.fpn(x)
        mlt = torch.mul(feature_total, attention)
        out = torch.add(feature_total, mlt)
        b_time = time.time() - end

        end = time.time()
        predict_out = self.rrgn(out)
        iter_time = time.time() - end
        return predict_out, attention, b_time, iter_time
