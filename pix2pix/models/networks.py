
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math
import torch.nn.functional as F
import numpy as np

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_NET(net, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    if net == 'search':
        net = AD2ATTACK()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


class AD2ATTACK(nn.Module):
    def __init__(self):
        super(AD2ATTACK, self).__init__()
        self.model = SR_Upsamling()
    def forward(self, input, block_num):
        return self.model(input, block_num)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class RSE(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(RSE, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, padding=5 // 2, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 5, padding=5 // 2, groups=group),
            nn.ReLU(inplace=True),
        )
        self.spa = SpatialAttention(7)
        self.end = nn.Conv2d(out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        out = self.body(x)
        out = out * self.spa(out)
        out = self.end(out)
        out = F.relu(out + x)
        return out


class FeatureExtraction(nn.Module):
    def __init__(self, level):
        super(FeatureExtraction, self).__init__()
        if level == 1:
            self.conv0 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1)  # X1

        self.conv1 = RSE(64, 64)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2), nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2), nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2), nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2), nn.LeakyReLU(negative_slope=0.2))
        self.convt_G = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.convt_F = nn.Sequential(self.convt_G, nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.convt_F(out)
        return out


class FeatureExtraction2(nn.Module):
    def __init__(self, level):
        super(FeatureExtraction2, self).__init__()
        self.conv1 = RSE(64, 64)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2), nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2), nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2), nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2), nn.LeakyReLU(negative_slope=0.2))
        self.convt_G = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.convt_F = nn.Sequential(self.convt_G, nn.LeakyReLU(negative_slope=0.2))



    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.convt_F(out)
        return out
class FeatureExtraction3(nn.Module):
    def __init__(self, level):
        super(FeatureExtraction3, self).__init__()
        self.conv1 = RSE(64, 64)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2), nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2), nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2), nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2), nn.LeakyReLU(negative_slope=0.2))
        self.convt_G = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.convt_F = nn.Sequential(self.convt_G, nn.LeakyReLU(negative_slope=0.2))



    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.convt_F(out)
        return out

class ImageReconstruction(nn.Module):
    def __init__(self):
        super(ImageReconstruction, self).__init__()
        self.conv_R = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
    def forward(self, LR, convt_F):
        conv_R = self.conv_R(convt_F)
        HR =  torch.nn.functional.interpolate(LR, scale_factor=2, mode='bilinear') + conv_R
        return HR


class SR_Upsamling(nn.Module):

    def __init__(self):
        super(SR_Upsamling, self).__init__()
        self.FeatureExtraction1 = FeatureExtraction(level=1)
        self.FeatureExtraction2 = FeatureExtraction2(level=2)
        self.FeatureExtraction3 = FeatureExtraction3(level=3)
        self.ImageReconstruction1 = ImageReconstruction()
        self.ImageReconstruction2 = ImageReconstruction()
        self.ImageReconstruction3 = ImageReconstruction()
    def forward(self, x , block_num):
        if block_num == 1:
            convt_F1 = self.FeatureExtraction1(x)
            HR_2 = self.ImageReconstruction1(x, convt_F1)
            return HR_2
        elif block_num == 2:
            convt_F1 = self.FeatureExtraction1(x)
            HR_2 = self.ImageReconstruction1(x, convt_F1)
            convt_F2 = self.FeatureExtraction2(convt_F1)
            HR_4 = self.ImageReconstruction2(HR_2, convt_F2)
            return  HR_4
        elif block_num ==3:
            convt_F1 = self.FeatureExtraction1(x)
            HR_2 = self.ImageReconstruction1(x, convt_F1)
            convt_F2 = self.FeatureExtraction2(convt_F1)
            HR_4 = self.ImageReconstruction2(HR_2, convt_F2)
            convt_F3 = self.FeatureExtraction3(convt_F2)
            HR_8 = self.ImageReconstruction3(HR_4, convt_F3)
            return HR_8





