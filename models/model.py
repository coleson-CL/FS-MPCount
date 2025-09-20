import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
from math import sqrt
import numpy as np
import math
from timm.layers import trunc_normal_, DropPath, to_2tuple

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
        bn=False,
        relu=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu is not None:
            y = self.relu(y)
        return y


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (
            1 - mix_factor.expand_as(fea2)
        )
        return out


class FCAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(FCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        # 一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input):  # torch.Size([3, 3, 320, 320])
        x = self.avg_pool(input)  # torch.Size([3, 3, 1, 1])
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(
            -1, -2
        )  # torch.Size([3, 3, 1])局部
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)  # torch.Size([3, 1, 3])全局
        out1 = (
            torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)
        )  # (1,64,1,1)
        # x1 = x1.transpose(-1, -2).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = (
            torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        # x1 = x1.unsqueeze(-1)
        # x2 = x2.transpose(-1, -2).unsqueeze(-1)
        # print(out1.shape, out2.shape, x1.shape, x2.shape)
        # out2 = self.fc(x)
        out2 = self.sigmoid(out2)
        out = self.mix(out1, out2)
        # out = self.mix(x1, x2)
        out = (
            self.conv1(out.squeeze(-1).transpose(-1, -2))
            .transpose(-1, -2)
            .unsqueeze(-1)
        )
        out = self.sigmoid(out)
        return input * out


class BasicConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        bias=True,
        norm=False,
        relu=True,
        transpose=False,
    ):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock_Conv, self).__init__()
        self.conv1 = BasicConv(
            in_channel, out_channel, kernel_size=3, stride=1, relu=True
        )
        self.trans_layer = BasicConv(
            out_channel, out_channel, kernel_size=3, stride=1, relu=False
        )
        self.conv2 = BasicConv(
            out_channel, out_channel, kernel_size=3, stride=1, relu=False
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans_layer(out)
        out = self.conv2(out)
        return out + x


class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = ResBlock_Conv(in_channel=nc, out_channel=nc)

    def forward(self, x):
        yy = self.block(x)
        return yy


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Frequency_Spectrum_Dynamic_Aggregation(nn.Module):
    def __init__(self, nc):
        super(Frequency_Spectrum_Dynamic_Aggregation, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            # SELayer(channel=nc),
            FCAttention(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0),
        )
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            # SELayer(channel=nc),
            FCAttention(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0),
        )

    def forward(self, x):
        ori_mag = torch.abs(x)
        ori_pha = torch.angle(x)
        mag = self.processmag(ori_mag)
        mag = ori_mag + mag
        pha = self.processpha(ori_pha)
        pha = ori_pha + pha
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out


class BidomainNonlinearMapping(nn.Module):
    def __init__(self, in_nc):
        super(BidomainNonlinearMapping, self).__init__()
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = Frequency_Spectrum_Dynamic_Aggregation(in_nc)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 1, 1, 0)

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm="backward")
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm="backward")

        xcat = torch.cat([x, x_freq_spatial], 1)
        x_out = self.cat(xcat)

        return x_out


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def get_conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    bias,
    attempt_use_lk_impl=True,
):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = (
        kernel_size[0] == kernel_size[1]
        and kernel_size[0] > 5
        and padding == (kernel_size[0] // 2, kernel_size[1] // 2)
    )

    # if attempt_use_lk_impl and need_large_impl:
    #     print('---------------- trying to import iGEMM implementation for large-kernel conv')
    #     try:
    #         from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
    #         print('---------------- found iGEMM implementation ')
    #     except:
    #         DepthWiseConv2dImplicitGEMM = None
    #         print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
    #     if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
    #             and out_channels == groups and stride == 1 and dilation == 1:
    #         print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
    #         return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """

    def __init__(
        self,
        channels,
        kernel_size,
        deploy=False,
        use_sync_bn=False,
        attempt_use_lk_impl=True,
    ):
        super().__init__()
        self.lk_origin = get_conv2d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=channels,
            bias=deploy,
            attempt_use_lk_impl=attempt_use_lk_impl,
        )
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError("Dilated Reparam Block requires kernel_size >= 5")

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(
                    "dil_conv_k{}_{}".format(k, r),
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=k,
                        stride=1,
                        padding=(r * (k - 1) + 1) // 2,
                        dilation=r,
                        groups=channels,
                        bias=False,
                    ),
                )
                self.__setattr__(
                    "dil_bn_k{}_{}".format(k, r),
                    get_bn(channels, use_sync_bn=use_sync_bn),
                )

    def forward(self, x):
        if not hasattr(self, "origin_bn"):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__("dil_conv_k{}_{}".format(k, r))
            bn = self.__getattr__("dil_bn_k{}_{}".format(k, r))
            out = out + bn(conv(x))
        return out

    def switch_to_deploy(self):
        if hasattr(self, "origin_bn"):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__("dil_conv_k{}_{}".format(k, r))
                bn = self.__getattr__("dil_bn_k{}_{}".format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(
                origin_k.size(0),
                origin_k.size(0),
                origin_k.size(2),
                stride=1,
                padding=origin_k.size(2) // 2,
                dilation=1,
                groups=origin_k.size(0),
                bias=True,
                attempt_use_lk_impl=self.attempt_use_lk_impl,
            )
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__("origin_bn")
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__("dil_conv_k{}_{}".format(k, r))
                self.__delattr__("dil_bn_k{}_{}".format(k, r))


def upsample(x, scale_factor=2, mode="bilinear"):
    if mode == "nearest":
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)
    else:
        return F.interpolate(
            x, scale_factor=scale_factor, mode=mode, align_corners=False
        )


class DGModel_base(nn.Module):
    def __init__(self, pretrained=True, den_dropout=0.5):
        super().__init__()

        self.den_dropout = den_dropout

        # vgg = models.vgg16_bn(pretrained=True,weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
        # vgg = models.vgg16_bn(pretrained=True) if pretrained else models.vgg16_bn()
        # vgg = models.vgg16_bn(pretrained=False)
        # vgg.load_state_dict(torch.load('path/to/your/vgg16_bn.pth'))
        vgg = models.vgg16_bn(
            weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
        )
        self.enc1 = nn.Sequential(
            *list(vgg.features.children())[:23]
        )  # torch.Size([8, 256, 80, 80])
        self.enc2 = nn.Sequential(
            *list(vgg.features.children())[23:33]
        )  # torch.Size([8, 512, 40, 40])
        self.enc3 = nn.Sequential(
            *list(vgg.features.children())[33:43]
        )  # torch.Size([8, 512, 20, 20])
        self.reparam = DilatedReparamBlock(512, 7, deploy=False)
        self.dec3 = nn.Sequential(
            ConvBlock(512, 1024, bn=True), ConvBlock(1024, 512, bn=True)
        )

        self.dec2 = nn.Sequential(
            ConvBlock(1024, 512, bn=True), ConvBlock(512, 256, bn=True)
        )

        self.dec1 = nn.Sequential(
            ConvBlock(512, 256, bn=True), ConvBlock(256, 128, bn=True)
        )

        self.den_dec = nn.Sequential(
            ConvBlock(512 + 256 + 128, 256, kernel_size=1, padding=0, bn=True),
            nn.Dropout2d(p=den_dropout),
        )

        self.den_head = nn.Sequential(ConvBlock(256, 1, kernel_size=1, padding=0))
        self.BNM = BidomainNonlinearMapping(in_nc=512)

    def forward_fe(self, x):
        x1 = self.enc1(x)  # torch.Size([8, 256, 80, 80])
        x2 = self.enc2(x1)  # torch.Size([8, 512, 40, 40])
        x3 = self.enc3(x2)  # torch.Size([8, 512, 20, 20])
        x3 = self.BNM(x3)
        x3 = self.reparam(x3)
        x = self.dec3(x3)
        y3 = x
        x = upsample(x, scale_factor=2)  # x: torch.Size([8, 512, 40, 40])
        x = torch.cat([x, x2], dim=1)

        x = self.dec2(x)
        y2 = x
        x = upsample(x, scale_factor=2)  # x: torch.Size([8, 256, 80, 80])
        x = torch.cat([x, x1], dim=1)  # x: torch.Size([8, 512, 80, 80])

        x = self.dec1(x)
        y1 = x

        y2 = upsample(y2, scale_factor=2)
        y3 = upsample(y3, scale_factor=4)

        y_cat = torch.cat([y1, y2, y3], dim=1)  # [8, 896, 80, 80]

        return y_cat, x3

    def forward(self, x):
        y_cat, _ = self.forward_fe(x)

        y_den = self.den_dec(y_cat)
        d = self.den_head(y_den)
        d = upsample(d, scale_factor=4)

        return d

    def den_FDIT(self, distance_map):
        distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
        distance_map = distance_map.numpy()
        distance_map[distance_map < 1e-2] = 0


class DGModel_mem(DGModel_base):
    def __init__(self, pretrained=True, mem_size=1024, mem_dim=256, den_dropout=0.5):
        super().__init__(pretrained, den_dropout)

        self.mem_size = mem_size
        self.mem_dim = mem_dim

        self.mem = nn.Parameter(
            torch.FloatTensor(1, self.mem_dim, self.mem_size).normal_(0.0, 1.0)
        )

        self.den_dec = nn.Sequential(  # Conv1*1
            ConvBlock(512 + 256 + 128, self.mem_dim, kernel_size=1, padding=0, bn=True),
            nn.Dropout2d(p=den_dropout),
        )
        # 定义一个预测头
        self.den_head = nn.Sequential(
            ConvBlock(self.mem_dim, 1, kernel_size=1, padding=0)
        )

    def forward_mem(self, y):
        b, k, h, w = y.shape  # k:256
        m = self.mem.repeat(b, 1, 1)
        m_key = m.transpose(1, 2)  # (b, 1024, 256)
        y_ = y.view(b, k, -1)  # (b,256,32*32)
        logits = torch.bmm(m_key, y_) / sqrt(k)  # (b, 1024, 1024)
        y_new = torch.bmm(
            m_key.transpose(1, 2), F.softmax(logits, dim=1)
        )  # (b,256,1024)
        y_new_ = y_new.view(b, k, h, w)

        return y_new_, logits

    def forward(self, x):
        y_cat, _ = self.forward_fe(x)
        y_den = self.den_dec(y_cat)
        y_den_new, _ = self.forward_mem(y_den)
        d = self.den_head(y_den_new)

        d = upsample(d, scale_factor=4)
        return d


class DGModel_memcls(DGModel_mem):
    def __init__(
        self,
        pretrained=True,
        mem_size=1024,
        mem_dim=256,
        den_dropout=0.5,
        cls_dropout=0.5,
        cls_thrs=0.5,
    ):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout)

        self.cls_dropout = cls_dropout
        self.cls_thrs = cls_thrs

        self.cls_head = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            nn.Dropout2d(p=self.cls_dropout),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid(),
        )

    def transform_cls_map_gt(self, c_gt):
        return upsample(c_gt, scale_factor=4, mode="nearest")

    def transform_cls_map_pred(self, c):
        c_new = c.clone().detach()
        c_new[c < self.cls_thrs] = 0
        c_new[c >= self.cls_thrs] = 1
        c_resized = upsample(c_new, scale_factor=4, mode="nearest")

        return c_resized

    def transform_cls_map(self, c, c_gt=None):
        if c_gt is not None:
            return self.transform_cls_map_gt(c_gt)
        else:
            return self.transform_cls_map_pred(
                c
            )  # 对c进行二值化，得到一个根据阈值划分的0 1矩阵

    def forward(self, x, c_gt=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y_cat, x3 = self.forward_fe(x)

        y_den = self.den_dec(y_cat)
        y_den_new, _ = self.forward_mem(y_den)  # ~F

        c = self.cls_head(x3)
        c_resized = self.transform_cls_map(c, c_gt)
        d = self.den_head(y_den_new)
        # fdit=self.den_FDIT(d)

        dc = d * c_resized  # 得到Dori或者Daug
        dc = upsample(dc, scale_factor=4)

        return dc, c


class DGModel_final(DGModel_memcls):
    def __init__(
        self,
        pretrained=True,
        mem_size=1024,
        mem_dim=256,
        cls_thrs=0.5,
        err_thrs=0.5,
        den_dropout=0.5,
        cls_dropout=0.5,
        has_err_loss=False,
    ):
        super().__init__(
            pretrained, mem_size, mem_dim, den_dropout, cls_dropout, cls_thrs
        )

        self.err_thrs = err_thrs
        self.has_err_loss = has_err_loss

        self.den_dec = nn.Sequential(
            ConvBlock(512 + 256 + 128, self.mem_dim, kernel_size=1, padding=0, bn=True)
        )

    def jsd(self, logits1, logits2):
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        jsd = F.mse_loss(p1, p2)
        return jsd

    # def jsd(self, logits1, logits2):
    #     sigmoid1 = F.sigmoid(logits1)
    #     sigmoid2 = F.sigmoid(logits2)
    #     # 根据阈值 0.6 生成二值图像
    #     X1 = (sigmoid1 > 0.6).float()
    #     X2 = (sigmoid2 > 0.6).float()
    #     intersection = torch.sum(X1 * X2)
    #     # 计算两个图的并集
    #     union = torch.sum(X1) + torch.sum(X2)
    #     # 计算Dice相似度
    #     dice = 1-((2. * intersection + 1) / (union + 1))
    #     return dice

    def forward_train(self, img1, img2, c_gt=None):
        # print("img1 shape:", img1.shape)
        y_cat1, x3_1 = self.forward_fe(img1)
        y_cat2, x3_2 = self.forward_fe(img2)
        y_den1 = self.den_dec(y_cat1)
        y_den2 = self.den_dec(y_cat2)
        y_in1 = F.instance_norm(y_den1, eps=1e-5)
        y_in2 = F.instance_norm(y_den2, eps=1e-5)

        e_y = torch.abs(y_in1 - y_in2)

        e_mask = (e_y < self.err_thrs).clone().detach()
        loss_err = F.l1_loss(y_in1, y_in2) if self.has_err_loss else 0
        y_den_masked1 = F.dropout2d(y_den1 * e_mask, self.den_dropout)
        y_den_masked2 = F.dropout2d(y_den2 * e_mask, self.den_dropout)

        y_den_new1, logits1 = self.forward_mem(y_den_masked1)
        y_den_new2, logits2 = self.forward_mem(y_den_masked2)
        loss_con = self.jsd(logits1, logits2)  # 表示两个分支提取域不变特征的差异损失

        c1 = self.cls_head(x3_1)
        c2 = self.cls_head(x3_2)

        c_resized_gt = self.transform_cls_map_gt(
            c_gt
        )  # c_gt为Noen不执行任何操作，不为None时执行上采样
        c_resized1 = self.transform_cls_map_pred(c1)  # 通过PC Head得到Mask
        c_resized2 = self.transform_cls_map_pred(c2)
        c_err = torch.abs(
            c_resized1 - c_resized2
        )  # 若是1则表示两个分支中有一个分类为包含头，若为0表示两个分支预测是一样的，同时有头或无头
        c_resized = torch.clamp(c_resized_gt + c_err, 0, 1)

        d1 = self.den_head(y_den_new1)
        d2 = self.den_head(y_den_new2)

        dc1 = upsample(d1 * c_resized, scale_factor=4)
        dc2 = upsample(d2 * c_resized, scale_factor=4)
        # print("dc1.shape",dc1.shape)
        c_err = upsample(c_err, scale_factor=4)

        return (
            dc1,
            dc2,
            c1,
            c2,
            c_err,
            loss_con,
            loss_err,
        )  # dc1, dc2是D’，经过分类Mask遮掩后的密度图；c1, c2；loss_err平均绝对误差损失
