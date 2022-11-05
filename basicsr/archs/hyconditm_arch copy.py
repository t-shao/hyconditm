import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, make_layer
from .condition_module import ConditionedTransform
from .rrdbnet_arch import ResidualDenseBlock, RRDB


class HybridConditionModule(nn.Module):
    def __init__(self, in_channels, out_channels, global_cond_channels, init_mid_channels=16,
                 down_method='stride', up_method='bilinear'):
        super().__init__()

        self.in_conv = HyCondModConvBlock(in_channels, init_mid_channels)                           # in_channels -> 16
        self.enc_1 = HyCondModEncBlock(init_mid_channels, init_mid_channels * 2, down_method)       # 16 -> 32  1/2
        self.enc_2 = HyCondModEncBlock(init_mid_channels * 2, init_mid_channels * 4, down_method)   # 32 -> 64  1/4
        self.enc_3 = HyCondModEncBlock(init_mid_channels * 4, init_mid_channels * 8, down_method)   # 64 -> 128  1/8
        self.global_cond = HyCondModGlobalConditionBlock(init_mid_channels * 8, global_cond_channels)  # 128 -> 64
        self.dec_1 = HyCondModDecBlock(init_mid_channels * 8, init_mid_channels * 4, up_method)     # 128 -> 64  1/4
        self.dec_2 = HyCondModDecBlock(init_mid_channels * 4, init_mid_channels * 2, up_method)     # 64 -> 32  1/2
        self.dec_3 = HyCondModDecBlock(init_mid_channels * 2, init_mid_channels, up_method)         # 32 -> 16  1
        self.out_conv = HyCondModConvBlock(init_mid_channels, out_channels)                         # 16 -> out_channels

    def forward(self, x):
        x_1 = self.in_conv(x)       # 16
        x_2 = self.enc_1(x_1)       # 32
        x_3 = self.enc_2(x_2)       # 64
        x_4 = self.enc_3(x_3)       # 128
        z = self.global_cond(x_4)   # global_cond_channels
        y = self.dec_1(x_4, x_3)    # 64
        y = self.dec_2(y, x_2)      # 32
        y = self.dec_3(y, x_1)      # 16
        y = self.out_conv(y)        # out_channels

        return z, y


class HyCondModConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class HyCondModEncBlock(nn.Module):
    """
        input: (N, in_channels, H, W)
        output: (N, out_channels, H / 2, W / 2)
    """
    def __init__(self, in_channels, out_channels, downscale_method='stride'):
        super().__init__()

        if downscale_method == 'stride':
            self.down = HyCondModConvBlock(in_channels, out_channels, stride=2)
        elif downscale_method == 'pool':
            self.down = nn.Sequential(
                nn.MaxPool2d(2),
                HyCondModConvBlock(in_channels, out_channels)
            )
        else:
            raise NotImplementedError

        self.conv = HyCondModConvBlock(out_channels, out_channels)

    def forward(self, x):
        return self.conv(self.down(x))


class HyCondModDecBlock(nn.Module):
    """
        input: (N, in_channels, H, W)
        output: (N, out_channels, 2 * H, 2 * W)
    """
    def __init__(self, in_channels, out_channels, upscale_method='bilinear'):
        super().__init__()

        if upscale_method == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                HyCondModConvBlock(in_channels, out_channels)
            )
        elif upscale_method == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            raise NotImplementedError

        self.conv = HyCondModConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class HyCondModGlobalConditionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.cond = nn.Sequential(
            HyCondModConvBlock(in_channels, out_channels, kernel_size=1, padding=0),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.cond(x)


class SpatialTransformBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels):
        super().__init__()

        self.sft = ConditionedTransform(in_channels, out_channels, cond_channels, 'spatial')
        self.rdb = ResidualDenseBlock(out_channels)

    def forward(self, x, cond):
        y = self.sft(x, cond)
        y = self.rdb(y)

        return y


class RefinementBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_features, num_blocks=3):
        super().__init__()

        self.conv_first = nn.Conv2d(in_channels, n_features, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_blocks, num_feat=n_features, pytorch_init=True)
        self.conv_last = nn.Conv2d(n_features, out_channels, 3, 1, 1)

    def forward(self, x):
        res = self.conv_last(self.body(self.conv_first(x)))

        return x + res


@ARCH_REGISTRY.register()
class HyCondITMv1(nn.Module):
    def __init__(self, in_channels, transform_channels, global_cond_channels, spatial_cond_channels):
        super().__init__()

        self.cond_net = HybridConditionModule(in_channels, spatial_cond_channels, global_cond_channels)

        self.global_transform_1 = ConditionedTransform(
            in_channels, transform_channels, global_cond_channels, 'global')
        self.global_transform_2 = ConditionedTransform(
            transform_channels, transform_channels, global_cond_channels, 'global')
        self.global_transform_3 = ConditionedTransform(
            transform_channels, in_channels, global_cond_channels, 'global', activation=False)

        '''
        self.spatial_transform_1 = SpatialTransformBlock(in_channels, transform_channels, spatial_cond_channels)
        self.spatial_transform_2 = SpatialTransformBlock(transform_channels, transform_channels, spatial_cond_channels)
        self.spatial_transform_3 = SpatialTransformBlock(transform_channels, transform_channels, spatial_cond_channels)

        self.refinement = nn.Sequential(
            nn.Conv2d(transform_channels, transform_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(transform_channels, 3, 1)
        )
        '''

        self.spatial_transform_1 = ConditionedTransform(
            in_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        self.spatial_transform_2 = ConditionedTransform(
            transform_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        self.spatial_transform_3 = ConditionedTransform(
            transform_channels, in_channels, spatial_cond_channels, 'spatial', ada_method='vanilla', activation=False)

        self.refinement = RefinementBlock(in_channels, in_channels, transform_channels)

    def forward(self, x):
        global_cond, spatial_cond = self.cond_net(x)

        coarsely_tuned_x = self.global_transform_1(x, global_cond)
        coarsely_tuned_x = self.global_transform_2(coarsely_tuned_x, global_cond)
        coarsely_tuned_x = self.global_transform_3(coarsely_tuned_x, global_cond)

        spatially_modulated_x = self.spatial_transform_1(coarsely_tuned_x, spatial_cond)
        spatially_modulated_x = self.spatial_transform_2(spatially_modulated_x, spatial_cond)
        spatially_modulated_x = self.spatial_transform_3(spatially_modulated_x, spatial_cond)

        return self.refinement(spatially_modulated_x), coarsely_tuned_x
