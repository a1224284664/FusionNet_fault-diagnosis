import resnet as rn
import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
from gltb import GlobalLocalEnhance
# from slide import SlideAttention
from sparse import SparseConvBlockWithMixedPooling
class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out
    

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)
    

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )
class ResNet(torch.nn.Module):
    def __init__(self, channels=3, output_nc=2,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True, learnable = False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1

        self.resnet = rn.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
           

        self.relu = nn.ReLU()


        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex6 = nn.Upsample(scale_factor=0.5)
        
        self.learnable = learnable 
        
        self.upsamplex2l1_single = nn.ConvTranspose2d(256*expand, 256*expand, 4, 2, 1)
        self.adjust_conv1 = nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1)
        self.adjust_conv2 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.adjust_conv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.adjust_conv4 = nn.ConvTranspose2d(1536, 768, kernel_size=3, stride=1, padding=1)
        
        if self.learnable:
            self.upsamplex2l1 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
            self.upsamplex2l2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.conv_pred1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)



        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()
        self.active3d = nn.Tanh()


    def forward_single(self, img):
        device = 'cuda'
        ###ResNet
        x1 = self.resnet.conv1(img)
        x1 = self.resnet.bn1(x1)
        x1 = self.resnet.relu(x1)
        xp11 = self.resnet.maxpool(x1)
        xp12 = self.resnet.layer1(xp11)
        xp13 = self.resnet.layer2(xp12)
        xp14 = self.resnet.layer3(xp13)
    #     # print(f"xp14: {xp14.shape}")
        xp14 = self.adjust_conv1(xp14)  
        # print(f"xp14: {xp14.shape}")
        xp14 = self.upsamplex4(xp14)
        # print(f"xp14 adjusted: {xp14.shape}")


        xp13 = self.adjust_conv2(xp13) 
        # print(f"xp13: {xp13.shape}")
        xp13 = self.upsamplex4(xp13)
        # print(f"xp13: {xp13.shape}")

        xp12 = self.adjust_conv3(xp12)  
        xp12 = self.upsamplex2(xp12)
        # print(f"xp12 adjusted: {xp12.shape}")

        xp11 = self.adjust_conv3(xp11) 
        # print(f"xp11: {xp11.shape}")
        xp11 = self.upsamplex2(xp11)
        # print(f"xp11: {xp11.shape}")

        return xp14, xp13, xp12, xp11

class SwinTransformer(ResNet):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=10, head_dim=32, window_size=7,resnet_stages_num=5,backbone='resnet',
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__(backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,)
        self.block =  GlobalLocalEnhance(in_channels=768, num_heads=8, reduction=8)
        self.sparse =SparseConvBlockWithMixedPooling(in_channels=channels, out_channels=128)
        # self.slide_1 = SlideAttention(input_resolution=(56, 56), dim=64, num_heads=8, ka=1, padding_mode='zeros')
        # self.slide_2 = SlideAttention(input_resolution=(28, 28), dim=64, num_heads=8, ka=3, padding_mode='zeros')
        # self.slide_3 = SlideAttention(input_resolution=(28, 28), dim=128, num_heads=8, ka=3, padding_mode='zeros')
        # self.slide_4 = SlideAttention(input_resolution=(28, 28), dim=1256, num_heads=8, ka=3, padding_mode='zeros')
        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward_swin(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x= self.stage3(x)
        x = self.stage4(x)
        return x
    def forward(self, img):
        # print(f"img: {img.shape}")
        x1, x2, x3, x4 = self.forward_single(img)
        skip = self.sparse(img)
        # print(f"x1: {x1.shape}")
        x1 = self.forward_swin(x1)
        x2 = self.forward_swin(x2)
        x3 = self.forward_swin(x3)
        x4 = self.forward_swin(x4)
        skip = self.forward_swin(skip)
        
        x = self.block(x1, x2, x3, x4, skip)
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)



def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), resnet_stages_num=4,**kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads,  resnet_stages_num=resnet_stages_num,**kwargs)


# input=torch.rand(1,3,224,224)
# model=swin_t()
# out=model(input)
# print(out.shape)