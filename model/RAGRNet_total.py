import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
from model.Res2Net import res2net50_v1b_26w_4s
BatchNorm = nn.BatchNorm2d

class ConvEncoder(nn.Module):
    def __init__(self, n, channels):
        super(ConvEncoder, self).__init__()
        self.indentity = nn.Identity()
        self.stem_conv = DSCConv(channels, channels, (n, n), padding=(n//2, n//2))
        self.norm = nn.LayerNorm(channels)
        self.linear1 = nn.Conv2d(channels, channels*4, kernel_size=1)
        self.gelu = nn.GELU()
        self.linear2 = nn.Conv2d(channels*4, channels, kernel_size=1)

    def forward(self, x):
        indentity = self.indentity(x)
        x = self.stem_conv(x)
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.linear2(self.gelu(self.linear1(x)))
        out = x + indentity

        return out

class SEWeightModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DSCConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super(DSCConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    groups=in_ch)

        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    groups=1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)

        return out


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):

        return self.reduce(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)

        return self.sigmoid(x2)


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))

        return h


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1, bias=None, modulation=False,act="silu"):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
#------------------------------------------------------------------
        self.bn = nn.BatchNorm2d(outc)
        self.act = get_activation(act, inplace=True)
#------------------------------------------------------------------
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        p = self._get_p(offset, dtype)

        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.act(self.bn(self.conv(x_offset)))

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N]*padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class PSAModule(nn.Module):
    def __init__(self, inplans, planes, conv_kernels=[1,3,5,7], stride=1, conv_groups=[1,2,4,8]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

        self.branch0 = nn.Sequential(
            BasicConv2d(planes//4, planes//4, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(planes//4, planes//4, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(planes // 4, planes // 4, 3, padding=1, dilation=1),
            BasicConv2d(planes//4, planes//4, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(planes//4, planes//4, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(planes // 4, planes // 4, 5, padding=2, dilation=1),
            BasicConv2d(planes//4, planes//4, kernel_size=(5, 1), padding=(2, 0))

        )
        self.branch3 = nn.Sequential(
            BasicConv2d(planes//4, planes//4, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(planes//4, planes//4, 7, padding=3, dilation=1),
            BasicConv2d(planes//4, planes//4, kernel_size=(7, 1), padding=(3, 0))
        )
        self.conv_cat = nn.Conv2d(4 * planes//4, planes//4, 3, padding=1)
        self.aggr = nn.Conv2d(planes, planes, 1, 1, 0)
        self.act = nn.GELU()
    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])
        x1_se = self.branch0(x1)
        x2_se = self.branch1(x2+x1_se)
        x3_se = self.branch2(x3+x2_se)
        x4_se = self.branch3(x4+x3_se)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        # x_se = self.conv_cat(x_se)
        x_se = self.se(x_se)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        out = self.act(self.aggr(out)) * x

        return out


class ChannelAtt(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAtt, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)

        return x

class RAGR(nn.Module):
    def __init__(self, num_in, plane_mid, mids, abn=BatchNorm, normalize=False):
        super(RAGR, self).__init__()
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = abn(num_in)

        self.local1 = nn.Sequential(
            nn.Conv2d(num_in, num_in, kernel_size=3),
            abn(num_in),
            nn.ReLU(inplace=True)
        )
        self.local2 = nn.Sequential(
            nn.Conv2d(num_in, num_in, kernel_size=5, padding=3),
            abn(num_in),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Conv2d(2*num_in, num_in, 1)

    def forward(self, x, edge):
        x = F.upsample(x, (edge.size()[-2], edge.size()[-2]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)
        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + self.blocker(self.conv_extend(x_state))

        xlocal = self.local2(self.local1(x))
        out = torch.cat((out, xlocal), dim=1)
        out = self.conv_cat(out)

        return out

class CPM(nn.Module):
    def __init__(self, channel):
        super(CPM, self).__init__()
        self.ms = PSAModule(channel, channel)
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

        self.edg_conv = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.sal_conv = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, sal, edg):
        sal_r = self.ms(sal)
        sal_c = self.ca(sal_r) * sal_r
        sal_A = self.sa(sal_c) * sal_c
        edg_s = self.sigmoid(edg) * edg
        edg_o = self.edg_conv(edg_s * sal_A)
        sal_o = self.sal_conv(torch.cat((sal_A, edg_s), 1))

        return (self.sigmoid(sal) + sal_o), (self.sigmoid(edg) + edg_o)

class GIA(nn.Module):
    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()
        self.C = in_channels
        self.O = out_channels
        assert in_channels == out_channels
        self.ca = ChannelAtt(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputsa):
        inputs = self.conv(inputsa)
        inputs = self.act(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs
        inputs = inputs + inputsa

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)

        return out


class MRA(nn.Module):
    def __init__(self, channel, abn=BatchNorm):
        super(MRA, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.S_conv = nn.Sequential(
            BasicConv2d(2 * channel, channel, 1),
        )

        self.conv3 = nn.Sequential(
            BasicConv2d(channel, channel, 1),
        )

        self.convreduce = nn.Sequential(
            BasicConv2d(2*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1),
            abn(channel)
        )

        self.GIA = GIA(channel,channel)

    def forward(self, fl, fh, fs):
        fsl = F.interpolate(fs, size=fl.size()[2:], mode='bilinear')
        fh = self.conv1(fl * fh + fh)
        fl = self.conv2(fh * fl + fl)
        out_pre = torch.cat((fh, fl), 1)
        out_pre = self.S_conv(out_pre)
        out_pre = out_pre * fsl + out_pre
        local = self.conv3(out_pre)
        local = self.GIA(local)

        return local


class MGBF(nn.Module):
    def __init__(self, abn=BatchNorm, in_fea=[128,128,128], mid_fea=64, out_fea=1):
        super(MGBF, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1),
            DeformConv2d(mid_fea, mid_fea, kernel_size=3),
            abn(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1),
            DeformConv2d(mid_fea, mid_fea, kernel_size=3),
            abn(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1),
            DeformConv2d(mid_fea, mid_fea, kernel_size=3),
            abn(mid_fea)
        )
        self.conv33 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            abn(128)
        )
        self.conv11 = nn.Conv2d(mid_fea, mid_fea, kernel_size=1)
        self.reduce = nn.Sequential(
            nn.Conv2d(mid_fea*2, mid_fea, kernel_size=1),
            DeformConv2d(mid_fea, mid_fea, kernel_size=3),
            abn(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea*2, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h1, w1 = x1.size()
        _, _, h2, w2 = x2.size()

        edge1_fea = self.conv1(x1)
        edge2_fea = self.conv2(x2)
        edge3_fea = self.conv3(x3)
        edge32 = F.interpolate(edge3_fea, size=(h2, w2), mode='bilinear', align_corners=True)
        edge32 = torch.cat([edge32, edge2_fea], dim=1)
        edge32 = self.reduce(edge32)
        edge32 = F.interpolate(edge32, size=(h1, w1), mode='bilinear', align_corners=True)

        edge_fea = torch.cat([edge32, edge1_fea], dim=1)
        edge_fea = self.conv33(edge_fea)

        return edge_fea


class RAGRNet(nn.Module):
    def __init__(self, channel=128, abn=BatchNorm):
        super(RAGRNet, self).__init__()
        # Backbone model
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.reduce_sal1 = Reduction(256, channel)
        self.reduce_sal2 = Reduction(512, channel)
        self.reduce_sal3 = Reduction(1024, channel)
        self.reduce_sal4 = Reduction(2048, channel)
        self.reduce_sal5 = Reduction(2048, channel)

        self.reduce_edg1 = Reduction(256, channel)
        self.reduce_edg2 = Reduction(512, channel)
        self.reduce_edg3 = Reduction(1024, channel)
        self.reduce_edg4 = Reduction(2048, channel)
        self.reduce_edg5 = Reduction(2048, channel)

        self.S1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S5 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S6 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S7 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S8 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S9 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S10 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S_conv1 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.S_conv2 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.S_conv3 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.S_conv4 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )

        self.sigmoid = nn.Sigmoid()

        self.CPM4 = CPM(channel)
        self.CPM3 = CPM(channel)
        self.CPM2 = CPM(channel)
        self.CPM1 = CPM(channel)

        self.MRA1 = MRA(channel)
        self.MRA2 = MRA(channel)

        self.MGBF = MGBF(abn)
        self.RAGR = RAGR(128, 32, 4, abn)

        self.resize = nn.Sequential(
            DeformConv2d(channel, channel, kernel_size=3, stride=2, padding=1),
            abn(channel)
        )

        self.upsample_2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.convenc = ConvEncoder(3,128)
        self.conv11 = nn.Conv2d(channel, channel, 1)

    def forward(self, x):
        size = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)
        x = self.resnet.maxpool(x0)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x_sal1 = self.reduce_sal1(x1)
        x_sal2 = self.reduce_sal2(x2)
        x_sal3 = self.reduce_sal3(x3)
        x_sal4 = self.reduce_sal4(x4)

        x_edg1 = self.reduce_edg1(x1)
        x_edg2 = self.reduce_edg2(x2)
        x_edg3 = self.reduce_edg3(x3)
        x_edg4 = self.reduce_edg4(x4)

        edge_fea = self.MGBF(x_edg2,x_edg3,x_edg4)
        x_graph = self.RAGR(x_sal4, edge_fea.detach())
        x_graph1 = self.RAGR(x_graph, edge_fea.detach())

##  HSCI Start ##
        x_graph4 = self.resize(self.resize(x_graph1))
        x_sal4 = x_graph4 + x_sal4
        x_sal4 = self.convenc(x_sal4)
        edge_fea4 = self.resize(self.resize(edge_fea))
        x_edg4 = self.sigmoid(edge_fea4) * x_edg4 + x_edg4
        x_edg4 = self.convenc(x_edg4)
        sal4, edg4 = self.CPM4(x_sal4, x_edg4)
##  HSCI End ##

##  HSCI Start ##
        sal4_3 = F.interpolate(sal4, size=x_sal3.size()[2:], mode='bilinear')
        edg4_3 = F.interpolate(edg4, size=x_sal3.size()[2:], mode='bilinear')
        x_sal3 = self.S_conv1(torch.cat((sal4_3, x_sal3), 1))
        x_edg3 = self.S_conv2(torch.cat((edg4_3, x_edg3), 1))
        x_graph3 = self.resize(x_graph1)
        x_sal3 = x_graph3 + x_sal3
        edge_fea3 = self.resize(edge_fea)
        x_edg3 = self.sigmoid(edge_fea3) * x_edg3 + x_edg3
        sal3, edg3 = self.CPM3(x_sal3, x_edg3)
##  HSCI End ##

##  HSCI Start ##
        sal3_2 = F.interpolate(sal3, size=x_sal2.size()[2:], mode='bilinear')
        edg3_2 = F.interpolate(edg3, size=x_sal2.size()[2:], mode='bilinear')
        x_sal2 = self.MRA1(x_sal2, sal3_2, sal4)
        x_edg2 = self.S_conv3(torch.cat((edg3_2, x_edg2), 1))
        x_sal2 = x_graph1 + x_sal2
        x_edg2 = self.sigmoid(edge_fea) * x_edg2 + x_edg2
        sal2, edg2 = self.CPM2(x_sal2, x_edg2)
##  HSCI End ##

##  HSCI Start ##
        sal2_1 = F.interpolate(sal2, size=x_sal1.size()[2:], mode='bilinear')
        edg2_1 = F.interpolate(edg2, size=x_sal1.size()[2:], mode='bilinear')
        x_sal1 = self.MRA2(x_sal1, sal2_1, sal4)
        x_edg1 = self.S_conv4(torch.cat((edg2_1, x_edg1), 1))
        x_graph11 = self.upsample_2(x_graph1)
        x_sal1 = x_sal1 + x_graph11
        edge_fea1 = self.upsample_2(edge_fea)
        x_edg1 = self.sigmoid(edge_fea1) * x_edg1
        sal1, edg1 = self.CPM1(x_sal1, x_edg1)
##  HSCI End ##

##  Sal Head Start##
        sal_out = self.S1(sal1)
        edg_out = self.S2(edg1)
        sal2 = self.S3(sal2)
        edg2 = self.S4(edg2)
        sal3 = self.S5(sal3)
        edg3 = self.S6(edg3)
        sal4 = self.S7(sal4)
        edg4 = self.S8(edg4)
        graph_sal = self.S9(x_graph)
        graph_sal1 = self.S9(x_graph1)
        edge_sal = self.S10(edge_fea)

        sal_out = F.interpolate(sal_out, size=size, mode='bilinear', align_corners=True)
        edg_out = F.interpolate(edg_out, size=size, mode='bilinear', align_corners=True)
        sal2 = F.interpolate(sal2, size=size, mode='bilinear', align_corners=True)
        edg2 = F.interpolate(edg2, size=size, mode='bilinear', align_corners=True)
        sal3 = F.interpolate(sal3, size=size, mode='bilinear', align_corners=True)
        edg3 = F.interpolate(edg3, size=size, mode='bilinear', align_corners=True)
        sal4 = F.interpolate(sal4, size=size, mode='bilinear', align_corners=True)
        edg4 = F.interpolate(edg4, size=size, mode='bilinear', align_corners=True)
        edge_sal = F.interpolate(edge_sal, size=size, mode='bilinear', align_corners=True)
        graph_sal = F.interpolate(graph_sal, size=size, mode='bilinear', align_corners=True)
        g1_sal = F.interpolate(graph_sal1, size=size, mode='bilinear', align_corners=True)
##  Sal Head End ##

        return sal_out, self.sigmoid(sal_out), edg_out, sal2, self.sigmoid(sal2), edg2,  sal3, self.sigmoid(sal3), edg3,\
            sal4, self.sigmoid(sal4), edg4, edge_sal, graph_sal, self.sigmoid(graph_sal), g1_sal, self.sigmoid(g1_sal)


if __name__ == '__main__':
    a = torch.randn(8,3,256,256)
    device = torch.device('cpu')
    model = RAGRNet().to(device)
    o = model(a)
    print(o[0].shape)
    print(o[1].shape)
    print(o[2].shape)
    print(o[3].shape)

    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f'Total params: {Total_params / 1e6}M')
    print(f'Trainable params: {Trainable_params / 1e6}M')
    print(f'Non-trainable params: {NonTrainable_params / 1e6}M')

    flops, params = profile(model, (a,))
    print('flops: ', flops, 'params: ', params)