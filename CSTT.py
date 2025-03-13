from __future__ import absolute_import
from numpy.lib.utils import who

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.linalg as lin
from torch.nn.modules.activation import ReLU
import torchvision
import numpy as np
import math
from .Sysu_STA import batch_divide, PositionalEncoding
from .freqtrans import DCTBlock2D_v2
from torch.utils import checkpoint

class ShuffledConv1D(nn.Module):
    def __init__(self,in_dim,out_dim,groups,bias=False) -> None:
        super().__init__()
        self.groups = groups
        self.group_dim = in_dim//groups
        self.in_dim = in_dim
        self.proj_dim = out_dim
        self.gconv1 = nn.Conv1d(in_channels=in_dim,out_channels=self.proj_dim,groups = groups,kernel_size=1,bias=bias)
        self.gconv2 = nn.Conv1d(in_channels=self.proj_dim,out_channels=self.proj_dim,groups=self.group_dim,kernel_size=1,bias=bias)
        torch.nn.init.xavier_normal_(self.gconv1.weight)
        torch.nn.init.xavier_normal_(self.gconv2.weight)
    def forward(self,x):
        #Expect input in size: BxCxN
        b_x = x.size(0)
        len = x.size(-1)
        x = self.gconv1(x)
        #Divide the output in G groups
        #Note on Marhsalling: BxGx3C_gxN
        x = x.view(b_x,self.groups,-1,len)
        x = torch.transpose(x,1,2).reshape(b_x,-1,len)
        #Out: Bx3CxL
        return self.gconv2(x)

class MHA(nn.Module):
    def __init__(self,dim,heads) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim/heads)**0.5
        self.act = torch.nn.Softmax(dim=-1)
        torch.nn.init.xavier_normal_(self.head_p1)
        torch.nn.init.xavier_normal_(self.head_p2)
    def forward(self,x):
        #Input: B, 3HD, N
        b_x = x.size(0)
        l_x = x.size(-1)
        x = torch.transpose(x,-1,-2).view(b_x,l_x,3,self.heads,self.dim)
        x = x.permute(0,2,3,1,4) #Bx3,H,NxD
        q,k,v = torch.chunk(x,chunks=3,dim=1)

        qk = torch.einsum('...ij,...kj->...ik',q,k)/self.scale


        qk = self.act(qk)
        
        o = torch.einsum('...ij,...jk->...ik',qk,v)
        o = o.permute(0,1,3,2,4).reshape(b_x,l_x,-1)

        #BxNx(HxD)
        return o.transpose(-1,-2), qk.squeeze(1)

class CSTB(nn.Module):
    def __init__(self,mhdim,num_groups = 8,num_heads = 8) -> None:
        super().__init__()
        self.mhdim = mhdim
        self.indim = mhdim//num_heads
        self.ln1 = nn.LayerNorm(mhdim)
        self.multihead_conv = ShuffledConv1D(self.mhdim,self.mhdim*3,groups=num_groups)
        self.imha = MHA(self.indim,heads=num_heads)
        self.MLP = ShuffledConv1D(self.mhdim,self.mhdim,groups=num_groups)
        self.ln2 = nn.LayerNorm(mhdim)
    def forward(self,x):
        xn = self.ln1(x.transpose(-1,-2)).transpose(-1,-2)
        xn = self.multihead_conv(xn)
        x = self.imha(xn)[0] + x
        xn = self.ln2(x.transpose(-1,-2)).transpose(-1,-2)
        xn = self.MLP(xn)
        return xn+x

class CGTB(nn.Module):
    def __init__(self,mhdim,num_groups = 8,num_heads = 8) -> None:
        super().__init__()
        self.mhdim = mhdim
        self.indim = mhdim//num_heads
        self.ln1 = nn.LayerNorm(mhdim)
        self.multihead_conv = torch.nn.Conv1d(self.mhdim,self.mhdim*3,1,bias=False)
        torch.nn.init.xavier_normal_(self.multihead_conv.weight)
        self.imha = MHA(self.indim,heads=num_heads)
        self.MLP = torch.nn.Conv1d(self.mhdim,self.mhdim,1,bias=False)
        torch.nn.init.xavier_normal_(self.MLP.weight)
        self.ln2 = nn.LayerNorm(mhdim)
    def forward(self,x):
        xn = self.ln1(x.transpose(-1,-2)).transpose(-1,-2)
        xn = self.multihead_conv(xn)
        x = self.imha(xn)[0] + x
        xn = self.ln2(x.transpose(-1,-2)).transpose(-1,-2)
        xn = self.MLP(xn)
        return xn+x

class ResNet_CSTT(nn.Module):
    def __init__(self, num_classes, loss={'xent'},weight_lock=0, **kwargs):
        super().__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)

        block_ct = 0
        for child in resnet50.children():
            block_ct +=1
            if block_ct < weight_lock+1:
                for params in child.parameters():
                    params.requires_grad = False

        stem = list(resnet50.children())[:-2]
        lastBlockList = list(stem[-1].children())
        
        lastBlockList[0]._modules['conv2'].stride = (1,1)
        lastBlockList[0]._modules['downsample'][0].stride = (1,1)

        stem[-1] = nn.Sequential(*lastBlockList)

        self.base = nn.Sequential(*stem)
        #self.base_2 = nn.Sequential(*stem)
        self.fin_act1 = nn.ELU()
        self.fin_act = nn.ELU()

        self.feat_dim = 2048
        
        self.num_heads = 8
        
        ph = 4
        pw = 1
        
        self.avgpool = nn.AdaptiveAvgPool2d((ph,pw))
        self.mhdim = self.feat_dim*ph*pw
        
        self.attent1 = nn.Sequential(
            CSTB(self.mhdim,num_groups=16,num_heads=16),
        )

        
        self.pre_bn = nn.LayerNorm(self.mhdim)
        
        self.fin_bn = nn.BatchNorm1d(self.mhdim)
        self.classifier = nn.Linear(self.mhdim, num_classes,bias=False)
        #nn.init.kaiming_normal_(self.classifier.weight.data,a=0.01)
    def forward(self, x,ax=None, seq_num=None):
        '''Input Shape:
            B x N x D x H x W
            B: Batch Size
            N: Sequence Length
            D: Channel Size
            H: Image Height
            W: Image Width
        '''
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        #x = checkpoint.checkpoint_sequential(self.base,self.stem_len,x)
        #x = self.base(x)
        for m in self.base:
            x = checkpoint.checkpoint(m,x,use_reentrant=False)
        
        x = x.view(b,t,-1,x.size(2),x.size(3))
        c = x

        test_norm = lin.norm(c,2,2)
        norm_sum = torch.sum(test_norm,(2,3),keepdim=True)

        test_norm = test_norm/norm_sum
        
        
        
        c = c.view(b*t,c.size(2),c.size(3),c.size(4))
        c = self.avgpool(c)
        
        co = c.view(b,t,-1,c.size(-2),c.size(-1))
        co = torch.mean(co,(3,4))
        
        c = c.view(b,t,-1).transpose(-1,-2)
        
        for i in range(self.attent1.__len__):
            c = self.attent1[i](c)

        f = c.mean(-1)

        w = self.fin_bn(f)


        y = self.classifier(w)

        if not self.training:
            return batch_divide(w,seq_num)
        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent','htri','ifr'}:
            return y, w, test_norm
        elif self.loss == {'xent', 'htri'}:
            return y, w
        elif self.loss == {'cent'}:
            return y, w
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class SYSU_NTAB(nn.Module):
    #Single-Branch Net With a dummy branch for secondary flipped stream
    def __init__(self, num_classes, loss={'xent'},weight_lock=0, **kwargs):
        super(SYSU_NTAB, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)

        block_ct = 0
        for child in resnet50.children():
            block_ct +=1
            if block_ct < weight_lock+1:
                for params in child.parameters():
                    params.requires_grad = False

        stem = list(resnet50.children())[:-2]
        lastBlockList = list(stem[-1].children())
        
        lastBlockList[0]._modules['conv2'].stride = (1,1)
        lastBlockList[0]._modules['downsample'][0].stride = (1,1)

        stem[-1] = nn.Sequential(*lastBlockList)

        self.base = nn.Sequential(*stem)
        #self.base_2 = nn.Sequential(*stem)
        self.fin_act = nn.ELU()

        self.feat_dim = 2048
        ph = 4
        pw = 1
        
        self.avgpool = nn.AdaptiveAvgPool2d((ph,pw))
        self.mhdim = self.feat_dim*ph*pw
        
        self.fin_bn = nn.BatchNorm1d(self.mhdim)
        self.classifier = nn.Linear(self.mhdim, num_classes,bias=True)
        #nn.init.kaiming_normal_(self.classifier.weight.data,a=0.01)
    def forward(self, x,ax=None, seq_num=None):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        for m in self.base:
            x = checkpoint.checkpoint(m,x,use_reentrant=False)
        x = x.view(b,t,-1,x.size(2),x.size(3))

        c = x

        test_norm = lin.norm(c,2,2)
        norm_sum = torch.sum(test_norm,(2,3),keepdim=True)

        test_norm = test_norm/norm_sum

        c = c.view(b*t,c.size(2),c.size(3),c.size(4))
        c = self.avgpool(c)
        
        c = c.view(b,t,-1).transpose(-1,-2)
        

        f = c.mean(-1)

        w = self.fin_bn(f)

        y = self.classifier(w)

        if not self.training:
            return batch_divide(w,seq_num)
        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent','htri','ifr'}:
            return y, w, test_norm
        elif self.loss == {'xent', 'htri'}:
            return y, w
        elif self.loss == {'cent'}:
            return y, w
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class SYSU_GRU(nn.Module):
    #Single-Branch Net With a dummy branch for secondary flipped stream
    def __init__(self, num_classes, loss={'xent'},weight_lock=0, **kwargs):
        super(SYSU_GRU, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)

        block_ct = 0
        for child in resnet50.children():
            block_ct +=1
            if block_ct < weight_lock+1:
                for params in child.parameters():
                    params.requires_grad = False

        stem = list(resnet50.children())[:-2]
        lastBlockList = list(stem[-1].children())
        
        lastBlockList[0]._modules['conv2'].stride = (1,1)
        lastBlockList[0]._modules['downsample'][0].stride = (1,1)

        stem[-1] = nn.Sequential(*lastBlockList)
        self.base = nn.Sequential(*stem)
        #self.base_2 = nn.Sequential(*stem)
        self.fin_act = nn.ELU()

        self.feat_dim = 2048
        ph = 4
        pw = 1
        
        self.avgpool = nn.AdaptiveAvgPool2d((ph,pw))
        self.mhdim = self.feat_dim*ph*pw
        self.gru = nn.GRU(self.mhdim,self.mhdim,1,batch_first=True)
        self.fin_bn = nn.BatchNorm1d(self.mhdim)
        self.classifier = nn.Linear(self.mhdim, num_classes,bias=True)
        #nn.init.kaiming_normal_(self.classifier.weight.data,a=0.01)
    def forward(self, x,ax, seq_num=None):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        for m in self.base:
            x = checkpoint.checkpoint(m,x,use_reentrant=False)
        x = x.view(b,t,-1,x.size(2),x.size(3))

        c = x

        test_norm = lin.norm(c,2,2)
        norm_sum = torch.sum(test_norm,(2,3),keepdim=True)

        test_norm = test_norm/norm_sum

        c = c.view(b*t,c.size(2),c.size(3),c.size(4))
        c = self.avgpool(c)
        
        c = c.view(b,t,-1)
        
        #print(c.size())
        c, _ = self.gru(c)
        f = c.transpose(-1,-2).mean(-1)
        
        w = self.fin_bn(f)

        y = self.classifier(w)

        if not self.training:
            return batch_divide(w,seq_num)
        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent','htri','ifr'}:
            return y, w, test_norm
        elif self.loss == {'xent', 'htri'}:
            return y, w
        elif self.loss == {'cent'}:
            return y, w
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
