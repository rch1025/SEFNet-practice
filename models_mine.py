from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from utils import *
from torch.autograd import Variable
import sys
import math
from layers_mine import *


class Model(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.m = data.m # 변수 개수
        self.w = args.window # window_size
        self.h = args.horizon # forecasting length
        self.k = args.k
        self.hw = args.hw # AR 모듈(Linear layer)의 input size
        self.hidA = args.hidA # attention Q, K, V의 linear layer의 hidden size
        self.hidR = args.hidR # intra series내의 LSTM의 hidden size
        self.hidP = args.hidP # RegionAwareConv의 hidden size
        self.num_layers = args.n_layer # LSTM의 num_layer
        self.dp = args.dropout
        self.dropout = nn.Dropout(p=self.dp)
        self.activate = nn.LeakyReLU()
        self.highway = nn.Linear(self.hw, 1) # AR 모듈
        self.output = nn.Linear(self.hidA+self.hidR, 1) # parametric matrix fusion을 위한 layer -> inter와 intra의 결과를 concat하여 input으로 받음
        self.regionconvhid = self.k * 4*self.hidP + self.k # RegionAwareConv가 Q, K, V에 입력될 때 사용되는 input size값
        
        self.lstm = nn.LSTM(1, self.hidR, bidirectional=False, batch_first=True, num_layers=self.num_layers)
        self.rac = RegionAwareConv(P=self.w, m=self.m, k=self.k, hidP=self.hidP)
        self.q_linear = nn.Linear(self.regionconvhid, self.hidA, bias = True)
        self.k_linear = nn.Linear(self.regionconvhid, self.hidA, bias = True)
        self.v_linear = nn.Linear(self.regionconvhid, self.hidA, bias = True)
        self.attn_layer = DotAtt()
        self.inter = nn.Parameter(torch.FloatTensor(self.m, self.hidA), requires_grad=True)
        self.intra = nn.Parameter(torch.FloatTensor(self.m, self.hidR), requires_grad=True)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        self.init_weights() # weight 초기화 -> xavier_uniform_ 사용
        
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
    
    def forward(self, x, feat=None):
        
        # Inter-Region embedding
        rac = self.rac(x) # RegionAwareConv
        q = self.q_linear(rac) # rac의 결과가 attention에 입력
        k = self.k_linear(rac)
        v = self.v_linear(rac)
        q = nn.Dropout(p=0.2)(q)
        k = nn.Dropout(p=0.2)(k)
        v = nn.Dropout(p=0.2)(v)
        inter_ = self.attn_layer(q, k, v) # i
        
        # Intra-Region embedding
        r = x.permute(0, 2, 1).contiguous().view(-1, x.size(1), 1) # LSTM에 입력될 수 있게 변형
        r_out, hc = self.lstm(r, None)
        last_hid = self.dropout(r_out[:, -1, :])
        intra_ = last_hid.view(-1, self.m, self.hidR) # t
        
        # parametric-matrix fusion
        inter_ = torch.mul(self.inter, inter_)
        intra_ = torch.mul(self.intra, intra_)
        
        res = torch.cat([intra_, inter_], dim=2) # inter와 intra를 concat
        res = self.output(res)
        res = res.squeeze(2)
        
        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res+z
            
        return res, None