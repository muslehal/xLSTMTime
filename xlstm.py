
import torch.nn.functional as F
from torch import nn
import torch
import argparse
import numpy as np
from einops import rearrange

import argparse

from xlstm1.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

from xlstm1.blocks.mlstm.block import mLSTMBlockConfig
from xlstm1.blocks.slstm.block import sLSTMBlockConfig


mlstm_config = mLSTMBlockConfig()
slstm_config = sLSTMBlockConfig()



config = xLSTMBlockStackConfig(
        mlstm_block=mlstm_config,
        slstm_block=slstm_config,
        num_blocks=3,
        embedding_dim=256,
        add_post_blocks_norm=True,
        
        _block_map = 1,

        #slstm_at="all",
        context_length=336
    )

    





class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    


class series_decomp2(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp2, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean



class xlstm(torch.nn.Module):
    def __init__(self, configs, enc_in):
        super(xlstm, self).__init__()
        self.configs = configs
        self.enc_in = enc_in
        self.batch_norm = nn.BatchNorm1d(self.enc_in)


        
        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp2(kernel_size)
        self.Linear_Seasonal = nn.Linear(configs.context_points,configs.target_points)
        self.Linear_Trend = nn.Linear(configs.context_points,configs.target_points)
        self.Linear_Decoder = nn.Linear(configs.context_points,configs.target_points)
        self.Linear_Seasonal.weight = nn.Parameter((1/configs.context_points)*torch.ones([configs.target_points,configs.context_points]))
        self.Linear_Trend.weight = nn.Parameter((1/configs.context_points)*torch.ones([configs.target_points,configs.context_points]))
    
        self.mm= nn.Linear(self.configs.target_points, self.configs.n2)


        self.mm2= nn.Linear(config.embedding_dim, configs.target_points)
        self.mm3= nn.Linear(configs.context_points,self.configs.n2)

        self.xlstm_stack = xLSTMBlockStack(config )

        


    def forward(self, x):
        #print(x.shape)

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        #print(x.shape)


        x=self.mm(x)
        #print(x.shape)

        
        #x = self.batch_norm(x)
    
        x = self.xlstm_stack(x)
        

        x=self.mm2(x)

        x=x.permute(0,2,1)

    
        return x



