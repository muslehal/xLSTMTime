#good
import sys
import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

import numpy as np
import pandas as pd
import os
#import torchcd 
from torch import nn
from src.learner import Learner
from src.callback.core import *
from src.callback.tracking import *
from src.callback.scheduler import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from datautils import get_dls


import time
import random
import argparse
import datetime
from functools import partial
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast

from xlstm1.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

from xlstm1.blocks.mlstm.block import mLSTMBlockConfig
from xlstm1.blocks.slstm.block import sLSTMBlockConfig
from xlstm import xlstm

from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor
assert torch.__version__ >= '1.8.0', "DDP-based MoE requires Pytorch >= 1.8.0"

from dataclasses import dataclass

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n1',type=int,default=128,help='First Embedded representation')#256
parser.add_argument('--n2',type=int,default=256,help='Second Embedded representation')
parser.add_argument('--ch_ind', type=int, default=1, help='Channel Independence; True 1 False 0')
parser.add_argument('--d_state', type=int, default=128, help='d_state parameter of Mamba')#256
parser.add_argument('--dconv', type=int, default=2, help='d_conv parameter of Mamba')
parser.add_argument('--e_fact', type=int, default=2, help='expand factor parameter of Mamba')
parser.add_argument('--residual', type=int, default=1, help='Residual Connection; True 1 False 0')



parser.add_argument('--model_name2', type=str, default='xLSTMTime', help='model_name2')
# IntegratedModel   model1 model2 dlinear
parser.add_argument('--dset', type=str, default='ettm1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64    , help='batch size')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')

parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--use_time_features', type=int, default=1, help='whether to use time features or not')
# Patch
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
# parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=256, help='Transformer d_model')
#parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')
# Optimization args
parser.add_argument('--n_epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--model_id', type=int, default=1, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
# training
parser.add_argument('--is_train', type=int, default=1, help='training the model')



#parser = argparse.ArgumentParser(description='Swin Transformer training and evaluation script', add_help=False)

#parser.add_argument('Swin Transformer training and evaluation script', add_help=False)
parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)




args = parser.parse_args()
#print('args:', args)
#args.save_model_name = 'patchtst_supervised'+'_cw'+str(args.context_points)+'_tw'+str(args.target_points) +'_IMAGE_SIZE'+str(args.IMAGE_SIZE)+'_NUM_CLASSES'+str(args.NUM_CLASSES) +'_patch'+str(args.patch_len) + '_stride'+str(args.stride)+'_epochs'+str(args.n_epochs) + '_model' + str(args.model_id)
#args.save_path = 'saved_models/' + args.dset + '/patchtst_supervised/' + args.model_type + '/'
#if not os.path.exists(args.save_path): os.makedirs(args.save_path)
args.save_model_name = str(args.model_name2)+'_cw'+str(args.context_points)+'_tw'+str(args.target_points) +'_patch'+str(args.patch_len) + '_stride'+str(args.stride)+'_epochs'+str(args.n_epochs) + '_model' + str(args.model_id)
args.save_path = 'saved_models/' + args.dset #/My model/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)


configs =args


def get_model(c_in,args):
    """
    c_in: number of input variables
    """

    #get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)


    ## get model
    model =  xlstm ( configs,enc_in=c_in,
           )
    return model

def combined_loss(input, target, alpha=0.5):
    """
    A combined loss function that computes a weighted sum of MSELoss and L1Loss.
    `alpha` is the weight for MSELoss and (1-alpha) is the weight for L1Loss.
    """
    mse_loss = torch.nn.MSELoss(reduction='mean')
    l1_loss = torch.nn.L1Loss(reduction='mean')
    return alpha * mse_loss(input, target) + (1 - alpha) * l1_loss(input, target)

def find_lr():
    # get dataloader
    dls = get_dls(args)    
    model = get_model(dls.vars, args)
    

    # get loss
    #loss_func = torch.nn.MSELoss(reduction='mean')
    loss_func = torch.nn.L1Loss(reduction='mean')
    #loss_func=combined_loss
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    #cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    # define learner
    learn = Learner(dls,model,  loss_func , cbs=cbs  )  #cbs=cbs                      
    # fit the data to the model
    return learn.lr_finder()


def train_func(lr=args.lr):
    # get dataloader
    dls = get_dls(args)
    #print('in out', dls.vars, dls.c, dls.len)
    
    # get model
    model = get_model(dls.vars, args)
    #model = get_model(dls.vars, args, model_type)

    # get loss
    #loss_func = torch.nn.MSELoss(reduction='mean')
    loss_func = torch.nn.L1Loss(reduction='mean')
    #loss_func=combined_loss

    #delta = 0.25
    #loss_func = HuberLoss(delta)
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [
    #cbs = [
         #PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_model_name, 
                     path=args.save_path )
        ]

    # define learner
    learn = Learner(dls, model , loss_func,
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse,mae]
                        )
                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)


def test_func():
    weight_path =args.save_path+'/' + args.save_model_name + '.pth'
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    #model = torch.load(weight_path)
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    #cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs)#cbs=cbs
    out  = learn.test(dls.test, weight_path=weight_path, scores=[mse,mae])         # out: a list of [pred, targ, score_values]
    return out


import matplotlib.pyplot as plt
def plot_feature_actual_vs_predicted(actual, predicted, feature_idx):
    """
    Plot the actual vs predicted values for a specific feature for the first sequence.

    Parameters:
    - actual (np.array or torch.Tensor): Array of actual values.
    - predicted (np.array or torch.Tensor): Array of predicted values.
    - feature_idx (int): Index of the feature to plot.
    """
   
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()

    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()

    ## Selecting the feature across all time steps
    #actual_feature = actual[0:, feature_idx]
    #predicted_feature = predicted[0:, feature_idx]
    
    # Select the first sequence for the given feature index
    actual_feature = actual[0, :, feature_idx]
    predicted_feature = predicted[0, :, feature_idx]
    #actual_feature = np.mean(actual[: , : ,feature_idx ], axis=0 )
    #predicted_feature = np.mean(predicted[: , : ,feature_idx ], axis=0)

    
    # Plot the first sequence
    plt.figure(figsize=(10, 6))
    plt.plot(actual_feature, label="Actual", color='blue')
    plt.plot(predicted_feature, label="Predicted", color='red', linestyle='--')
    plt.title(f"Actual vs Predicted for Feature {feature_idx}, Sequence 0")
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
   
    if args.is_train:

        suggested_lr = find_lr()
        print('suggested lr:', suggested_lr)
        train_func(suggested_lr)

    else:   # testing mode

        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)
        
        for feature_idx in range(7):  # Assuming there are 7 features
            plot_feature_actual_vs_predicted(out[1], out[0], feature_idx)

   
    print('----------- Complete! -----------')

