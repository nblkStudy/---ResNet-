# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
#
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision import datasets,transforms,models

from sklearn.model_selection import train_test_split

import help
from MyDataSet import MyDataset
from medicalnet_model import generate_model

# X_train, X_val, y_train, y_val = train_test_split(help.features,help.labels['label'].values,test_size = 0.2,random_state = 42,stratify=help.labels['label'].values)
#
# train_datasets = MyDataset(datas=X_train,labels=y_train,shape=3,input_D=help.input_D,input_H=help.input_H,input_W=help.input_W,phase='train')
# val_datasets = MyDataset(datas=X_val,labels=y_val,shape=3,input_D=help.input_D,input_H=help.input_H,input_W=help.input_W,phase='train')
#
# train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=16, shuffle=True)
# val_loader = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=8, shuffle=False)
#
# checkpoint = torch.load(help.checkpoint_pretrain_resnet_10_23dataset,map_location=help.device)
#
# medicanet_resnet3d_10,parameters = generate_model(sample_input_W=help.input_W,
#                                                    sample_input_H=help.input_H,
#                                                    sample_input_D=help.input_D,
#                                                    num_seg_classes=help.num_seg_classes,
#                                                    phase='train',
#                                                    pretrain_path=help.checkpoint_pretrain_resnet_10_23dataset)
# params = [
#         { 'params': parameters['base_parameters'], 'lr': 0.001 },
#         { 'params': parameters['new_parameters'], 'lr': 0.001*100 }
#         ]
# optimizer = optim.Adam(params, weight_decay=1e-3)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
#
# epochs = 100
# help.train_data(medicanet_resnet3d_10,train_loader,val_loader,epochs,optimizer,scheduler,help.criterion,help.medicanet_3d_resnet10_checkpoint_path,help.device)
# torch.save(medicanet_resnet3d_10, 'medicanet_resnet3d_10.pth')


##########################################################################
# 模型测试
My_model = torch.load('medicanet_resnet3d_10.pth')
test_datasets = MyDataset(datas=help.temp_data,shape=3,input_D=help.input_D,input_H=help.input_H,input_W=help.input_W,phase='test')
test_loader = DataLoader(dataset=test_datasets)
# loadmodel = help.load_checkpoint(help.medicanet_3d_resnet10_checkpoint_path,'medicanet_resnet3d_10','test',help.device)
help.all_predict(test_loader,My_model,help.device,help.result_medicanet_3d_resnet10)