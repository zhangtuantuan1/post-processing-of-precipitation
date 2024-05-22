import os
import torch
from copy import deepcopy
import numpy as np
import xarray as xr
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random
import netCDF4 as nc
from tqdm import tqdm
# from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import zipfile
import torchvision.models as models
import time
import math

start = time.perf_counter()


def set_seed(seed=500):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


def minmax():
    basin_lable = pd.read_csv(r'./autodl-nas/precipitation-basin - 副本.csv')
    basin_lable1 = basin_lable.iloc[:, 3]
    basin_lable2 = np.empty((45, 61), dtype=list, order='C')
    for i in range(61):
        basin_lable2[:, i] = basin_lable1[(i * 45):(i + 1) * 45]
    data = nc.Dataset(r'./autodl-nas/pre_sum1.nc')
    data = data.variables['pre_sum'][:]
    data_ = nc.Dataset(r'./autodl-nas/merge_total_precipitation_sum-0.nc')
    data_ = data_.variables['total_precipitation_sum'][:, 0, :, :]

    XX = np.empty((data.shape[0] * 508, 1), dtype=list, order='C')
    XX_ = np.empty((data.shape[0] * 508, 1), dtype=list, order='C')
    for k in range(data.shape[0]):
        s = -1
        for i in range(61):
            for j in range(45):
                if basin_lable2[j, i] >= 0:
                    s = s + 1
                    XX[508 * k + s, 0] = data[k, j, i]
                    XX_[508 * k + s, 0] = data_[k, j, i]
    max_g = np.max(XX)
    min_g = np.min(XX)
    max_p = np.max(XX_)
    min_p = np.min(XX_)
    return max_g, min_g, max_p, min_p
max_g, min_g, max_p, min_p = minmax()


def load_data():
    # print(12)
    data1 = xr.open_dataset(r'./autodl-nas/0_pressure_sum.nc')
    data2 = xr.open_dataset(r'./autodl-nas/0_sh500_sum.nc')
    data3 = xr.open_dataset(r'./autodl-nas/0_sh850_sum.nc')
    data4 = xr.open_dataset(r'./autodl-nas/0_total_precipitation_sum.nc')
    data5 = xr.open_dataset(r'./autodl-nas/0_u_surface_sum.nc')
    data6 = xr.open_dataset(r'./autodl-nas/0_u500_sum.nc')
    data7 = xr.open_dataset(r'./autodl-nas/0_u850_sum.nc')
    data8 = xr.open_dataset(r'./autodl-nas/0_v_surface_sum.nc')
    data9 = xr.open_dataset(r'./autodl-nas/0_v850_sum.nc')
    data10 = xr.open_dataset(r'./autodl-nas/0_v500_sum.nc')
    data11 = xr.open_dataset(r'./autodl-nas/0_type_sum_array.nc')
    data12 = xr.open_dataset(r'./autodl-nas/0_sh1000_sum.nc')
    data13 = xr.open_dataset(r'./autodl-nas/0_v1000_sum.nc')
    data14 = xr.open_dataset(r'./autodl-nas/0_u1000_sum.nc')
    data15 = xr.open_dataset(r'./autodl-nas/dem_sum.nc')

    data16 = np.load(r'./autodl-tmp/0_Z500_sum.npy', allow_pickle=True)
    data17 = np.load(r'./autodl-tmp/0_Z850_sum.npy', allow_pickle=True)
    data18 = np.load(r'./autodl-tmp/0_Z1000_sum.npy', allow_pickle=True)
    data19 = np.load(r'./autodl-tmp/0_tmp500_sum.npy', allow_pickle=True)
    data20 = np.load(r'./autodl-tmp/0_tmp850_sum.npy', allow_pickle=True)
    data21 = np.load(r'./autodl-tmp/0_tmp1000_sum.npy', allow_pickle=True)
    label = xr.open_dataset(r'./autodl-nas/pre_sum.nc')
    data1_pressure = data1['pressure_sum'][:].values  # (693420, 3, 5, 5)
    data2_sh500 = data2['sh500_sum'][:].values
    data3_sh850 = data3['sh850_sum'][:].values
    data4_total_precipitation = data4['total_precipitation_sum'][:].values
    data5_u_surface = data5['u_surface_sum'][:].values
    data6_u500 = data6['u500_sum'][:].values
    data7_u850 = data7['u850_sum'][:].values
    data8_v_surface = data8['v_surface_sum'][:].values
    data9_v850 = data9['v850_sum'][:].values
    data10_v500 = data10['v500_sum'][:].values
    data11_type = data11['type'][:].values
    data12_sh1000 = data12['sh1000_sum'][:].values
    data13_v1000 = data13['v1000_sum'][:].values
    data14_u1000 = data14['u1000_sum'][:].values
    data15_dem = data15['dem'][:].values
    label_pre = label['pre_sum'][:].values
    N = int(len(label_pre) * 0.25)
    dict_train1 = {
        'pressure': data1_pressure[N:],
        'sh500': data2_sh500[N:],
        'sh850': data3_sh850[N:],
        'total_precipitation': data4_total_precipitation[N:],
        'u_surface': data5_u_surface[N:],
        'u500': data6_u500[N:],
        'u850': data7_u850[N:],
        'v_surface': data8_v_surface[N:],
        'v850': data9_v850[N:],
        'v500': data10_v500[N:],
        'type': data11_type[N:],
        'sh1000': data12_sh1000[N:],
        'v1000': data13_v1000[N:],
        'u1000': data14_u1000[N:],
        'dem': data15_dem[N:],
        'Z500': data16[N:],
        'Z850': data17[N:],
        'Z1000': data18[N:],
        'tmp500': data19[N:],
        'tmp850': data20[N:],
        'tmp1000': data21[N:],
        
        
        

        'label': label_pre[N:]}
    dict_valid1 = {
        'pressure': data1_pressure[:N],
        'sh500': data2_sh500[:N],
        'sh850': data3_sh850[:N],
        'total_precipitation': data4_total_precipitation[:N],
        'u_surface': data5_u_surface[:N],
        'u500': data6_u500[:N],
        'u850': data7_u850[:N],
        'v_surface': data8_v_surface[:N],
        'v850': data9_v850[:N],
        'v500': data10_v500[:N],
        'type': data11_type[:N],
        'sh1000': data12_sh1000[:N],
        'v1000': data13_v1000[:N],
        'u1000': data14_u1000[:N],
        'dem': data15_dem[:N],
        'Z500': data16[:N],
        'Z850': data17[:N],
        'Z1000': data18[:N],
        'tmp500': data19[:N],
        'tmp850': data20[:N],
        'tmp1000': data21[:N],
        
        
        'label': label_pre[:N]}

    dict_train2 = {
        'pressure': data1_pressure[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'sh500': data2_sh500[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'sh850': data3_sh850[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'total_precipitation': data4_total_precipitation[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'u_surface': data5_u_surface[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'u500': data6_u500[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'u850': data7_u850[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'v_surface': data8_v_surface[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'v850': data9_v850[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'v500': data10_v500[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'type': data11_type[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'sh1000': data12_sh1000[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'v1000': data13_v1000[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'u1000': data14_u1000[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'dem': data15_dem[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'Z500': data16[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'Z850': data17[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'Z1000': data18[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'tmp500': data19[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'tmp850': data20[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        'tmp1000': data21[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))],
        
        


        'label': label_pre[(list(range(0, N, 1))) + (list(range(2 * N, 4 * N, 1)))]}
    dict_valid2 = {
        'pressure': data1_pressure[N:(2 * N)],
        'sh500': data2_sh500[N:(2 * N)],
        'sh850': data3_sh850[N:(2 * N)],
        'total_precipitation': data4_total_precipitation[N:(2 * N)],
        'u_surface': data5_u_surface[N:(2 * N)],
        'u500': data6_u500[N:(2 * N)],
        'u850': data7_u850[N:(2 * N)],
        'v_surface': data8_v_surface[N:(2 * N)],
        'v850': data9_v850[N:(2 * N)],
        'v500': data10_v500[N:(2 * N)],
        'type': data11_type[N:(2 * N)],
        'sh1000': data12_sh1000[N:(2 * N)],
        'v1000': data13_v1000[N:(2 * N)],
        'u1000': data14_u1000[N:(2 * N)],
        'dem': data15_dem[N:(2 * N)],
        'Z500': data16[N:(2 * N)],
        'Z850': data17[N:(2 * N)],
        'Z1000': data18[N:(2 * N)],
        'tmp500': data19[N:(2 * N)],
        'tmp850': data20[N:(2 * N)],
        'tmp1000': data21[N:(2 * N)],


        'label': label_pre[N:(2 * N)]}

    dict_train3 = {
        'pressure': data1_pressure[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'sh500': data2_sh500[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'sh850': data3_sh850[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'total_precipitation': data4_total_precipitation[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'u_surface': data5_u_surface[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'u500': data6_u500[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'u850': data7_u850[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'v_surface': data8_v_surface[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'v850': data9_v850[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'v500': data10_v500[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'type': data11_type[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'sh1000': data12_sh1000[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'v1000': data13_v1000[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'u1000': data14_u1000[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'dem': data15_dem[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'Z500': data16[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'Z850': data17[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'Z1000': data18[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'tmp500': data19[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'tmp850': data20[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],
        'tmp1000': data21[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))],        
        
        
        
        'label': label_pre[(list(range(0, 2 * N, 1))) + (list(range(3 * N, 4 * N, 1)))]}
    dict_valid3 = {
        'pressure': data1_pressure[(2 * N):(3 * N)],
        'sh500': data2_sh500[(2 * N):(3 * N)],
        'sh850': data3_sh850[(2 * N):(3 * N)],
        'total_precipitation': data4_total_precipitation[(2 * N):(3 * N)],
        'u_surface': data5_u_surface[(2 * N):(3 * N)],
        'u500': data6_u500[(2 * N):(3 * N)],
        'u850': data7_u850[(2 * N):(3 * N)],
        'v_surface': data8_v_surface[(2 * N):(3 * N)],
        'v850': data9_v850[(2 * N):(3 * N)],
        'v500': data10_v500[(2 * N):(3 * N)],
        'type': data11_type[(2 * N):(3 * N)],
        'sh1000': data12_sh1000[(2 * N):(3 * N)],
        'v1000': data13_v1000[(2 * N):(3 * N)],
        'u1000': data14_u1000[(2 * N):(3 * N)],
        'dem': data15_dem[(2 * N):(3 * N)],
        'Z500': data16[(2 * N):(3 * N)],
        'Z850': data17[(2 * N):(3 * N)],
        'Z1000': data18[(2 * N):(3 * N)],
        'tmp500': data19[(2 * N):(3 * N)],
        'tmp850': data20[(2 * N):(3 * N)],
        'tmp1000': data21[(2 * N):(3 * N)],
        
        
        'label': label_pre[(2 * N):(3 * N)]}

    dict_train4 = {
        'pressure': data1_pressure[:(3 * N)],
        'sh500': data2_sh500[:(3 * N)],
        'sh850': data3_sh850[:(3 * N)],
        'total_precipitation': data4_total_precipitation[:(3 * N)],
        'u_surface': data5_u_surface[:(3 * N)],
        'u500': data6_u500[:(3 * N)],
        'u850': data7_u850[:(3 * N)],
        'v_surface': data8_v_surface[:(3 * N)],
        'v850': data9_v850[:(3 * N)],
        'v500': data10_v500[:(3 * N)],
        'type': data11_type[:(3 * N)],
        'sh1000': data12_sh1000[:(3 * N)],
        'v1000': data13_v1000[:(3 * N)],
        'u1000': data14_u1000[:(3 * N)],
        'dem': data15_dem[:(3 * N)],
        'Z500': data16[:(3 * N)],
        'Z850': data17[:(3 * N)],
        'Z1000': data18[:(3 * N)],
        'tmp500': data19[:(3 * N)],
        'tmp850': data20[:(3 * N)],
        'tmp1000': data21[:(3 * N)],        
        


        'label': label_pre[:(3 * N)]}
    dict_valid4 = {
        'pressure': data1_pressure[(3 * N):],
        'sh500': data2_sh500[(3 * N):],
        'sh850': data3_sh850[(3 * N):],
        'total_precipitation': data4_total_precipitation[(3 * N):],
        'u_surface': data5_u_surface[(3 * N):],
        'u500': data6_u500[(3 * N):],
        'u850': data7_u850[(3 * N):],
        'v_surface': data8_v_surface[(3 * N):],
        'v850': data9_v850[(3 * N):],
        'v500': data10_v500[(3 * N):],
        'type': data11_type[(3 * N):],
        'sh1000': data12_sh1000[(3 * N):],
        'v1000': data13_v1000[(3 * N):],
        'u1000': data14_u1000[(3 * N):],
        'dem': data15_dem[(3 * N):],
        'Z500': data16[(3 * N):],
        'Z850': data17[(3 * N):],
        'Z1000': data18[(3 * N):],
        'tmp500': data19[(3 * N):],
        'tmp850': data20[(3 * N):],
        'tmp1000': data21[(3 * N):],        
        
        
        'label': label_pre[(3 * N):]}

    train_dataset1 = EarthDataSet(dict_train1)
    valid_dataset1 = EarthDataSet(dict_valid1)
    train_dataset2 = EarthDataSet(dict_train2)
    valid_dataset2 = EarthDataSet(dict_valid2)
    train_dataset3 = EarthDataSet(dict_train3)
    valid_dataset3 = EarthDataSet(dict_valid3)
    train_dataset4 = EarthDataSet(dict_train4)
    valid_dataset4 = EarthDataSet(dict_valid4)

    return train_dataset1, valid_dataset1, train_dataset2, valid_dataset2, train_dataset3, valid_dataset3, train_dataset4, valid_dataset4

class EarthDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['pressure'])

    def __getitem__(self, idx):
        return (self.data['pressure'][idx], self.data['sh500'][idx],
                self.data['sh850'][idx], self.data['total_precipitation'][idx],
                self.data['u_surface'][idx], self.data['u500'][idx],
                self.data['u850'][idx], self.data['v_surface'][idx], self.data['v850'][idx],
                self.data['v500'][idx], self.data['type'][idx], self.data['sh1000'][idx], self.data['v1000'][idx],
                self.data['u1000'][idx], self.data['dem'][idx],self.data['Z500'][idx],self.data['Z850'][idx],self.data['Z1000'][idx],self.data['tmp500'][idx],self.data['tmp850'][idx],self.data['tmp1000'][idx]), self.data['label'][idx]


class CNN_LSTM(nn.Module):
    def __init__(self, n_cnn_layer: int = 1, kernals: list = [3], n_lstm_units: int = 8, n_lstm_units1: int = 32):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=0)  # 32
        self.bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(6, 6, kernel_size=9, stride=1, padding=0)
        self.conv4 = nn.Conv2d(6, 6, kernel_size=9, stride=1, padding=0)
        self.lstm = nn.LSTM(16 * 1, n_lstm_units, 2, bidirectional=True, batch_first=True)  # dropout = 0.25
        self.linear = nn.Linear(16, 1)

    def forward(self, pressure, sh500, sh850, total_precipitation, u_surface, u500, u850, v_surface, v850, v500, type,
                sh1000, v1000, u1000, dem,Z500,Z850,Z1000,tmp500,tmp850,tmp1000):
        Seqs = []
        for i in range(3):
            pressure_t = pressure[:, i, :, :].unsqueeze(1)  # [batch,1,5,5]
            sh500_t = sh500[:, i, :, :].unsqueeze(1)
            sh850_t = sh850[:, i, :, :].unsqueeze(1)
            total_precipitation_t = total_precipitation[:, i, :, :].unsqueeze(1)
            u_surface_t = u_surface[:, i, :, :].unsqueeze(1)
            u500_t = u500[:, i, :, :].unsqueeze(1)
            u850_t = u850[:, i, :, :].unsqueeze(1)
            v_surface_t = v_surface[:, i, :, :].unsqueeze(1)
            v850_t = v850[:, i, :, :].unsqueeze(1)
            v500_t = v500[:, i, :, :].unsqueeze(1)
            sh1000_t = sh1000[:, i, :, :].unsqueeze(1)
            v1000_t = v1000[:, i, :, :].unsqueeze(1)
            u1000_t = u1000[:, i, :, :].unsqueeze(1)
            type_t = type[:, i, :, :].unsqueeze(1)
            dem_t = dem[:, i, :, :].unsqueeze(1)
            seq1 = torch.cat(
                [total_precipitation_t, u_surface_t, v_surface_t, u500_t, u850_t, v850_t, v500_t, pressure_t, sh500_t,
                 sh850_t, sh1000_t, v1000_t, u1000_t, dem_t], dim=1)  # [batch,10,5,5]

            Z500_t = Z500[:, i, :, :].unsqueeze(1)
            Z850_t = Z850[:, i, :, :].unsqueeze(1)
            Z1000_t = Z1000[:, i, :, :].unsqueeze(1)
            tmp500_t = tmp500[:, i, :, :].unsqueeze(1)
            tmp850_t = tmp850[:, i, :, :].unsqueeze(1)
            tmp1000_t = tmp1000[:, i, :, :].unsqueeze(1)
            
            seq2 = torch.cat([Z500_t, Z850_t, Z1000_t,tmp500_t, tmp850_t, tmp1000_t], dim=1)  # [batch,10,5,5]
            seq2 = self.conv3(seq2)  # [batch,32,3,3]
            seq2 = self.conv4(seq2)
            seq3 = torch.cat([seq1, seq2], dim=1)
            seq3 = self.conv1(seq3)  # [batch,32,3,3]
            seq3 = self.conv2(seq3)  # [batch,10,1,1]
            seq3 = torch.flatten(seq3, start_dim=1).unsqueeze(1)  # [batch,10*1*1]
            Seqs.append(seq3)  # [batch,1, 10*1*1]  *  3

        x = torch.cat([Seqs[i] for i in range(3)], dim=1)  # [batch,3,10]
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = x[:, 2, :]
        return x


def rmse(preds, y):
    return np.sqrt(sum((preds - y) ** 2) / preds.shape[0])


def coreff(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    c1 = sum((x - x_mean) * (y - y_mean))
    c2 = sum((x - x_mean) ** 2) * sum((y - y_mean) ** 2)
    return c1 / np.sqrt(c2)


def abss(x, y):
    return sum(abs(x - y)) / x.shape[0]


fit_params = {
    'n_epochs': 200,
    'learning_rate': 0.0005,
    'batch_size': 5000, }

train_dataset1, valid_dataset1, train_dataset2, valid_dataset2, train_dataset3, valid_dataset3, train_dataset4, valid_dataset4 = load_data()
train_loader1 = DataLoader(train_dataset1, batch_size=fit_params['batch_size'], shuffle=True, pin_memory=True,
                           num_workers=8)
valid_loader1 = DataLoader(valid_dataset1, batch_size=fit_params['batch_size'], shuffle=False, pin_memory=True,
                           num_workers=8)
train_loader2 = DataLoader(train_dataset2, batch_size=fit_params['batch_size'], shuffle=True, pin_memory=True,
                           num_workers=8)
valid_loader2 = DataLoader(valid_dataset2, batch_size=fit_params['batch_size'], shuffle=False, pin_memory=True,
                           num_workers=8)
train_loader3 = DataLoader(train_dataset3, batch_size=fit_params['batch_size'], shuffle=True, pin_memory=True,
                           num_workers=8)
valid_loader3 = DataLoader(valid_dataset3, batch_size=fit_params['batch_size'], shuffle=False, pin_memory=True,
                           num_workers=8)
train_loader4 = DataLoader(train_dataset4, batch_size=fit_params['batch_size'], shuffle=True, pin_memory=True,
                           num_workers=8)
valid_loader4 = DataLoader(valid_dataset4, batch_size=fit_params['batch_size'], shuffle=False, pin_memory=True,
                           num_workers=8)


def train(train_loader, valid_loader):
    set_seed()

    model = CNN_LSTM()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'])  # weight_decay=0.001
    loss_fn = nn.MSELoss()

    model.to(device)
    loss_fn.to(device)
    best_sco = 1000

    for i in range(fit_params['n_epochs']):
        print('Epoch: {}/{}'.format(i + 1, fit_params['n_epochs']))

        model.train()
        for step, ((pressure, sh500, sh850, total_precipitation, u_surface, u500, u850, v_surface, v850, v500, type,
                    sh1000, v1000, u1000, dem,Z500,Z850,Z1000,tmp500,tmp850,tmp1000),
                   label) in enumerate(train_loader):
            pressure = pressure.to(device).float()
            sh500 = sh500.to(device).float()
            sh850 = sh850.to(device).float()
            total_precipitation = total_precipitation.to(device).float()
            u_surface = u_surface.to(device).float()
            u500 = u500.to(device).float()
            u850 = u850.to(device).float()
            v_surface = v_surface.to(device).float()
            v850 = v850.to(device).float()
            v500 = v500.to(device).float()
            type = type.to(device).float()
            sh1000 = sh1000.to(device).float()
            v1000 = v1000.to(device).float()
            u1000 = u1000.to(device).float()
            dem = dem.to(device).float()
            Z500 = Z500.to(device).float()
            Z850 = Z850.to(device).float()
            Z1000 = Z1000.to(device).float()
            tmp500 = tmp500.to(device).float()
            tmp850 = tmp850.to(device).float()
            tmp1000 = tmp1000.to(device).float()            
            
            

            optimizer.zero_grad()
            label = label.to(device).float()
            preds = model(pressure, sh500, sh850, total_precipitation, u_surface, u500, u850, v_surface, v850, v500,
                          type, sh1000, v1000, u1000, dem, Z500, Z850, Z1000,tmp500,tmp850,tmp1000)            # loss = loss_fn(torch.exp(preds * (max_g - min_g) + min_g),torch.exp(label * (max_g - min_g) + min_g))
            loss = loss_fn((preds * (max_g - min_g) + min_g), (label * (max_g - min_g) + min_g))  ###均方误差
            loss.backward()
            optimizer.step()



    with torch.no_grad():
        model.eval()
        y_true, y_pred, y_yubao = [], [], []
        for step, (
                (
                pressure, sh500, sh850, total_precipitation, u_surface, u500, u850, v_surface, v850, v500, type, sh1000,
                v1000, u1000, dem,Z500,Z850,Z1000,tmp500,tmp850,tmp1000),
                label) in enumerate(valid_loader):
            pressure = pressure.to(device).float()
            sh500 = sh500.to(device).float()
            sh850 = sh850.to(device).float()
            total_precipitation = total_precipitation.to(device).float()
            u_surface = u_surface.to(device).float()
            u500 = u500.to(device).float()
            u850 = u850.to(device).float()
            v_surface = v_surface.to(device).float()
            v850 = v850.to(device).float()
            v500 = v500.to(device).float()
            type = type.to(device).float()
            label = label.to(device).float()
            sh1000 = sh1000.to(device).float()
            v1000 = v1000.to(device).float()
            u1000 = u1000.to(device).float()
            dem = dem.to(device).float()
            Z500 = Z500.to(device).float()
            Z850 = Z850.to(device).float()
            Z1000 = Z1000.to(device).float()
            tmp500 = tmp500.to(device).float()
            tmp850 = tmp850.to(device).float()
            tmp1000 = tmp1000.to(device).float()              
            

            preds = model(pressure, sh500, sh850, total_precipitation, u_surface, u500, u850, v_surface, v850, v500,
                          type, sh1000, v1000, u1000, dem,Z500,Z850,Z1000,tmp500,tmp850,tmp1000)
            preds_pre = total_precipitation[:, 2, 2, 2].unsqueeze(1)
            y_pred.append(preds)
            y_true.append(label)
            y_yubao.append(preds_pre)
        y_true = torch.cat(y_true, axis=0)
        y_true1 = (y_true * (max_g - min_g) + min_g)

        y_pred = torch.cat(y_pred, axis=0)
        y_pred1 = (y_pred * (max_g - min_g) + min_g)

        y_yubao = torch.cat(y_yubao, axis=0)
        y_yubao1 = (y_yubao * (max_p - min_p) + min_p)

    return y_true1, y_pred1, y_yubao1


y_true1, y_pred1, y_yubao1 = train(train_loader1, valid_loader1)
y_true2, y_pred2, y_yubao2 = train(train_loader2, valid_loader2)
y_true3, y_pred3, y_yubao3 = train(train_loader3, valid_loader3)
y_true4, y_pred4, y_yubao4 = train(train_loader4, valid_loader4)

y_true = torch.cat((y_true1, y_true2, y_true3, y_true4), axis=0)
y_pred = torch.cat((y_pred1, y_pred2, y_pred3, y_pred4), axis=0)
y_yubao = torch.cat((y_yubao1, y_yubao2, y_yubao3, y_yubao4), axis=0)

y_true = y_true.cpu().detach().numpy()
y_pred = y_pred.cpu().detach().numpy()
y_yubao = y_yubao.cpu().detach().numpy()

y_true = torch.from_numpy(y_true)

y_pred = torch.from_numpy(y_pred)

y_yubao = torch.from_numpy(y_yubao)

y_huizong = torch.cat([y_true, y_pred, y_yubao], 1)


np.save('cnn-cnn-lstm-sum1.npy', y_huizong)

end = time.perf_counter()
run_time = (end - start) / 60

print("运行时间：", run_time, "分")



