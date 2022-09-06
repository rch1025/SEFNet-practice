import sys
import torch
import numpy as np
from torch.autograd import Variable

class DataBasicLoader(object):
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window # window size (20)
        self.h = args.horizon # forecast length (1)
        self.d = 0
        self.add_his_day = False
        self.rawdat = np.loadtxt(open("data/{}.txt".format(args.dataset)), delimiter=',') # 데이터 불러오기
        print('################ self.rawdat.shape ################')
        print('rawdat.shape', self.rawdat.shape)
        print()

        if (len(self.rawdat.shape)==1):
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1))
        
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape # n_sample : 관측치 개수, m_group : 변수 개수
        print('################ self.dat.shape ################')
        print('관측치 수 :', self.n, '변수 개수 :', self.m)
        print()
        
        self.scale = np.ones(self.m)
        # args.train은 train data의 비율이 들어있음
        self._pre_train(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        # self.train (train 데이터 비율) * self.n (관측치 수)
        self._split(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        print('################ [train, valid, test] shape ################')
        print('size of train/val/test sets', len(self.train[0]), len(self.val[0]), len(self.test[0]))
        print()
        
    def _pre_train(self, train, valid, test):
        self.train_set = train_set = range(self.P+self.h-1, train) # (window_size)+(horizon-1)
        self.valid_set = valid_set = range(train, valid)
        self.test_set = test_set = range(valid, self.n)
        self.tmp_train = self._batchify(train_set, self.h, useraw = True)
        # trainset에 대한 첫 번째 window와 이후 window들에 대한 정답을 concat함
            # (20, 47)과 (154, 47) concat -> (174, 47)
        train_mx = torch.cat((self.tmp_train[0][0], self.tmp_train[1]), 0).numpy() # (batch_size, n_features) : 174, 47
        self.max = np.max(train_mx, 0)
        self.min = np.min(train_mx, 0)
        self.peak_hold = np.mean(train_mx, 0)
        # train set의 min과 max 정보로 전체 데이터 scaling
        self.dat = (self.rawdat - self.min) / (self.max - self.min + 1e-12)
        print('################ _pre_train ################')
        print(f'self.tmp_train X: {self.tmp_train[0].shape}')
        print(f'self.tmp_train Y: {self.tmp_train[1].shape}')
        print(f'train_mx (trainset의 shape) : {train_mx.shape}')
        print('self.train_set :', range(self.P+self.h-1, train))
        print('self.valid_set :', range(train, valid))
        print('self.test_set :', range(valid, self.n))
        print(f'_pre_train (정규화된 데이터 dat의 shape) : {self.dat.shape}')
        print()
        
    def _split(self, train, valid, test):
        self.train = self._batchify(self.train_set, self.h) # torch.Size([179, 20, 47]) torch.Size([179, 47])
        self.val = self._batchify(self.valid_set, self.h)
        self.test = self._batchify(self.test_set, self.h)
        if (train==valid):
            self.val = self.test
        
    def _batchify(self, idx_set, horizon, useraw=False):
        n = len(idx_set) # idx_set은 dataset을 의미
        Y = torch.zeros((n, self.m))
        X = torch.zeros((n, self.P, self.m))
        
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P # end 시점에서 window size만큼 제외
            ## normalization을 하지 않는 것
            if useraw: 
                X[i,:self.P,:] = torch.from_numpy(self.rawdat[start:end, :])
                Y[i,:] = torch.from_numpy(self.rawdat[idx_set[i], :])
            ## trainset의 min max 정보를 활용하여 scaling한 뒤 자른 것
            else:
                his_window = self.dat[start:end, :]
                X[i, :self.P, :] = torch.from_numpy(his_window) # size (window, m)
                Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]
        
    def get_batches(self, data, batch_size, shuffle=True):
        inputs = data[0]
        targets = data[1]
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt,:]
            Y = targets[excerpt,:]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()
            model_inputs = Variable(X)

            data = [model_inputs, Variable(Y)]
            yield data
            start_idx += batch_size