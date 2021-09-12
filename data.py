import matplotlib.pyplot as plt
import librosa, librosa.display
import os
import numpy as np
import torch
from  torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
#import librosa
import random
from matplotlib import pyplot as plt
import copy

np.random.seed(42)

def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1
    #return (seq-np.min(seq))/(np.max(seq)-np.min(seq))    
    #return librosa.util.normalize(seq,axis=1)


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def load_data(opt):
    train_dataset=None
    test_dataset=None
    val_dataset=None
    test_N_dataset=None
    test_S_dataset = None
    test_V_dataset = None
    test_F_dataset = None
    test_Q_dataset = None

    if opt.dataset=="ecg":

        #- signal
        N_samples_s = np.load(os.path.join(opt.dataroot, "N_samples.npy")) #NxCxL
        S_samples_s = np.load(os.path.join(opt.dataroot, "S_samples.npy"))
        V_samples_s = np.load(os.path.join(opt.dataroot, "V_samples.npy"))
        F_samples_s = np.load(os.path.join(opt.dataroot, "F_samples.npy"))
        Q_samples_s = np.load(os.path.join(opt.dataroot, "Q_samples.npy"))
        
        #- freq
        N_samples_f = np.load(os.path.join(opt.dataroot,'n_spectrogram.npy'))
        S_samples_f = np.load(os.path.join(opt.dataroot,'s_spectrogram.npy'))
        V_samples_f = np.load(os.path.join(opt.dataroot,'v_spectrogram.npy'))
        F_samples_f = np.load(os.path.join(opt.dataroot,'f_spectrogram.npy'))
        Q_samples_f = np.load(os.path.join(opt.dataroot,'q_spectrogram.npy'))
        
        #N_samples_s = N_samples_s[0:100]
        #N_samples_f = N_samples_f[0:100]
        

        # normalize signal
        for i in range(N_samples_s.shape[0]):
            for j in range(opt.nc):
                N_samples_s[i][j]=normalize(N_samples_s[i][j][:])
        N_samples_s=N_samples_s[:,:opt.nc,:]

        for i in range(S_samples_s.shape[0]):
            for j in range(opt.nc):
                S_samples_s[i][j] = normalize(S_samples_s[i][j][:])
        S_samples_s = S_samples_s[:, :opt.nc, :]

        for i in range(V_samples_s.shape[0]):
            for j in range(opt.nc):
                V_samples_s[i][j] = normalize(V_samples_s[i][j][:])
        V_samples_s = V_samples_s[:, :opt.nc, :]

        for i in range(F_samples_s.shape[0]):
            for j in range(opt.nc):
                F_samples_s[i][j] = normalize(F_samples_s[i][j][:])
        F_samples_s = F_samples_s[:, :opt.nc, :]

        for i in range(Q_samples_s.shape[0]):
            for j in range(opt.nc):
                Q_samples_s[i][j] = normalize(Q_samples_s[i][j][:])
        Q_samples_s = Q_samples_s[:, :opt.nc, :]
        
        # normalize freq
        for i in range(N_samples_f.shape[0]):
            N_samples_f[i] = normalize(N_samples_f[i])
                    
        for i in range(S_samples_f.shape[0]):
            S_samples_f[i] = normalize(S_samples_f[i])

        for i in range(V_samples_f.shape[0]):
            V_samples_f[i] = normalize(V_samples_f[i])
            
        for i in range(F_samples_f.shape[0]):
            F_samples_f[i] = normalize(F_samples_f[i])

        for i in range(Q_samples_f.shape[0]):
            Q_samples_f[i] = normalize(Q_samples_f[i])    
    


        # train / test
        #- signal
        test_N_s, test_N_y_s, train_N_s, train_N_y_s = getFloderK(N_samples_s,opt.folder,0)
        test_S_s, test_S_y_s = S_samples_s, np.ones((S_samples_s.shape[0], 1))
        test_V_s, test_V_y_s = V_samples_s, np.ones((V_samples_s.shape[0], 1))
        test_F_s, test_F_y_s = F_samples_s, np.ones((F_samples_s.shape[0], 1))
        test_Q_s, test_Q_y_s = Q_samples_s, np.ones((Q_samples_s.shape[0], 1))
        
        #- freq
        test_N_f, test_N_y_f, train_N_f, train_N_y_f = getFloderK(N_samples_f,opt.folder,0)
        test_S_f, test_S_y_f = S_samples_f, np.ones((S_samples_f.shape[0], 1))
        test_V_f, test_V_y_f = V_samples_f, np.ones((V_samples_f.shape[0], 1))
        test_F_f, test_F_y_f = F_samples_f, np.ones((F_samples_f.shape[0], 1))
        test_Q_f, test_Q_y_f = Q_samples_f, np.ones((Q_samples_f.shape[0], 1))        


        # train / val
        #- signal
        train_N_s, val_N_s, train_N_y_s, val_N_y_s = getPercent(train_N_s, train_N_y_s, 0.1, 0)
        test_S_s, val_S_s, test_S_y_s, val_S_y_s = getPercent(test_S_s, test_S_y_s, 0.1, 0)
        test_V_s, val_V_s, test_V_y_s, val_V_y_s = getPercent(test_V_s, test_V_y_s, 0.1, 0)
        test_F_s, val_F_s, test_F_y_s, val_F_y_s = getPercent(test_F_s, test_F_y_s, 0.1, 0)
        test_Q_s, val_Q_s, test_Q_y_s, val_Q_y_s = getPercent(test_Q_s, test_Q_y_s, 0.1, 0)

        val_data_s=np.concatenate([val_N_s,val_S_s,val_V_s,val_F_s,val_Q_s])
        val_y_s=np.concatenate([val_N_y_s,val_S_y_s,val_V_y_s,val_F_y_s,val_Q_y_s])
        
        #- freq
        train_N_f, val_N_f, train_N_y_f, val_N_y_f = getPercent(train_N_f, train_N_y_f, 0.1, 0)
        test_S_f, val_S_f, test_S_y_f, val_S_y_f = getPercent(test_S_f, test_S_y_f, 0.1, 0)
        test_V_f, val_V_f, test_V_y_f, val_V_y_f = getPercent(test_V_f, test_V_y_f, 0.1, 0)
        test_F_f, val_F_f, test_F_y_f, val_F_y_f = getPercent(test_F_f, test_F_y_f, 0.1, 0)
        test_Q_f, val_Q_f, test_Q_y_f, val_Q_y_f = getPercent(test_Q_f, test_Q_y_f, 0.1, 0)

        val_data_f=np.concatenate([val_N_f,val_S_f,val_V_f,val_F_f,val_Q_f])
        val_y_f=np.concatenate([val_N_y_f,val_S_y_f,val_V_y_f,val_F_y_f,val_Q_y_f])


        # train_N = np.stack((train_N_s, train_N_f), axis=0)
        # train_N_y = np.stack((train_N_y_s, train_N_y_f), axis=0)
        # test_N = np.stack((test_N_s, test_N_f), axis=0)

        print("\n############ signal dataset ############")
        print("train_s data size:{}".format(train_N_s.shape))
        print("val_s data size:{}".format(val_data_s.shape))
        print("test_s N data size:{}".format(test_N_s.shape))
        print("test_s S data size:{}".format(test_S_s.shape))
        print("test_s V data size:{}".format(test_V_s.shape))
        print("test_s F data size:{}".format(test_F_s.shape))
        print("test_s Q data size:{}".format(test_Q_s.shape))

        print("\n############ frequency dataset ############")
        print("train_f data size:{}".format(train_N_f.shape))
        print("val_f data size:{}".format(val_data_f.shape))
        print("test_f N data size:{}".format(test_N_f.shape))
        print("test_f S data size:{}".format(test_S_f.shape))
        print("test_f V data size:{}".format(test_V_f.shape))
        print("test_f F data size:{}".format(test_F_f.shape))
        print("test_f Q data size:{}".format(test_Q_f.shape))
        
        if not opt.istest and opt.n_aug>0:
             length = len(train_N_s)

             #signal-noise freq-masking
             train_N_s,train_N_y_s=data_aug(train_N_s,train_N_y_s,times=opt.n_aug,aug_type='s')
             train_N_f,train_N_y_f=data_aug(train_N_f,train_N_y_f,times=opt.n_aug,aug_type='fm')
             
             #signal-noise freq-noise
             train_N_s,train_N_y_s=data_aug(train_N_s,train_N_y_s,times=opt.n_aug,aug_type='s')
             train_N_f,train_N_y_f=data_aug(train_N_f,train_N_y_f,times=opt.n_aug,aug_type='fn')
                          
             print("[signal] after aug, train data size:{}".format(train_N_s.shape))
             print("[frequency] after aug, train data size:{}".format(train_N_f.shape))
             
             #print(train_N_s.shape, train_N_f.shape)
             
             #save fig before augemntation
             fig = plt.figure(figsize=(5,10))
             plt.subplot(2,1,1)
             plt.plot(train_N_s[0][0])
             
             plt.subplot(2,1,2)
             img = librosa.display.specshow(train_N_f[0][0], sr=360, hop_length = 2, y_axis="linear", x_axis="time")
            
             fig.savefig("aug/real{0}.png".format(0))
             
             #save fig after augmentation
             fig = plt.figure(figsize=(5,10))
             plt.subplot(2,1,1)
             plt.plot(train_N_s[length][0])
             
             plt.subplot(2,1,2)
             img = librosa.display.specshow(train_N_f[length][0], sr=360, hop_length = 2, y_axis="linear", x_axis="time")
            
             fig.savefig("aug/aug{0}.png".format(0))
             
             
             


        train_dataset_s  = TensorDataset(torch.Tensor(train_N_s),torch.Tensor(train_N_y_s))
        val_dataset_s    = TensorDataset(torch.Tensor(val_data_s), torch.Tensor(val_y_s))
        test_N_dataset_s = TensorDataset(torch.Tensor(test_N_s), torch.Tensor(test_N_y_s))
        test_S_dataset_s = TensorDataset(torch.Tensor(test_S_s), torch.Tensor(test_S_y_s))
        test_V_dataset_s = TensorDataset(torch.Tensor(test_V_s), torch.Tensor(test_V_y_s))
        test_F_dataset_s = TensorDataset(torch.Tensor(test_F_s), torch.Tensor(test_F_y_s))
        test_Q_dataset_s = TensorDataset(torch.Tensor(test_Q_s), torch.Tensor(test_Q_y_s))

        train_dataset_f  = TensorDataset(torch.Tensor(train_N_f),torch.Tensor(train_N_y_f))
        val_dataset_f    = TensorDataset(torch.Tensor(val_data_f), torch.Tensor(val_y_f))
        test_N_dataset_f = TensorDataset(torch.Tensor(test_N_f), torch.Tensor(test_N_y_f))
        test_S_dataset_f = TensorDataset(torch.Tensor(test_S_f), torch.Tensor(test_S_y_f))
        test_V_dataset_f = TensorDataset(torch.Tensor(test_V_f), torch.Tensor(test_V_y_f))
        test_F_dataset_f = TensorDataset(torch.Tensor(test_F_f), torch.Tensor(test_F_y_f))
        test_Q_dataset_f = TensorDataset(torch.Tensor(test_Q_f), torch.Tensor(test_Q_y_f))


    # assert (train_dataset is not None  and test_dataset is not None and val_dataset is not None)

    dataloader = {"train": DataLoader(#------------------------------------------------signal
                        dataset=MultimodalDataset(train_dataset_s, train_dataset_f),  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=True),
                    "val": DataLoader(
                        dataset=MultimodalDataset(val_dataset_s, val_dataset_f),  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=False),
                    "test_N":DataLoader(
                            dataset=MultimodalDataset(test_N_dataset_s, test_N_dataset_f),  # torch TensorDataset format
                            batch_size=opt.batchsize,  # mini batch size
                            shuffle=True,
                            num_workers=int(opt.workers),
                            drop_last=False),
                    "test_S": DataLoader(
                        dataset=MultimodalDataset(test_S_dataset_s, test_S_dataset_f),  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=False),
                    "test_V": DataLoader(
                        dataset=MultimodalDataset(test_V_dataset_s, test_V_dataset_f),  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=False),
                    "test_F": DataLoader(
                        dataset=MultimodalDataset(test_F_dataset_s, test_F_dataset_f),  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=False),
                    "test_Q": DataLoader(
                        dataset=MultimodalDataset(test_Q_dataset_s, test_Q_dataset_f),  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=False),
                    }
    return dataloader


def getFloderK(data,folder,label):
    normal_cnt = data.shape[0]
    folder_num = int(normal_cnt / 5)
    folder_idx = folder * folder_num

    folder_data = data[folder_idx:folder_idx + folder_num]

    remain_data = np.concatenate([data[:folder_idx], data[folder_idx + folder_num:]])
    if label==0:
        folder_data_y = np.zeros((folder_data.shape[0], 1))
        remain_data_y=np.zeros((remain_data.shape[0], 1))
    elif label==1:
        folder_data_y = np.ones((folder_data.shape[0], 1))
        remain_data_y = np.ones((remain_data.shape[0], 1))
    else:
        raise Exception("label should be 0 or 1, get:{}".format(label))
    return folder_data,folder_data_y,remain_data,remain_data_y

def getPercent(data_x,data_y,percent,seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=percent,random_state=seed)
    return train_x, test_x, train_y, test_y

def get_full_data(dataloader):

    full_data_x=[]
    full_data_y=[]
    for batch_data in dataloader:
        batch_x,batch_y=batch_data[0],batch_data[1]
        batch_x=batch_x.numpy()
        batch_y=batch_y.numpy()

        # print(batch_x.shape)
        # assert False
        for i in range(batch_x.shape[0]):
            full_data_x.append(batch_x[i,0,:])
            full_data_y.append(batch_y[i])

    full_data_x=np.array(full_data_x)
    full_data_y=np.array(full_data_y)
    assert full_data_x.shape[0]==full_data_y.shape[0]
    print("full data size:{}".format(full_data_x.shape))
    return full_data_x,full_data_y


def data_aug(train_x,train_y,times=2, aug_type='s'):
    fig = plt.figure(figsize=(5,5))
    res_train_x=[]
    res_train_y=[]
    
    #print('aug', res_train_x.shape)
    
    for idx in range(train_x.shape[0]):
        x=train_x[idx]
        y=train_y[idx]
        #res_train_x.append(x)
        #res_train_y.append(y)

        #for i in range(times):
        #x_aug=aug_ts(x)

        if(aug_type=='s'): #signal noise
            x_aug = aug_signal_noise(x)
        elif(aug_type=='fn'): #freq noise
            x_aug = aug_freq_noise(x)
        elif(aug_type=='fm'): #freq masking
            x_aug = aug_freq_masking(x)
        
        #np.append(res_train_x, x_aug,axis=0)
        #np.append(res_train_y, y)
        res_train_x.append(x_aug)
        res_train_y.append(y)
        
        #show real/augmented img
        '''
        img = librosa.display.specshow(x_aug[0], sr=360, hop_length = 2, y_axis="linear", x_axis="time")
        fig.savefig("aug/aug{0}.png".format(idx))
        img = librosa.display.specshow(x[0], sr=360, hop_length = 2, y_axis="linear", x_axis="time")
        fig.savefig("aug/real{0}.png".format(idx))
        '''
    #res_train_x=np.array(res_train_x)
    #res_train_y=np.array(res_train_y)
    
    res_train_x = np.concatenate((train_x, np.array(res_train_x)), axis=0)
    res_train_y = np.concatenate((train_y, np.array(res_train_y)), axis=0)
    
    #print(res_train_x.shape)
    return res_train_x,res_train_y


def aug_ts(feat , T = 5, F = 5, time_mask_num = 1, freq_mask_num = 1):
   
    return feat
   

#freq masking
def aug_freq_masking(feat , T = 10, F = 10, time_mask_num = 1, freq_mask_num = 1):
    feat1 = copy.deepcopy(feat)
    feat1_size = 128
    seq_len = 128

    # time mask
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=T)
        t = int(t)
        t0 = random.randint(1, seq_len - t) 
        feat1[:, :,t0 : t0 + t] = 0
        #print(feat1[:,:,t0:t0+t])
        

    # freq mask
    for _ in range(freq_mask_num):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        f0 = random.randint(1, feat1_size - f - 95)
        feat1[:, f0 : f0 + f] = 0

    return feat1

#add noise to signal
def aug_signal_noise(x):
    sigma = random.uniform(0.01,0.03)
    noise = np.random.normal(loc=0, scale=sigma, size=x.shape)
    output = normalize(x+noise)
    return output

#add noise to freq
def aug_freq_noise(x):
    #x_1 = copy.deepcopy(x)
    sigma = random.uniform(0.01,0.03)
    noise = np.random.normal(loc=0, scale=sigma, size=x.shape)
    output = normalize(x+noise)
        
    return output