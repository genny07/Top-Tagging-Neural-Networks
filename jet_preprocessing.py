#!/usr/bin/env python
# coding: utf-8

# # Preprocessing of the Gregor Dataset
# 
# ## Started June 3, 2019
# ### Genevieve Hayes

# In[23]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import seaborn as sb
import matplotlib.pylab as plt
#import matplotlib as mpl
import h5py
import tables
import math
import time
import sklearn.metrics as sklm


# In[24]:


run_gpu=False
if run_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


# In[25]:


#for massless particles
#define function to transform px, py, pz ---> pt, eta and phi
def get_pt_eta_phi(px, py, pz):
    pt = math.sqrt(px**2 + py**2) #transverse momentum
    phi = None
    eta = None
    if (px == 0 and py == 0 and pz == 0):
        theta = 0
    else:
        theta = math.atan2(pt, pz)
    cos_theta = math.cos(theta)
    if cos_theta**2 < 1:
        eta = -0.5 * math.log((1 - cos_theta) / (1 + cos_theta))
    elif pz == 0.0:
        eta = 0
    else:
        eta = 10e10
    if(px == 0 and py == 0):
        phi = 0
    else:
        phi = math.atan2(py, px)
    return pt, eta, phi


# In[26]:


#define function to transform E, px, py, pz ---> pt, eta and phi
def get_massive_pt_eta_phi(E, px, py, pz):
    pt = math.sqrt(px**2 + py**2) #transverse momentum
    phi = None
    eta = None
    if (px == 0 and py == 0 and pz == 0):
        theta = 0
    else:
        theta = math.atan2(pt, pz)
    cos_theta = math.cos(theta)
    if cos_theta**2 < 1:
        eta = 0.5 * math.log((E + pz)/(E - pz)) #this accounts for the mass
    elif pz == 0.0:
        eta = 0
    else:
        eta = 10e10
    if(px == 0 and py == 0):
        phi = 0
    else:
        phi = math.atan2(py, px)
    return pt, eta, phi


# In[27]:


################################
# Training dataset loading class
################################

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split

class TrainDataset(Dataset):
    # Initialize our data, download, etc.  
    def __init__(self):
        #open hdf5 file for reading
        hdf_train = pd.HDFStore('train.h5',mode='r')
        hdf_train.keys()
        table_train = hdf_train.get('/table')
        
        #numpy representation of DataFrame
        array_train = table_train.values
        
        self.len = array_train.shape[0]
        self.width = array_train.shape[1]
        self.x_data = torch.from_numpy(array_train[:,0:800]).float() #x data is only the 4 vectors of the constituents 
        self.y_data = torch.from_numpy(array_train[:,-1]).long() #y data is only the labels (0=QCD, 1=top)
        
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.len
    
# Validation dataset loading class
class ValDataset(Dataset):
    # Initialize our data, download, etc.
    def __init__(self):
        #open hdf5 file for reading
        hdf_val = pd.HDFStore('val.h5',mode='r')
        hdf_val.keys()
        table_val = hdf_val.get('/table')
        
        #numpy representation of DataFrame
        array_val = table_val.values
        
        self.len = array_val.shape[0]
        self.x_data = torch.from_numpy(array_val[:,0:800]).float()
        self.y_data = torch.from_numpy(array_val[:,-1]).long()
        
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.len


# In[28]:


training_dataset = TrainDataset()
validation_dataset = ValDataset()
train_len = training_dataset.len
val_len = validation_dataset.len


# In[29]:


# print(type(training_dataset))
# print(training_dataset.len)
# print(training_dataset.width)
# print(len(training_dataset[0][0][:]))
# print(training_dataset.x_data[13][2])
# print(training_dataset.y_data.shape)
# pt1,eta1,phi1 = get_pt_eta_phi(training_dataset[0][0][1],training_dataset[0][0][2],training_dataset[0][0][3])
# print("pt = {:.4f} \neta = {:.4f} \nphi = {:.4f}\n".format(pt1,eta1,phi1))
# pt2,eta2,phi2 = get_massive_pt_eta_phi(training_dataset[0][0][0],training_dataset[0][0][1],training_dataset[0][0][2],training_dataset[0][0][3])
# print("pt_m = {:.4f} \neta_m = {:.4f} \nphi_m = {:.4f}\n".format(pt2,eta2,phi2))


# In[ ]:


#######################################################
#Transform dataset from E, px, py, pz ---> pT, eta, phi 
#######################################################
#Training Dataset
number_of_jets = train_len

start_time = time.time()

#training_list = []
training_list_m = []

#training_preprocessed = []
training_preprocessed_m = []

#loop through each jet 
for j in range(number_of_jets):
    #training_list = []
    training_list_m = []
    #loop through each feature of the jet constituents within each jet
    for i in range(0,len(training_dataset[0][0][:])-3,4):
        E = training_dataset.x_data[j][i]
        px = training_dataset.x_data[j][i+1]
        py = training_dataset.x_data[j][i+2]
        pz = training_dataset.x_data[j][i+3]

        #pt,eta,phi = get_pt_eta_phi(px,py,pz)
        pt_m,eta_m,phi_m = get_massive_pt_eta_phi(E,px,py,pz)
        
        training_list_m.append(pt_m)
        training_list_m.append(eta_m)
        training_list_m.append(phi_m)
        #training_list.append(pt)
        #training_list.append(eta)
        #training_list.append(phi)
        
    if j > 0:
        #training_preprocessed = np.vstack([training_preprocessed, training_list])
        training_preprocessed_m = np.vstack([training_preprocessed_m, training_list_m])
    else:
        #training_preprocessed = training_list
        training_preprocessed_m = training_list_m

end_time = time.time()   
print("--- {} seconds ---".format(end_time-start_time)) 
print("--- {} min and {} sec ---".format(math.floor((end_time-start_time)/60),math.ceil((end_time-start_time)%60)))


# In[ ]:


#######################################################
#Transform dataset from E, px, py, pz ---> pT, eta, phi 
#######################################################
#Validation Dataset
number_of_jets2 = val_len

start_time = time.time()

#training_list = []
validation_list_m = []

#training_preprocessed = []
validation_preprocessed_m = []

#loop through each jet 
for j in range(number_of_jets2):
    #training_list = []
    validation_list_m = []
    #loop through each feature of the jet constituents within each jet
    for i in range(0,len(validation_dataset[0][0][:])-3,4):
        E = validation_dataset.x_data[j][i]
        px = validation_dataset.x_data[j][i+1]
        py = validation_dataset.x_data[j][i+2]
        pz = validation_dataset.x_data[j][i+3]

        #pt,eta,phi = get_pt_eta_phi(px,py,pz)
        pt_m,eta_m,phi_m = get_massive_pt_eta_phi(E,px,py,pz)
        
        validation_list_m.append(pt_m)
        validation_list_m.append(eta_m)
        validation_list_m.append(phi_m)
        
    if j > 0:

        validation_preprocessed_m = np.vstack([validation_preprocessed_m, validation_list_m])
    else:
        validation_preprocessed_m = validation_list_m

end_time = time.time()   
print("--- {} seconds ---".format(end_time-start_time)) 
print("--- {} min and {} sec ---".format(math.floor((end_time-start_time)/60),math.ceil((end_time-start_time)%60)))


# In[ ]:


#########################
#Adding the truth column
#########################

len_pp = len(training_preprocessed_m)
totalnumpy = training_dataset.y_data[0:len_pp].numpy()
total_train = np.hstack((training_preprocessed_m, np.atleast_2d(totalnumpy).T))

len_pp_val = len(validation_preprocessed_m)
totalnumpy_val = validation_dataset.y_data[0:len_pp].numpy()
total_val = np.hstack((validation_preprocessed_m, np.atleast_2d(totalnumpy_).T))

print(len(total_train))
print(len(total_train.T))


# In[ ]:


#Approximate number of hours it will take for all of the training set to be transformed:
#(1211000/5000)*64/60/60


# In[ ]:


#######################################################
#Create labels for each columns of preprocessed dataset
#######################################################
#Training dataset
train_labels = []
num_features_train = len(training_preprocessed_m[0][:])

for i in range(int(num_features_train/3)):
    num = str(i)
    label_pT = 'pT_'+num
    label_eta = 'eta_'+num
    label_phi = 'phi_'+num
    
    train_labels.append(label_pT)
    train_labels.append(label_eta)
    train_labels.append(label_phi)

train_labels.append('is_signal_top')

#Validation dataset
val_labels = []
num_features_val = len(validation_preprocessed_m[0][:])

for i in range(int(num_features_val/3)):
    num = str(i)
    val_label_pT = 'pT_'+num
    val_label_eta = 'eta_'+num
    val_label_phi = 'phi_'+num
    
    val_labels.append(val_label_pT)
    val_labels.append(val_label_eta)
    val_labels.append(va_label_phi)

val_labels.append('is_signal_top')


# In[ ]:


print(type(labels))


# In[ ]:


###################################
# Create a dataframe with labels
###################################
#Training dataset
df = pd.DataFrame(total, columns=labels)
df.to_hdf('prepro_train1.h5', key='table', mode='w')

#Validation dataset
df = pd.DataFrame(total_val, columns=labels)
df.to_hdf('prepro_val1.h5', key='table', mode='w')

#ppx = pd.Series({'ppx': training_preprocessed}) #series
#ppx.to_hdf('preprocessed_train.h5',key='table')


# In[ ]:


# # read hdf file which has been preprocessed
# hdf_pptrain = pd.HDFStore('prepro_train1.h5',mode='r')
# hdf_pptrain.keys()
# data = hdf_pptrain.get('/table')
# print(data.shape) #type: <class 'pandas.core.frame.DataFrame'>
# #print(data)


# In[ ]:


# hdf_train.close()
# hdf_pptrain.close()
# hdf_pptrain.close()


# In[ ]:




