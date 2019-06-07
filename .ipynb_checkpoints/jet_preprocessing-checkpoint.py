{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Preprocessing of the Gregor Dataset\n",
    "\n",
    "## Started June 3, 2019\n",
    "### Genevieve Hayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torchvision\n",
    "import torchvision.transforms as transforms\n",
    "import torch.nn.functional as F\n",
    "import seaborn as sb\n",
    "import matplotlib.pylab as plt\n",
    "#import matplotlib as mpl\n",
    "import h5py\n",
    "import tables\n",
    "import math\n",
    "import time\n",
    "import sklearn.metrics as sklm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "run_gpu=False\n",
    "if run_gpu:\n",
    "    device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")\n",
    "else:\n",
    "    device = torch.device(\"cpu\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "#for massless particles\n",
    "#define function to transform px, py, pz ---> pt, eta and phi\n",
    "def get_pt_eta_phi(px, py, pz):\n",
    "    pt = math.sqrt(px**2 + py**2) #transverse momentum\n",
    "    phi = None\n",
    "    eta = None\n",
    "    if (px == 0 and py == 0 and pz == 0):\n",
    "        theta = 0\n",
    "    else:\n",
    "        theta = math.atan2(pt, pz)\n",
    "    cos_theta = math.cos(theta)\n",
    "    if cos_theta**2 < 1:\n",
    "        eta = -0.5 * math.log((1 - cos_theta) / (1 + cos_theta))\n",
    "    elif pz == 0.0:\n",
    "        eta = 0\n",
    "    else:\n",
    "        eta = 10e10\n",
    "    if(px == 0 and py == 0):\n",
    "        phi = 0\n",
    "    else:\n",
    "        phi = math.atan2(py, px)\n",
    "    return pt, eta, phi"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "#define function to transform E, px, py, pz ---> pt, eta and phi\n",
    "def get_massive_pt_eta_phi(E, px, py, pz):\n",
    "    pt = math.sqrt(px**2 + py**2) #transverse momentum\n",
    "    phi = None\n",
    "    eta = None\n",
    "    if (px == 0 and py == 0 and pz == 0):\n",
    "        theta = 0\n",
    "    else:\n",
    "        theta = math.atan2(pt, pz)\n",
    "    cos_theta = math.cos(theta)\n",
    "    if cos_theta**2 < 1:\n",
    "        eta = 0.5 * math.log((E + pz)/(E - pz)) #this accounts for the mass\n",
    "    elif pz == 0.0:\n",
    "        eta = 0\n",
    "    else:\n",
    "        eta = 10e10\n",
    "    if(px == 0 and py == 0):\n",
    "        phi = 0\n",
    "    else:\n",
    "        phi = math.atan2(py, px)\n",
    "    return pt, eta, phi"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "################################\n",
    "# Training dataset loading class\n",
    "################################\n",
    "\n",
    "from torch.autograd import Variable\n",
    "from torch.utils.data import Dataset, DataLoader, random_split\n",
    "\n",
    "class TrainDataset(Dataset):\n",
    "    # Initialize our data, download, etc.  \n",
    "    def __init__(self):\n",
    "        #open hdf5 file for reading\n",
    "        hdf_train = pd.HDFStore('train.h5',mode='r')\n",
    "        hdf_train.keys()\n",
    "        table_train = hdf_train.get('/table')\n",
    "        \n",
    "        #numpy representation of DataFrame\n",
    "        array_train = table_train.values\n",
    "        \n",
    "        self.len = array_train.shape[0]\n",
    "        self.width = array_train.shape[1]\n",
    "        self.x_data = torch.from_numpy(array_train[:,0:800]).float() #x data is only the 4 vectors of the constituents \n",
    "        self.y_data = torch.from_numpy(array_train[:,-1]).long() #y data is only the labels (0=QCD, 1=top)\n",
    "        \n",
    "    def __getitem__(self,index):\n",
    "        return self.x_data[index],self.y_data[index]\n",
    "    \n",
    "    def __len__(self):\n",
    "        return self.len\n",
    "    \n",
    "# Validation dataset loading class\n",
    "class ValDataset(Dataset):\n",
    "    # Initialize our data, download, etc.\n",
    "    def __init__(self):\n",
    "        #open hdf5 file for reading\n",
    "        hdf_val = pd.HDFStore('val.h5',mode='r')\n",
    "        hdf_val.keys()\n",
    "        table_val = hdf_val.get('/table')\n",
    "        \n",
    "        #numpy representation of DataFrame\n",
    "        array_val = table_val.values\n",
    "        \n",
    "        self.len = array_val.shape[0]\n",
    "        self.x_data = torch.from_numpy(array_val[:,0:800]).float()\n",
    "        self.y_data = torch.from_numpy(array_val[:,-1]).long()\n",
    "        \n",
    "    def __getitem__(self,index):\n",
    "        return self.x_data[index],self.y_data[index]\n",
    "    \n",
    "    def __len__(self):\n",
    "        return self.len"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "training_dataset = TrainDataset()\n",
    "validation_dataset = ValDataset()\n",
    "train_len = training_dataset.len\n",
    "val_len = validation_dataset.len"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "# print(type(training_dataset))\n",
    "# print(training_dataset.len)\n",
    "# print(training_dataset.width)\n",
    "# print(len(training_dataset[0][0][:]))\n",
    "# print(training_dataset.x_data[13][2])\n",
    "# print(training_dataset.y_data.shape)\n",
    "# pt1,eta1,phi1 = get_pt_eta_phi(training_dataset[0][0][1],training_dataset[0][0][2],training_dataset[0][0][3])\n",
    "# print(\"pt = {:.4f} \\neta = {:.4f} \\nphi = {:.4f}\\n\".format(pt1,eta1,phi1))\n",
    "# pt2,eta2,phi2 = get_massive_pt_eta_phi(training_dataset[0][0][0],training_dataset[0][0][1],training_dataset[0][0][2],training_dataset[0][0][3])\n",
    "# print(\"pt_m = {:.4f} \\neta_m = {:.4f} \\nphi_m = {:.4f}\\n\".format(pt2,eta2,phi2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#######################################################\n",
    "#Transform dataset from E, px, py, pz ---> pT, eta, phi \n",
    "#######################################################\n",
    "#Training Dataset\n",
    "number_of_jets = train_len\n",
    "\n",
    "start_time = time.time()\n",
    "\n",
    "#training_list = []\n",
    "training_list_m = []\n",
    "\n",
    "#training_preprocessed = []\n",
    "training_preprocessed_m = []\n",
    "\n",
    "#loop through each jet \n",
    "for j in range(number_of_jets):\n",
    "    #training_list = []\n",
    "    training_list_m = []\n",
    "    #loop through each feature of the jet constituents within each jet\n",
    "    for i in range(0,len(training_dataset[0][0][:])-3,4):\n",
    "        E = training_dataset.x_data[j][i]\n",
    "        px = training_dataset.x_data[j][i+1]\n",
    "        py = training_dataset.x_data[j][i+2]\n",
    "        pz = training_dataset.x_data[j][i+3]\n",
    "\n",
    "        #pt,eta,phi = get_pt_eta_phi(px,py,pz)\n",
    "        pt_m,eta_m,phi_m = get_massive_pt_eta_phi(E,px,py,pz)\n",
    "        \n",
    "        training_list_m.append(pt_m)\n",
    "        training_list_m.append(eta_m)\n",
    "        training_list_m.append(phi_m)\n",
    "        #training_list.append(pt)\n",
    "        #training_list.append(eta)\n",
    "        #training_list.append(phi)\n",
    "        \n",
    "    if j > 0:\n",
    "        #training_preprocessed = np.vstack([training_preprocessed, training_list])\n",
    "        training_preprocessed_m = np.vstack([training_preprocessed_m, training_list_m])\n",
    "    else:\n",
    "        #training_preprocessed = training_list\n",
    "        training_preprocessed_m = training_list_m\n",
    "\n",
    "end_time = time.time()   \n",
    "print(\"--- {} seconds ---\".format(end_time-start_time)) \n",
    "print(\"--- {} min and {} sec ---\".format(math.floor((end_time-start_time)/60),math.ceil((end_time-start_time)%60)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#######################################################\n",
    "#Transform dataset from E, px, py, pz ---> pT, eta, phi \n",
    "#######################################################\n",
    "#Validation Dataset\n",
    "number_of_jets2 = val_len\n",
    "\n",
    "start_time = time.time()\n",
    "\n",
    "#training_list = []\n",
    "validation_list_m = []\n",
    "\n",
    "#training_preprocessed = []\n",
    "validation_preprocessed_m = []\n",
    "\n",
    "#loop through each jet \n",
    "for j in range(number_of_jets2):\n",
    "    #training_list = []\n",
    "    validation_list_m = []\n",
    "    #loop through each feature of the jet constituents within each jet\n",
    "    for i in range(0,len(validation_dataset[0][0][:])-3,4):\n",
    "        E = validation_dataset.x_data[j][i]\n",
    "        px = validation_dataset.x_data[j][i+1]\n",
    "        py = validation_dataset.x_data[j][i+2]\n",
    "        pz = validation_dataset.x_data[j][i+3]\n",
    "\n",
    "        #pt,eta,phi = get_pt_eta_phi(px,py,pz)\n",
    "        pt_m,eta_m,phi_m = get_massive_pt_eta_phi(E,px,py,pz)\n",
    "        \n",
    "        validation_list_m.append(pt_m)\n",
    "        validation_list_m.append(eta_m)\n",
    "        validation_list_m.append(phi_m)\n",
    "        \n",
    "    if j > 0:\n",
    "\n",
    "        validation_preprocessed_m = np.vstack([validation_preprocessed_m, validation_list_m])\n",
    "    else:\n",
    "        validation_preprocessed_m = validation_list_m\n",
    "\n",
    "end_time = time.time()   \n",
    "print(\"--- {} seconds ---\".format(end_time-start_time)) \n",
    "print(\"--- {} min and {} sec ---\".format(math.floor((end_time-start_time)/60),math.ceil((end_time-start_time)%60)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#########################\n",
    "#Adding the truth column\n",
    "#########################\n",
    "\n",
    "len_pp = len(training_preprocessed_m)\n",
    "totalnumpy = training_dataset.y_data[0:len_pp].numpy()\n",
    "total_train = np.hstack((training_preprocessed_m, np.atleast_2d(totalnumpy).T))\n",
    "\n",
    "len_pp_val = len(validation_preprocessed_m)\n",
    "totalnumpy_val = validation_dataset.y_data[0:len_pp].numpy()\n",
    "total_val = np.hstack((validation_preprocessed_m, np.atleast_2d(totalnumpy_).T))\n",
    "\n",
    "print(len(total_train))\n",
    "print(len(total_train.T))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Approximate number of hours it will take for all of the training set to be transformed:\n",
    "#(1211000/5000)*64/60/60"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#######################################################\n",
    "#Create labels for each columns of preprocessed dataset\n",
    "#######################################################\n",
    "#Training dataset\n",
    "train_labels = []\n",
    "num_features_train = len(training_preprocessed_m[0][:])\n",
    "\n",
    "for i in range(int(num_features_train/3)):\n",
    "    num = str(i)\n",
    "    label_pT = 'pT_'+num\n",
    "    label_eta = 'eta_'+num\n",
    "    label_phi = 'phi_'+num\n",
    "    \n",
    "    train_labels.append(label_pT)\n",
    "    train_labels.append(label_eta)\n",
    "    train_labels.append(label_phi)\n",
    "\n",
    "train_labels.append('is_signal_top')\n",
    "\n",
    "#Validation dataset\n",
    "val_labels = []\n",
    "num_features_val = len(validation_preprocessed_m[0][:])\n",
    "\n",
    "for i in range(int(num_features_val/3)):\n",
    "    num = str(i)\n",
    "    val_label_pT = 'pT_'+num\n",
    "    val_label_eta = 'eta_'+num\n",
    "    val_label_phi = 'phi_'+num\n",
    "    \n",
    "    val_labels.append(val_label_pT)\n",
    "    val_labels.append(val_label_eta)\n",
    "    val_labels.append(va_label_phi)\n",
    "\n",
    "val_labels.append('is_signal_top')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(type(labels))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "###################################\n",
    "# Create a dataframe with labels\n",
    "###################################\n",
    "#Training dataset\n",
    "df = pd.DataFrame(total, columns=labels)\n",
    "df.to_hdf('prepro_train1.h5', key='table', mode='w')\n",
    "\n",
    "#Validation dataset\n",
    "df = pd.DataFrame(total_val, columns=labels)\n",
    "df.to_hdf('prepro_val1.h5', key='table', mode='w')\n",
    "\n",
    "#ppx = pd.Series({'ppx': training_preprocessed}) #series\n",
    "#ppx.to_hdf('preprocessed_train.h5',key='table')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# # read hdf file which has been preprocessed\n",
    "# hdf_pptrain = pd.HDFStore('prepro_train1.h5',mode='r')\n",
    "# hdf_pptrain.keys()\n",
    "# data = hdf_pptrain.get('/table')\n",
    "# print(data.shape) #type: <class 'pandas.core.frame.DataFrame'>\n",
    "# #print(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# hdf_train.close()\n",
    "# hdf_pptrain.close()\n",
    "# hdf_pptrain.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
