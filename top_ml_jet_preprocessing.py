import numpy as np
import logging
from random import shuffle, seed
import math
from top_ml import plot_histo, plot_all_jets
from keras.preprocessing import sequence
import pdb
import sys
from top_ml.column_definition import *
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

import seaborn as sns

def load_data(file_name):
    print("loading " + file_name)
    x = np.load(file_name, encoding='latin1')
    x_data = x['data']
    #x_data = x_data[0:1000,:]
    return x_data


def test_inputs(labels, jet_masses, jets, jet_constits):
    """ Small tests to make sure our inputs make sense """
    # check that jet constits is divisible by three
    if(len(jet_constits[0]) % 3 == 0):
        print("Unit test passed: Jet constituent info is exactly divisible by 3")
    else:
        print("Error: length of jet constituents " +
              str(len(jet_constits[0])) + " is not divisible by 3")
        return

    # check that jet pt is always bigger than jet constituent pt
    for i in range(len(jets)):
        if(jets[i][0] < jet_constits[i][0]):
            print("Error, jet pt < jet constituent pt")
            print("Jet pt:")
            print(jets[i][0])
            print("Jet constit pt:")
            print(jet_constits[i][0])
            return
    return


def get_px_py_pz(pt, eta, phi):
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    return px, py, pz


def get_pt_eta_phi(px, py, pz):
    pt = math.sqrt(px**2 + py**2)
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


def rotate_x(px, py, pz, angle):
    pyy = py * math.cos(angle) - pz * math.sin(angle)
    pzz = pz * math.cos(angle) + py * math.sin(angle)
    return px, pyy, pzz


def rotate_function(pt, eta, phi, theta):
    px, py, pz = get_px_py_pz(pt, eta, phi)
    px, py, pz = rotate_x(px, py, pz, theta)
    pt, eta, phi = get_pt_eta_phi(px, py, pz)
    return pt, eta, phi

def lorentz_boost(bx,by,bz,pt,eta,phi):
    px, py, pz = get_px_py_pz(pt, eta, phi)
    e = math.sqrt(px**2+py**2+pz**2) #assuming massless particle
    b2 = bx*bx+by*by+bz*bz
    gamma = 1.0 / math.sqrt(1.0 - b2) # always positive, >=1
    bp = bx*px + by*py + bz*pz #dot product beta and p
    gamma2 = (gamma - 1.0)/b2 if b2 > 0 else 0.0 #always positive
    px = px + gamma2*bp*bx + gamma*bx*e;
    py = py + gamma2*bp*by + gamma*by*e;
    pz = pz + gamma2*bp*bz + gamma*bz*e;
    e = gamma*(e+bp)
    pt, eta, phi = get_pt_eta_phi(px,py,pz)
    return pt, eta, phi


def plot_distributions(labels, jet_masses, jets, jet_constits, step_name):
    """ Plot input distributions """
    show = False  # Save figures or show them
    #plot_histo(data=labels,name=step_name+"labels",title=step_name+" distribution ",xlabel="Label",ylabel="Jets",show=show)
    #plot_histo(data=jet_masses,name=step_name+"_jet_masses",title=step_name+" distribution ",xlabel="Jet Mass [GeV]",ylabel="Jets",show=show)
    #plot_histo(data=jets[:,0],name=step_name+"_jet_pt",title=step_name+" distribution ",xlabel="Jet Pt [GeV]",ylabel="Jets",show=show)
    #plot_histo(data=jets[:,1],name=step_name+"_jet_eta",title=step_name+" distribution ",xlabel="Jet Eta ",ylabel="Jets",show=show)
    #plot_histo(data=jets[:,2],name=step_name+"_jet_phi",title=step_name+" distribution ",xlabel="Jet Phi [radians]",ylabel="Jets",show=show)
    jet_constit_pt = []
    jet_constit_eta = []
    jet_constit_phi = []
    jet_constit_pt_1 = []
    jet_constit_pt_2 = []
    jet_constit_pt_3 = []
    jet_constit_pt_4 = []
    jet_constit_pt_5 = []
    for i in range(len(jet_constits)):
        for j in range(0, len(jet_constits[i]), 3):
            jet_constit_pt.append(jet_constits[i][j])
            jet_constit_eta.append(jet_constits[i][j + 1])
            jet_constit_phi.append(jet_constits[i][j + 2])
            if j == 0:
                jet_constit_pt_1.append(jet_constits[i][j])
            if j == 3:
                jet_constit_pt_2.append(jet_constits[i][j])
            if j == 6:
                jet_constit_pt_3.append(jet_constits[i][j])
            if j == 9:
                jet_constit_pt_4.append(jet_constits[i][j])
            if j == 12:
                jet_constit_pt_5.append(jet_constits[i][j])

    plot_histo(
        data=jet_constit_pt,
        name=step_name +
        "_jet_constit_pt",
        title=step_name +
        " distribution ",
        xlabel="Pt [GeV]",
        ylabel="Jet Constituents",
        show=show)
    plot_histo(
        data=jet_constit_pt_1,
        name=step_name +
        "_jet_constit_pt_1",
        title=step_name +
        " distribution ",
        xlabel="Pt [GeV]",
        ylabel="Jet Constituents",
        show=show)
    plot_histo(
        data=jet_constit_pt_2,
        name=step_name +
        "_jet_constit_pt_2",
        title=step_name +
        " distribution ",
        xlabel="Pt [GeV]",
        ylabel="Jet Constituents",
        show=show)
    plot_histo(
        data=jet_constit_pt_3,
        name=step_name +
        "_jet_constit_pt_3",
        title=step_name +
        " distribution ",
        xlabel="Pt [GeV]",
        ylabel="Jet Constituents",
        show=show)
    plot_histo(
        data=jet_constit_pt_4,
        name=step_name +
        "_jet_constit_pt_4",
        title=step_name +
        " distribution ",
        xlabel="Pt [GeV]",
        ylabel="Jet Constituents",
        show=show)
    plot_histo(
        data=jet_constit_pt_5,
        name=step_name +
        "_jet_constit_pt_5",
        title=step_name +
        " distribution ",
        xlabel="Pt [GeV]",
        ylabel="Jet Constituents",
        show=show)
    plot_histo(
        data=jet_constit_eta,
        name=step_name +
        "_jet_constit_eta",
        title=step_name +
        " distribution ",
        xlabel="Jet Eta [radians]",
        ylabel="Jet Constituents",
        show=show)
    plot_histo(
        data=jet_constit_phi,
        name=step_name +
        "_jet_constit_phi",
        title=step_name +
        " distribution ",
        xlabel="Jet Phi [radians]",
        ylabel="Jet Constituents",
        show=show)
    return


def scale_inputs(jets, jet_constits):
    """ Scales the pt of the jet and jet constituents """
    print("Scaling inputs")

    jet_constit_pt = []
    jet_constit_eta = []
    jet_constit_phi = []

    for i in range(len(jet_constits)):
        for j in range(0, len(jet_constits[i]), 3):
            jet_constit_pt.append(jet_constits[i][j])
            # jet_constit_eta.append(jet_constits[i][j+1])
            # jet_constit_phi.append(jet_constits[i][j+2])

    pt_scale = np.percentile(jet_constit_pt, [0.95])
    #pt_scale = max([max(jets[i,0]) for i in range(len(jets))])
    max_eta = 2.7
    max_phi = math.pi
    print("pt_scale" + str(pt_scale)+"#######################################################")

    for i in range(len(jet_constits)):
        jets[i][0] = jets[i][0] / pt_scale
        for j in range(0, len(jet_constits[i]), 3):
            jet_constits[i][j] = jet_constits[i][j] / pt_scale
            #jet_constits[i][j] = math.log(jet_constits[i][j]+1)
            #inputs[i][j+1] = (inputs[i][j+1]+max_eta)/(2*max_eta)
            #inputs[i][j+2] = (inputs[i][j+2]+max_phi)/(2*max_phi)
    return jets, jet_constits


def preprocess_inputs(
        path,
        sample_name,
        extension,
        n_constits,
        pt_prep_type,
        eta_phi_prep_type):
    """ Performs a certain type of pre-processing"""
    print("Loading data")
    x = load_data(path+sample_name+extension)
    
    print("Shuffling")
    np.random.seed(1)
    np.random.shuffle(x)
    print("Split up array")
    #n_constits = 10
    length = x.shape[0]
    #length = 2000
    # Split up stored information
    #x = x[:10000,:]
    #x = x[np.where(np.logical_and(600<x[:,get_column_no['jet pt']],x[:,get_column_no['jet pt']]<650))]
    everything =  x[:,0:get_column_no['constit start pt']] 
    labels =      x[:,get_column_no['label']]
    jet_masses =  x[:,get_column_no['jet mass']]
    jets =        x[:,get_column_no['jet pt']:(get_column_no['jet pt']+3)]
    jet_pt = x[:,get_column_no['jet pt']]
    jet_constits = x[:,get_column_no['constit start pt']:get_column_no['constit start pt'] + 3 * n_constits]
    jet_subjets = x[:,get_column_no['subjet start']:get_column_no['subjet start']+9]
    print("Deleting x")
    del x
    #jet_constits = np.array([x[i][get_column_no['constit start pt']:]
    #                         for i in range(length)])
    #jet_constits = np.array([x[i][5:] for i in range(length)])
    '''
    rows = np.where(np.logical_and(jets[:,0]>600,jets[:,0]<700))
    #rows = np.where(jet_masses[:]<100e3)
    labels = labels[rows]
    jet_masses = jet_masses[rows]
    jets = jets[rows]
    jet_constits = jet_constits[rows]
    everything = everything[rows]
    '''
    print("Length of jet constituents")
    print(len(jet_constits))
    # for i in range(len(jet_constits)):
    #     print(jets[i][0])
    # jet_constits = [jet_constits[i] for i in range(len(jets)) if(jets[i][0]>650 and jets[i][0]<800)]
    #test_inputs(labels, jet_masses, jets, jet_constits)
    #plot_distributions(labels, jet_masses, jets, jet_constits,"initial")

    jet_constit_pt = []
    jet_constit_eta = []
    jet_constit_phi = []
    jet_constit_scale = []
    jet_constit_log_pt = []
    jet_constit_log_mean = []
    jet_constit_log_scale = []
    jet_constits_pt_list = [[] for j in range(91)]
    jet_constits_log_pt_list = [[] for j in range(91)]

    theta_all = []
    sum_eta_all = []
    max_len = 0
    print("Getting pt scale")
    # get scale for pt
    '''
    for i in range(len(jet_constits)):
        count = 0
        for j in range(0, len(jet_constits[i]), 3):
            #jet_constits_sum[i] = jet_constits_sum[i]+jet_constits[i][j]
            #jet_constits_num[i] = jet_constits_num[i]+1
            if count > max_len:
                max_len = count
            jet_constits_pt_list[count].append(jet_constits[i][j])
            jet_constits_log_pt_list[count].append(
                math.log10(
                    jet_constits[i][j] +
                    1))  # assuming lognormally distributed
            jet_constit_log_pt.append(
                math.log10(
                    jet_constits[i][j] +
                    1))  # lognormal
            jet_constit_pt.append(jet_constits[i][j])
            jet_constit_eta.append(jet_constits[i][j + 1])
            jet_constit_phi.append(jet_constits[i][j + 2])
            count += 1
    # print(jet_constits_log_pt_list[0])
    for j in range(max_len + 1):
        # print(np.std(jet_constits_log_pt_list[j]))
        jet_constit_log_scale.append(np.std(jet_constits_log_pt_list[j]))
        # print(np.mean(jet_constits_log_pt_list[j]))
        jet_constit_log_mean.append(np.mean(jet_constits_log_pt_list[j]))
        min_max = abs(max(jet_constits_pt_list[j]))
        if (min_max <= 0.001):
            jet_constit_scale.append(1.0)  # do not scale
        else:
            jet_constit_scale.append(min_max)
        jet_constit_scale.extend(
            np.percentile(
                jet_constits_pt_list[j],
                [0.98]) * 300)
        # jet_constit_log_scale.extend(np.std(np.array(jet_constits_log_pt_list[j])))
        # jet_constit_log_mean.extend(np.mean(np.array(jet_constits_log_pt_list[j])))
        # print(j)
    '''
    #max_pt = max(jet_constit_pt)
    #min_pt = min(jet_constit_pt)
    MAX_PT = 1679.1593231
    MIN_PT = 0.0

    max_eta = 2.7
    max_phi = math.pi

    #log_jet_scale = np.std(jet_constit_log_pt)
    #log_jet_mean = np.mean(jet_constit_log_pt)

    def no_scale(jet_constit, jet):
        return jet_constit

    def pt_scale(jet_constit, jet):
        return jet_constit / pt_val

    def eta_scale(jet_constit, jet):
        return jet_constit / eta_val

    def phi_scale(jet_constit, jet):
        return jet_constit / phi_val

    def log_scale(jet_constit, jet):
        return math.log10(jet_constit + 1)

    def log_norm_scale(jet_constit, jet):
        return (math.log10(jet_constit + 1) - log_jet_mean) / log_jet_scale

    def pt_indiv_scale_log_norm(jet_constit, i):
        if jet_constit_log_scale[int(i / 3)] <= 0.001:
            return (math.log10(jet_constit + 1) -
                    jet_constit_log_mean[int(i / 3)])
        else:
            return (math.log10(jet_constit + 1) - \
                    jet_constit_log_mean[int(i / 3)]) / jet_constit_log_scale[int(i / 3)]

    def pt_min_max_scale(jet_constit, jet):
        #print("hardcoded")
        #print("min_pt" + str(min_pt)+"#######################################################")
        #print("max_pt" + str(max_pt)+"#######################################################")
        #min_pt = 0.0#######################################################
        #max_pt = 1679.1593231
        return (jet_constit - MIN_PT) / (MAX_PT - MIN_PT)

    def jet_pt_scale(jet_constit, jet):
        return jet_constit / jet

    def pt_indiv_scale(jet_constit, i):
        # print(int(i/3))
        return jet_constit / jet_constit_scale[int(i / 3)]

    def eta_min_max_scale(jet_constit, jet):
        return (jet_constit - min_eta) / (max_eta - min_eta)

    def phi_min_max_scale(jet_constit, jet):
        return (jet_constit - min_phi) / (max_phi - min_phi)

    def eta_shift(jet_constit, jet):
        return jet_constit - jet

    def phi_shift(jet_constit, jet):
        # must account for the wrap
        phi_trans = jet_constit - jet
        if phi_trans < -math.pi:
            return phi_trans + 2 * math.pi
        elif phi_trans >= math.pi:
            return phi_trans - 2 * math.pi
        else:
            return phi_trans

    def eta_shift_and_scale(jet_constit, jet):
        return ((jet_constit - jet) + 1) / 2

    def phi_shift_and_scale(jet_constit, jet):
        # must account for the wrap
        phi_trans = jet_constit - jet
        if phi_trans < -math.pi:
            return phi_trans + 2 * math.pi
        elif phi_trans >= math.pi:
            return phi_trans - 2 * math.pi
        else:
            return (phi_trans + 1) / 2

    def phi_eta_shift_and_rotate(pt, eta, phi, jet_eta, jet_phi, theta):
        # translate
        eta_p = jet_eta - eta
        phi_trans = jet_phi - phi
        if phi_trans < -math.pi:
            phi_p = phi_trans + 2 * math.pi
        elif phi_trans >= math.pi:
            phi_p = phi_trans - 2 * math.pi
        else:
            phi_p = phi_trans
        # rotate
        eta_pp = eta_p * math.cos(theta) + phi_p * math.sin(theta)
        phi_pp = -eta_p * math.sin(theta) + phi_p * math.cos(theta)
        return pt, eta_pp, phi_pp

    def phi_eta_shift_and_t_lorentz_rotate(
            pt, eta, phi, jet_eta, jet_phi, theta):
        # rotate
        constit_vec = ROOT.TLorentzVector()
        constit_vec.SetPtEtaPhiM(pt, eta_p, phi_p)
        constit_vec.RotateX(theta)
        eta_pp = constit_vec.Eta()
        phi_pp = constit_vec.Phi()
        pt_pp = constit_vec.Pt()
        return pt_pp, eta_pp, phi_pp

    def flip(eta):
        if(sum_eta < 0):
            phi_pp = -phi_pp

    # define scaling function
    if("no_scale" in pt_prep_type):
        print("Performing no pt scaling")
        function = no_scale
    elif(pt_prep_type == "log_scale"):
        print("Performing pt log scaling")
        function = log_scale
    elif("pt_log_norm_scale" in pt_prep_type):
        print("Performing pt log norm scaling")
        function = log_norm_scale
    elif("min_max_scale" in pt_prep_type):
        print("Performing pt min max scaling")
        function = pt_min_max_scale
    elif(pt_prep_type == "95_percentile_scale"):
        print("Performing pt 98 percentile scaling")
        function = pt_scale
    elif(pt_prep_type == "jet_pt_scale"):
        function = jet_pt_scale
    elif(pt_prep_type == "pt_indiv_scale_98_300"):
        function = pt_indiv_scale
    elif(pt_prep_type == "pt_indiv_scale_min_max"):
        function = pt_indiv_scale
    elif("pt_indiv_log_norm_scale" in pt_prep_type):
        function = pt_indiv_scale_log_norm
    else:
        print("ERROR: pt_prep_type " + pt_prep_type +
              "does not match any of the available options")

    if(eta_phi_prep_type == "no_scale"):
        print("Performing no eta phi scaling")
        eta_function = no_scale
        phi_function = no_scale
    elif(eta_phi_prep_type == "min_max_scale"):
        print("Performing eta phi min max scaling")
        eta_function = eta_min_max_scale
        phi_function = phi_min_max_scale
    elif(eta_phi_prep_type == "95_percentile_scale"):
        print("Performing eta phi 95 percentile scaling")
        eta_function = eta_scale
        phi_function = phi_scale
    elif("shift" in eta_phi_prep_type):
        print("Performing eta phi shifting")
        eta_function = eta_shift
        phi_function = phi_shift
    elif("shift_prim" in eta_phi_prep_type):
        print("Performing eta phi shifting about primary topocluster")
        eta_function = eta_shift
        phi_function = phi_shift
    elif(eta_phi_prep_type == "shift_and_scale"):
        eta_function = eta_shift_and_scale
        phi_function = phi_shift_and_scale
    elif(eta_phi_prep_type == "shift_and_rotate"):
        eta_function = eta_shift
        phi_function = phi_shift
        eta_phi_function = phi_eta_shift_and_rotate
    elif(eta_phi_prep_type == "shift_and_t_lorentz_rotate"):
        eta_phi_function = phi_eta_shift_and_t_lorentz_rotate
    else:
        print("ERROR: eta_phi_prep_type " + eta_phi_prep_type +
              "does not match any of the available options")

    #print("Length jet constit scale:")
    # print(len(jet_constit_scale))

    #print("Jet constit scale:")
    # print(jet_constit_scale)
    #[jet_constits_sum[i]/jet_constits_num[i] for i in range(len(jet_constits_num))]

    #pt_val = np.percentile(jet_constit_pt,[0.98])
    #eta_val =  np.percentile(jet_constit_eta,[0.98])
    #phi_val =  np.percentile(jet_constit_phi,[0.98])
    #to_save = np.hstack((everything,jet_constits))
    ## Save for use with fully connected network
    #file_name = pt_prep_type + "_pt_" + eta_phi_prep_type + "_eta_phi"+"nothing"
    #print("Saving scaled inputs as " + file_name)
    #save_in_data(file_name,to_save) 

    print("Translating")
    sys.stdout.flush()
    # Translate inputs
    for i in range(len(jet_constits)):
        #jets[i][0] = function(jets[i][0])
        eta_constit = None
        phi_constit = None
        
        if "shift" in eta_phi_prep_type:
            # Shift about centre of large R jet
            eta_constit = jets[i][1]
            phi_constit = jets[i][2]
        if "shift_prim" in eta_phi_prep_type:
            # Shift about primary jet constituent
            eta_constit = jet_constits[i][1]
            phi_constit = jet_constits[i][2] 
        if "subjet_shift" in eta_phi_prep_type:
            # Shift about primary subjet
            eta_constit = jet_subjets[i][1]
            phi_constit = jet_subjets[i][2]
        
        # Shift jet constituents
        for j in range(0,len(jet_constits[i]),3): 
            jet_constits[i][j] = function(jet_constits[i][j], j)
            
            if("shift" in eta_phi_prep_type):
                jet_constits[i][j+1] = eta_function(
                    jet_constits[i][j+1],
                    eta_constit)
                jet_constits[i][j+2] = phi_function(
                    jet_constits[i][j+2],
                    phi_constit)
            

        # Shift subjets
        for j in range(0,9,3): 
            if("shift" in eta_phi_prep_type):
                jet_subjets[i][j+1] = eta_function(
                    jet_subjets[i][j+1],
                    eta_constit)
                jet_subjets[i][j+2] = phi_function(
                    jet_subjets[i][j+2],
                    phi_constit)
    #plot_all_jets(jet_pt,jet_constits, labels, "translate")
    to_save = np.hstack((everything,jet_constits))
    ## Save for use with fully connected network
    #file_name = pt_prep_type + "_pt_" + eta_phi_prep_type + "_eta_phi"+"scale"
    #print("Saving scaled inputs as " + file_name)
    #save_in_data(file_name,to_save) 
    #file_name = pt_prep_type + "_pt_" + eta_phi_prep_type + "_eta_phi"+"translate"
    #print("Saving scaled inputs as " + file_name)
    #save_in_data(file_name,to_save)
    #return 

    
    if "boost" in eta_phi_prep_type:
        print("Performing Lorentz boosting")
        sys.stdout.flush()
        med_boost = np.median(jet_subjets[:,0]) # use median instead of mean due to lognormal distribution
        ave_boost = 1000 
        beta_values = []
        pt_values = [] 


        for i in range(len(jet_constits)):
            # Calculate beta
            beta = 0
            if "boost_UC" in eta_phi_prep_type:
                beta = -(jet_pt[i]/math.sqrt(jet_pt[i]**2+173**2))
            elif "boost_WF" in eta_phi_prep_type:
                beta =  (med_boost-jet_subjets[i,0])/(med_boost+jet_subjets[i,0])
            elif "boost_PT" in eta_phi_prep_type:
                beta =  (ave_boost-jet_pt[i])/(ave_boost+jet_pt[i])
            #phi_axis = jet_constits[i][5]
            #eta_axis = jet_constits[i][4]
            jet_subjets[i][0], jet_subjets[i][1],jet_subjets[i][2] = lorentz_boost(beta,0,0,jet_subjets[i][0], jet_subjets[i][1],jet_subjets[i][2])
            jet_subjets[i][3], jet_subjets[i][4],jet_subjets[i][5] = lorentz_boost(beta,0,0,jet_subjets[i][3], jet_subjets[i][4],jet_subjets[i][5])
            jet_subjets[i][6], jet_subjets[i][7],jet_subjets[i][8] = lorentz_boost(beta,0,0,jet_subjets[i][6], jet_subjets[i][7],jet_subjets[i][8])

            if "boost_PT" in eta_phi_prep_type:
                 # for i in range(len(jet_constits)):
                for j in range(0, len(jet_constits[i]), 3):
                    # Wojtek version
                    # print("boost factor:")
                    # print(beta)
                    jet_constits[i][j], jet_constits[i][j +1], jet_constits[i][j+2] =  lorentz_boost(
                                    beta,0,0,
                                    jet_constits[i][j], jet_constits[i][j+1], jet_constits[i][j+2]) 
                beta_values.append(beta)
                pt_values.append(jet_pt[i])  
            elif "boost_UC" in eta_phi_prep_type:
                for j in range(0, len(jet_constits[i]), 3):
                    # print("boost factor:")
                    # print(beta)
                    # UC Davis version
                    # note must be used with min max scaling 
                    jet_constits[i][j], jet_constits[i][j +1], jet_constits[i][j+2] =  lorentz_boost(
                                    beta,0,0,
                                    jet_constits[i][j]*MAX_PT, 
                                    jet_constits[i][j+1],
                                     jet_constits[i][j+2])
                    jet_constits[i][j] = jet_constits[i][j]/MAX_PT
                    beta_values.append(beta)
                    pt_values.append(jet_pt[i])
            elif "boost_WF" in eta_phi_prep_type:
                # for i in range(len(jet_constits)):
                for j in range(0, len(jet_constits[i]), 3):
                    # Wojtek version
                    # print("boost factor:")
                    # print(beta)
                    jet_constits[i][j], jet_constits[i][j +1], jet_constits[i][j+2] =  lorentz_boost(
                                    beta,0,0,
                                    jet_constits[i][j], jet_constits[i][j+1], jet_constits[i][j+2]) 
                beta_values.append(beta)
                pt_values.append(jet_pt[i])  

        #plot_all_jets(jet_pt, jet_constits, labels, "boost")

        # #Plot of boosts
        # plt.figure()
        # #print(pt_values)
        # #print(beta_values)
        # plt.hist2d(pt_values,beta_values,bins = 60,norm=LogNorm())
        # plt.xlabel(r"p$_T$")
        # plt.ylabel(r"$\beta$")
        # plt.colorbar()
        # plt.savefig("boost_vs_pt")
        # print("Length theta all")
        # print(len(theta_all))
        # print("Length jet constits all")
        # print(len(jet_constits))
        '''
        print("Re-translating after boosting")
        for i in range(len(jet_constits)):
            eta_constit = None
            phi_constit = None
            if "shift" in eta_phi_prep_type:
                # Shift about centre of large R jet
                eta_constit = jets[i][1]
                phi_constit = jets[i][2]
            if "shift_prim" in eta_phi_prep_type:
                # Shift about primary jet constituent
                eta_constit = jet_constits[i][1]
                phi_constit = jet_constits[i][2] 
            if "subjet_shift" in eta_phi_prep_type:
                # Shift about primary subjet
                eta_constit = jet_subjets[i][1]
                phi_constit = jet_subjets[i][2]
            for j in range(0,
                           len(jet_constits[i]),
                           3):  # as things are ordered by pt,eta,phi
                #jet_constits[i][j] = function(jet_constits[i][j],jets[i][0])
                jet_constits[i][j] = function(jet_constits[i][j], j)
                if("shift" in eta_phi_prep_type):
                    # print("before")
                    # print(jet_constits[i][j+1])
                    jet_constits[i][j+1] = eta_function(
                        jet_constits[i][j+1],
                        eta_constit)
                    jet_constits[i][j+2] = phi_function(
                        jet_constits[i][j+2],
                        phi_constit)
        '''
        #plot_all_jets(jet_pt, jet_constits, labels, "retrans")

    
    
    print("Calculating thetas for rotation")
    sys.stdout.flush()
    # calculate angle to rotate by
    ave_boost = np.median(jet_constits[:,0]) # use median instead of mean due to lognormal distribution
    beta_values = []
    pt_values = []
    for i in range(len(jet_constits)):
        # calculate theta by principal axis
        # rotate about first jet constit
        phi_axis = 0
        eta_axis = 0
        pt_axis = 0
        theta = 0
        y_axis = 0
        z_axis = 0
        if len(jet_constits[i]) >= 6:  # need two (pt,eta,phi) to rotate
            if "subjet" in eta_phi_prep_type:
                px, py, pz = get_px_py_pz(
                    jet_subjets[i][3], jet_subjets[i][4], jet_subjets[i][5])
                y_axis = py
                z_axis = pz
            elif "rotate_prim" in eta_phi_prep_type:
                # Calculate principal axis
                for j in range(0, len(jet_constits[i]), 3):
                    pt_i = jet_constits[i][j]
                    eta_i = jet_constits[i][j + 1]
                    phi_i = jet_constits[i][j + 2]
                    e_i = pt_i * math.cosh(eta_i)
                    rad_i = math.sqrt(eta_i**2 + phi_i**2)
                    if(rad_i != 0):
                        pt_axis += pt_i
                        phi_axis += (phi_i * e_i / rad_i)
                        eta_axis += (eta_i * e_i / rad_i)
                    count = count + 1
                    # print(count)
                px, py, pz = get_px_py_pz(pt_axis, eta_axis, phi_axis)
                y_axis = py
                z_axis = pz
            else:
                px, py, pz = get_px_py_pz(
                    jet_constits[i][3], jet_constits[i][4], jet_constits[i][5])
                y_axis = py
                z_axis = pz
            # Calculate angle for rotation
            theta = np.arctan2(y_axis, z_axis) + math.pi / 2
            #theta = math.atan2(phi_axis,eta_axis)
            theta_all.append(theta)
        else:
            theta = 0
            theta_all.append(theta)


    
    print("Rotating")
    sys.stdout.flush()
    if "rotate" in eta_phi_prep_type:
        print("actually rotating")
        for i in range(len(jet_constits)):
            for j in range(0, len(jet_constits[i]), 3):
                jet_constits[i][j], jet_constits[i][j +
                                                    1], jet_constits[i][j +
                                                                        2] = rotate_function(jet_constits[i][j], jet_constits[i][j +
                                                                                                                                 1], jet_constits[i][j +
                                                                                                                                                     2], theta_all[i])

    #plot_all_jets(jet_pt,jet_constits, labels, "rotation")
    #to_save = np.hstack((everything,jet_constits))
    ## Save for use with fully connected network
    #file_name = pt_prep_type + "_pt_" + eta_phi_prep_type + "_eta_phi"+"rotate"
    #print("Saving scaled inputs as " + file_name)
    #save_in_data(file_name,to_save) 

    if "flip_prim" in eta_phi_prep_type:
        print("Flipping about third highest pt constituent")
        sys.stdout.flush()
        for i in range(len(jet_constits)):
            # Move highest jet constituents to right hand plane
            do_eta_flip = False
            if jet_constits[i][7]<0.0:
                do_eta_flip = True
            for j in range(0,
                           len(jet_constits[i]),
                           3):  # as things are ordered by pt,eta,phi
                if do_eta_flip :
                    jet_constits[i][j + 1] = -jet_constits[i][j + 1]
    elif "flip_subj" in eta_phi_prep_type:
        print("Flipping about third highest pt subjet")
        sys.stdout.flush()
        for i in range(len(jet_constits)):
            # Move highest jet constituents to right hand plane
            do_eta_flip = False
            if jet_subjets[i][7]<0.0:
                do_eta_flip = True
            for j in range(0,
                           len(jet_constits[i]),
                           3):  # as things are ordered by pt,eta,phi
                if do_eta_flip :
                    jet_constits[i][j + 1] = -jet_constits[i][j + 1]
    elif "flip" in eta_phi_prep_type:
        print("Flipping about average pt position")
        sys.stdout.flush()
        # Move average jet pt to right hand plane
        for i in range(len(jet_constits)):
            sum_eta = 0
            sum_phi = 0
            for j in range(0, len(jet_constits[i]), 3):
                sum_eta += jet_constits[i][j] * \
                    jet_constits[i][j + 1]  # just to check
                sum_phi += jet_constits[i][j] * jet_constits[i][j + 2]
            for j in range(0,
                           len(jet_constits[i]),
                           3):  # as things are ordered by pt,eta,phi
                # if sum_phi < 0:
                #    jet_constits[i][j+2] = -jet_constits[i][j+2]
                if sum_eta < 0:
                    jet_constits[i][j + 1] = -jet_constits[i][j + 1]
    #plot_all_jets(jet_pt,jet_constits, labels, "flip")

    to_save = np.hstack((everything,jet_constits))
    # Save for use with fully connected network
    #file_name = pt_prep_type + "_pt_" + eta_phi_prep_type + "_eta_phi"+"small"
    file_name = pt_prep_type + "_pt_" + eta_phi_prep_type + "_eta_phi"+"flip"
    print("Saving scaled inputs as " + file_name)
    save_in_data(file_name,to_save) 
     

    # make final plots
    # plot_distributions(
    #     labels,
    #     jet_masses,
    #     jets,
    #     jet_constits,
    #     file_name +
    #     "_distros")

    return


def translate_inputs(jets, jet_constits):
    for i in range(len(jet_constits)):
        for j in range(0, len(jet_constits[i]), 3):
            jet_constits[i][j + 1] = jet_constits[i][j + 1] - jets[i][1]
            jet_constits[i][j + 2] = jet_constits[i][j + 2] - jets[i][2]


def reshape_for_rnn(jet_constits):
    """ Reshape jet constituents for rnn """
    new_inputs = []
    for i in range(len(jet_constits)):
        constits = jet_constits[i].reshape(-1, 3)
        #jet_p4 = inputs[i][1:4]
        #many_p4 = np.ones((constits.shape[0],3))
        #many_p4 = many_p4*jet_p4
        # print(constits.shape)
        # print(many_p4.shape)
        # print(inputs.shape)
        new_inputs.append(constits)
        if(i % 100000 == 0):
            print(i)
    new_inputs = sequence.pad_sequences(
        new_inputs, maxlen=None, dtype='float64')
    # print(new_inputs.shape)
    return new_inputs

def save_in_data(name, everything):
    directory = os.environ.get('DATADIR', 'Nonesuch')+"/top_tagging/numpy_arrays/"+name
    new_directory = directory.replace("flat_distribution",'')
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    np.savez(directory + "_everything", data=everything)
    return

# def save_in_data(name, constits, labels):
#     directory = "/data/jpearkes/top_tagging/numpy_arrays/"
#     np.savez(directory + name + "_inputs", data=constits)
#     np.savez(directory + name + "_labels", data=labels)

if __name__ == '__main__':

    eta_phi_prep_type_set=False
    pt_prep_type_set=False
    if len(sys.argv)==3:
        name, n_constits, sample_name = sys.argv
    elif len(sys.argv)==4:
        name, n_constits, sample_name, eta_phi_prep_type = sys.argv
        eta_phi_prep_type_set=True
    elif len(sys.argv)==5:
        name, n_constits, sample_name, eta_phi_prep_type,pt_prep_type = sys.argv
        eta_phi_prep_type_set=True
        pt_prep_type_set=True
    else:
        raise ValueError("need between 3 and 5 arguments")
    
    n_constits = int(n_constits)
    # Load input arrays
    #x= load_data("/data/jpearkes/top_tagging/numpy_arrays/flat_no_mass_cut_short_10_pt_min_max_scale_pt_shift_prim_and_rotate_and_flip_eta_phi_everything.npz")
    # JP  
    path = os.environ.get('DATADIR', 'Nonesuch')+"/top_tagging/subsampled/"
    #sample_name = "signal_and_background_flat_no_mass_cut"
    #sample_name = "flat_almost_no_pileup_truth"
    #sample_name = "cms_LHC2016pileup_notrim_ptsorted/flat_distribution"
    #sample_name = "cms_LHC2016pileup_notrim_ptsorted/flat_distribution_600_700"
    
    #sample_name = "flat_lowpileup"
    extension = ".npz"
    if not pt_prep_type_set:
        print("setting pt_prep_type to hard-coded default")
        pt_prep_type = sample_name+"_short_"+str(n_constits)+"_pt_min_max_scale"
    else:
        print("constructing pt_prep_type so it follows the normal pattern")
        pt_prep_type = sample_name+"_short_"+str(n_constits)+"_"+pt_prep_type
    if not eta_phi_prep_type_set:
        print("setting eta_phi_prep_type to hard-coded default")
        eta_phi_prep_type = "subjet_shift_and_rotate_and_flip"
    #eta_phi_prep_type = "subjet_shift_and_rotate_and_flip"
    #eta_phi_prep_type = "shift_prim_and_rotate_and_flip"
    #matplotlib.rc('text', usetex=True)

    print("pt_prep_type: "+pt_prep_type)
    print("eta_phi_prep_type: "+eta_phi_prep_type)
    
    preprocess_inputs(
        path,
        sample_name,
        extension,
        n_constits,
        pt_prep_type,
        eta_phi_prep_type)

    # path = "/tmp/wfedorko/top_tagging/outputs/"
    # sample_name = "Zprime_ttbar_allhad_batch6_ptpoint29_zpmass6040_npv"
    # extension = ".npz"


    #preprocess_inputs(labels, jet_masses, jets, jet_constits,"short_10_min_max_scale", "shift")
    #preprocess_inputs(labels, jet_masses, jets, jet_constits,"jet_pt_scale", "shift")
    #preprocess_inputs(labels, jet_masses, jets, jet_constits,"no_scale", "shift")

    '''
    preprocess_inputs(labels, jet_masses, jets, jet_constits,"no_scale", "min_max_scale")
    preprocess_inputs(labels, jet_masses, jets, jet_constits,"pt_scale", "min_max_scale")
    preprocess_inputs(labels, jet_masses, jets, jet_constits,"min_max_scale", "min_max_scale")
    preprocess_inputs(labels, jet_masses, jets, jet_constits,"log_scale", "min_max_scale")
    preprocess_inputs(labels, jet_masses, jets, jet_constits,"95_percentile_scale", "min_max_scale")

    preprocess_inputs(labels, jet_masses, jets, jet_constits,"no_scale", "no_scale")
    preprocess_inputs(labels, jet_masses, jets, jet_constits,"pt_scale", "no_scale")
    preprocess_inputs(labels, jet_masses, jets, jet_constits,"min_max_scale", "no_scale")
    preprocess_inputs(labels, jet_masses, jets, jet_constits,"log_scale", "no_scale")
    preprocess_inputs(labels, jet_masses, jets, jet_constits,"95_percentile_scale", "no_scale")
    '''
    '''
    # Scale inputs----------------------------------------------------
    print("Scaling inputs")
    #pdb.set_trace()
    jets_scaled, jet_constits_scaled = preprocess_inputs(jets, jet_constits,"no_scale")

    # Zero pad variable length jet constituents
    jet_constits_scaled_and_padded = sequence.pad_sequences(jet_constits_scaled, maxlen=None, padding = 'post',dtype='float64')

    # Save for use with fully connected network
    print("Saving scaled inputs")
    save_in_data("scaled",jet_constits_scaled_and_padded,labels)

    plot_distributions(labels, jet_masses, jets_scaled, jet_constits_scaled,"scaled")

    jet_constits_scaled_rnn = reshape_for_rnn(jet_constits_scaled)

    print("Saving scaled inputs for rnn")
    save_in_data("scaled_rnn",jet_constits_scaled_rnn,labels)

    # Translate inputs------------------------------------------------
    print("Translating inputs")
    jets_translated, jet_constits_translated = translate_inputs(jets_scaled, jet_constits_scaled)

    # Zero pad variable length jet constituents
    jet_constits_translated_and_padded = sequence.pad_sequences(jet_constits_translated_and_padded, maxlen=None, padding = 'pre',dtype='float64')

    # Save for use with fully connected network
    print("Saving translated inputs")
    save_in_data("translated",jet_constits_translated_and_padded,labels)

    plot_distributions(labels, jet_masses, jets_translated, jet_constits_translated,"translated")

    jet_constits_translated_rnn = reshape_for_rnn(jet_constits_translated)

    print("Saving translated inputs for rnn")
    save_in_data("translated_rnn",jet_constits_translated_rnn,labels)
    '''

    '''

    inputs = np.array([x[i][1:] for i in range(len(x))])

    print("done loading")
    print("length of inputs")
    print(len(inputs))
    print("length of labels")
    print(len(labels))
    print("inputs")
    print(inputs[0])
    print("labels")
    print(labels)
    #pdb.set_trace()

    #raw_data[::2]= sig_raw_data
    #raw_data[1::2]= bg_raw_data
    # Scale
    jet_pt = [inputs[i][1] for i in range(len(inputs))]
    max_pt = max(jet_pt)
    min_pt = min(jet_pt)
    max_mass = max([max(jets[i]) for i in range(len(jets))])
    max_eta = 2.7
    max_phi = math.pi
    #print("max_pt"+str(max_pt))
    print("max_mass"+str(max_mass))

    for i in range(len(inputs)):
        inputs[i][0] = jets[i]/max_mass
        for j in range(1,len(inputs[i])-2,3):
            inputs[i][j] = (inputs[i][j]-min_pt)/max_pt
            inputs[i][j+1] = (inputs[i][j+1]+max_eta)/(2*max_eta)
            inputs[i][j+2] = (inputs[i][j+2]+max_phi)/(2*max_phi)
    print("scaled inputs")
    print(inputs[0])
    inputs = sequence.pad_sequences(inputs, maxlen=None, padding = 'pre',dtype='float64')
    np.savez("scaled_inputs", data = inputs)
    np.savez("scaled_labels", data = labels)
    #pdb.set_trace()

    new_inputs = []#np.zeros((len(inputs),88,6))
    for i in range(len(inputs)):
        constits = inputs[i][4:].reshape(-1,3)
        jet_p4 = inputs[i][1:4]
        many_p4 = np.ones((constits.shape[0],3))
        many_p4 = many_p4*jet_p4
        #print(constits.shape)
        #print(many_p4.shape)
        #print(inputs.shape)
        new_inputs.append(np.hstack((many_p4,constits)))
        if(i%1000 == 0):
            print(i)
    print("saving data")
    new_inputs = sequence.pad_sequences(new_inputs, maxlen=None, padding = 'pre',dtype='float64')
    np.savez("inputs_rnn_shape",data=new_inputs)
    np.savez("labels", data = labels)
    print("done saving")
    #print(new_inputs.shape)


    new_inputs = sequence.pad_sequences(inputs, maxlen=None, padding = 'post',dtype='float64')
    np.savez("pre_processed_inputs", data = new_inputs)
    np.savez("labels", data = labels)


    #minmax_scale(inputs, feature_range=(0, 1), axis=0, copy=True)
    #pdb.set_trace()
    inputs_ = np.load("pre_processed_inputs.npz")
    #inputs_ = np.load("inputs.npz")
    inputs = inputs_['data']
    #inputs = inputs[:,3:]
    #print(inputs[0])
    labels_ = np.load("labels.npz")
    labels = labels_['data']

    new_inputs = np.zeros((inputs.shape[0],88,6))
    for i in range(inputs.shape[0]):
        constits = inputs[i][3:].reshape(-1,3)
        jet_p4 = inputs[i][0:3]
        many_p4 = np.ones((constits.shape[0],3))
        many_p4 = many_p4*jet_p4
        #print(constits.shape)
        #print(many_p4.shape)
        #print(inputs.shape)
        new_inputs[i] = np.hstack((many_p4,constits))
        if(i%1000 == 0):
            print(i)
    print("saving data")
    np.savez("inputs_rnn_shape",data=new_inputs)
    print("done saving")
    print(new_inputs.shape)

    inputs_ = np.load("scaled_inputs.npz")
    inputs = inputs_['data']
    inputs = inputs[:,1:]
    #print(inputs[0])
    plot_histo(data=inputs[:][1],name="pt_topos",title="Topocluster pt distribution after scaling",xlabel="pt [GeV]",ylabel="Topoclusters",show=True)
    labels_ = np.load("scaled_labels.npz")
    labels = labels_['data']
    '''
