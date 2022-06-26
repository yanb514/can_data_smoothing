#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 11:34:23 2022

@author: yanbing_wang

Given position, velocity and acceleration data from CAN bus, produce smooth and internally consistent dynamics
Data in 10Hz, from csv files
"""

from cvxopt import matrix, solvers, sparse, spmatrix
import matplotlib.pyplot as plt
import os
import csv
import numpy as np

dt = 0.1 # 10Hz sampling frequency

def _blocdiag(X, n):
    """
    makes diagonal blocs of X, for indices in [sub1,sub2]
    n indicates the total number of blocks (horizontally)
    """
    if not isinstance(X, spmatrix):
        X = sparse(X)
    a,b = X.size
    if n==b:
        return X
    else:
        mat = []
        for i in range(n-b+1):
            row = spmatrix([],[],[],(1,n))
            row[i:i+b]=matrix(X,(b,1))
            mat.append(row)
        return sparse(mat)
    
    
def _getQPMatrices(y, lam1, lam2, lam3):
    '''
    turn ridge regression (reg=l2) 
    1/M||y-Hx||_2^2 + \lam2/N ||Dx||_2^2
    to QP form
    min 1/2 z^T Q x + p^T z + r
    input:  y = [p,v,a]^T data vector 
        lam1: velocity regulation
        lam2: acceleration regulation
        lam3: jerk regulation
    return: Q, p, H, (G, h if l1)
    NOTE that this formulation assumes no missing data and uniform sampling
    internal consistency (forward Euler): v[k+1] = v[k] + dt*a[k], p[k+1] = p[k] + dt*v[k]
    '''
    
    # get data
    N = int((len(y) + 3)/3)
    
    # differentiation operator
    D1 = _blocdiag(matrix([-1,1],(1,2), tc="d"), N) * (1/dt)
    D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), N) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    
    # define some matrices
    I1 = spmatrix(1.0, range(N), range(N))
    I2 = spmatrix(1.0, range(N-1), range(N-1))
    I3 = spmatrix(1.0, range(N-2), range(N-2))
    
    H1 = sparse([[I1],[spmatrix([], [], [], (N,2*N-3))]])
    H2 = sparse([[spmatrix([], [], [], (N-1,N))], [I2], [spmatrix([], [], [], (N-1,N-2))]])
    H3 = sparse([[spmatrix([], [], [], (N-2,2*N-1))], [I3]])
    
    y = matrix(y, tc = 'd')
    P = -2 * (H1*y/N + lam1*D1.trans()*H2*y/(N-1) + lam2*D2.trans()*H3*y/(N-2))
    Q = 2 * (I1/N + lam1*D1.trans()*D1/(N-1) + lam2*D2.trans()*D2/(N-2) + lam3*D3.trans()*D3/(N-3))
     
    return Q, P


def rectify_1d(y, lam1, lam2, lam3):
    '''                        
    solve solve for ||H1y-x||_2^2 + \lam * ||H2y-Dx||_2^2 + \mu * ||H3y-DDx||_2^2
    '''  
    # get data and matrices
    Q, p = _getQPMatrices(y, lam1, lam2, lam3)
    
    sol=solvers.qp(P=Q, q=p)
    print("Solver status: ", sol["status"])
    
    # extract result
    N = int((len(y) + 3)/3)
    D1 = _blocdiag(matrix([-1,1],(1,2), tc="d"), N) * (1/dt)
    D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), N) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    
    xhat = sol["x"][:N]
    vhat = D1 * xhat
    ahat = D2 * xhat
    jhat = D3 * xhat
    
    return xhat, vhat, ahat, jhat
    
    
def get_meas_from_csv(csv_file, minrow = 0, maxrow = None):
    '''
    read for ego.csv
    '''
    p = []
    v = []
    a = []
    t = []
    with open(csv_file, "r") as f:
        reader=csv.reader(f)
        next(reader) # skip the header
        line = 0
        for row in reader:
            if line < minrow: # skip current line
                line += 1
                continue
            if maxrow and line > maxrow:
                break
            t.append(float(row[0]))
            p.append(float(row[1]))
            v.append(float(row[4]))
            a.append(float(row[5]))
            line += 1
      
    return p,v,a,t


def get_meas_pair(csv_ego, csv_leader, chunk_num):
    '''
    read for ego and leader.csv given a chunk number
    read leader first. Record the start and end timestamps for a certain chunk number
    and then select ego data of the same time range
    '''
    pl = []
    vl = []
    al = []
    tl = []
    start = 10e11
    end = -1e6
    with open(csv_leader, "r") as f:
        reader=csv.reader(f)
        next(reader) # skip the header
        line = 0
        for row in reader:
            if float(row[9]) == chunk_num:
                start = min(float(row[0]), start) 
                end = max(float(row[0]), end)
                tl.append(float(row[0]))
                pl.append(float(row[1]))
                vl.append(float(row[4]))
                try:
                    al.append(float(row[5]))
                except:
                    al.append(np.nan)
            elif float(row[9]) > chunk_num:
                break # early termination
            line += 1
            
    p = []
    v = []
    a = []
    t = []
    with open(csv_ego, "r") as f:
        reader=csv.reader(f)
        next(reader) # skip the header
        line = 0
        for row in reader:
            if float(row[0]) < start:
                line += 1
                continue
            if float(row[0]) > end:
                break
            t.append(float(row[0]))
            p.append(float(row[1]))
            v.append(float(row[4]))
            a.append(float(row[5]))
            line += 1
     
    # yl = pl + vl[:-1]  + al[:-2]
    return [t,p,v,a], [tl, pl,vl,al]


def plot_compare(l1, l2, lgd1, lgd2):
    '''
    plot two lists (l1 and l2)
    each list follows time, pos, velocity, accel, (jerk)
    with legends lgd1 and lgd2
    '''
    
    n = len(l2)-1
    f, axs = plt.subplots(1,n,figsize=(20,5))
    titles = ["position", "speed", "acceleration", "jerk"]
    units = ["m", "m/s", "m/s2", "m/s3"]

    for i in range(n):
        try:
            axs[i].scatter(l1[0][:len(l1[i+1])], l1[i+1], s=0.5, label = lgd1)
            axs[i].plot(l2[0][:len(l2[i+1])], l2[i+1], c="r", label = lgd2)
            axs[i].set_title(titles[i])
            axs[i].set_xlabel("time (s)")
            axs[i].set_ylabel(units[i])
            axs[i].legend()
        except:
            # pass
            axs[i].plot(l2[0][:len(l2[i+1])], l2[i+1], c="r", label = lgd2)
            axs[i].set_title(titles[i])
            axs[i].set_xlabel("time (s)")
            axs[i].set_ylabel(units[i])
            axs[i].legend()
    

if __name__ == '__main__':
    
    # set parameters
    minrow = 0
    maxrow = 2000 # specify number of rows to read in csv file, if none, read all rows
    lam1 = 0 # speed regularization
    lam2 = 0 # acceleration regularization
    lam3 = 1 # jerk regularization
    chunk_num = 29 # to select car-following range in lead.csv
    
    #%% read CSV files
    data_folder = os.path.join(os.getcwd(),"can_data/VanderTest_files")
    csv_ego = "2021-08-02-12-14-33_2T3P1RFV6LC082594_ego.csv"
    csv_leader = "2021-08-02-12-14-33_2T3P1RFV6LC082594_lead.csv"
    
    # Read ego only
    # p,v,a,t = get_meas_from_csv(os.path.join(data_folder, csv_leader), minrow=minrow, maxrow = maxrow)
    
    # Read ego-leader pair given a chunk number
    print("Read ego-leader data for chunk number {}".format(chunk_num))
    ego, lead = get_meas_pair(os.path.join(data_folder, csv_ego), 
                              os.path.join(data_folder, csv_leader), 
                              chunk_num = chunk_num)
    t, p,v,a = ego
    tl, pl,vl,al = lead
    
    # Plot ego-leader pair
    plot_compare(ego, lead, "ego", "leader")
    
    
    #%% solve (for ego and leader independently)
    # solve for ego
    print("Smooth ego vehicle data")
    y = p + v[:-1] + a[:-2]
    phat, vhat, ahat, jhat = rectify_1d(y, lam1, lam2, lam3)
    
    # solve for leader
    print("Smooth lead vehicle data")
    yl = pl + vl[:-1] + al[:-2]
    plhat, vlhat, alhat, jlhat = rectify_1d(yl, lam1, 0, lam3)
    
    #%% plot raw vs. smoothed
    plot_compare(ego, [t,phat,vhat,ahat,jhat], "raw_ego", "smoothed_ego")
    plot_compare(lead, [tl,plhat,vlhat,alhat,jlhat], "raw_lead", "smoothed_lead")
    
    
    #%% Compare position differentiation (GPS) with speed data (CAN)
    # dpdt = np.diff(p)/0.1
    # plt.figure()
    # plt.plot(t[:-1], dpdt, label="pos diff")
    # plt.plot(t, v, label= "speed meas")
    # plt.ylim([19,34])
    # plt.title("speed")
    # plt.xlabel("time (s)")
    # plt.ylabel("m/s")
    # plt.legend()
    
    
    
    
    
    
    
    
    
    