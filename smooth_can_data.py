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
    print(sol["status"])
    
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
    
    
def get_meas_from_csv(csv_file, maxrow = None):
    p = []
    v = []
    a = []
    t = []
    with open(csv_file, "r") as f:
        reader=csv.reader(f)
        next(reader) # skip the header
        line = 0
        for row in reader:
            if maxrow and line > maxrow:
                break
            t.append(float(row[0]))
            p.append(float(row[1]))
            v.append(float(row[4]))
            a.append(float(row[5]))
            line += 1
      
    y = p + v[:-1] + a[:-2]
    return t, y



if __name__ == '__main__':
    
    # set parameters
    maxrow = None # specify number of rows to read in csv file, if none, read all rows
    lam1 = 1 # speed regulation
    lam2 = 10 # acceleration regulation
    lam3 = 10000 # jerk regulation
    
    # read CSV files
    data_folder = os.path.join(os.getcwd(),"can_data")
    # csv_file = "2021-08-02-13-23-22_2T3W1RFV0MC103811_ego.csv"
    csv_file = "2021-08-02-14-21-13_2T3P1RFV2MW181087_ego.csv"
    t,y = get_meas_from_csv(os.path.join(data_folder, csv_file), maxrow = maxrow)
    N = int((len(y) + 3)/3)
    p = y[:N]
    v = y[N:2*N-1]
    a = y[2*N-1:]
    
    # solve
    phat, vhat, ahat, jhat = rectify_1d(y, lam1, lam2, lam3)
    
    # plot
    f, axs = plt.subplots(1,4,figsize=(20,5))
    axs[0].scatter(t, p, s=0.5, label = "raw")
    axs[0].plot(t, phat, c="r", label = "smoothed")
    axs[0].set_title("position")
    axs[0].set_xlabel("time (s)")
    axs[0].set_ylabel("m")
    axs[0].legend()
    
    axs[1].scatter(t[:-1], v, s=0.5, label = "raw")
    axs[1].plot(t[:-1], vhat, c="r", label = "smoothed")
    axs[1].set_title("speed")
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("m/s")
    axs[1].legend()
    
    axs[2].scatter(t[:-2], a, s=0.5, label = "raw")
    axs[2].plot(t[:-2], ahat, c="r", label = "smoothed")
    axs[2].set_title("acceleration")
    axs[2].set_xlabel("time (s)")
    axs[2].set_ylabel("m/s2")
    axs[2].legend()
    
    axs[3].plot(t[:-3], jhat, c="r", label = "smoothed")
    axs[3].set_title("jerk")
    axs[3].set_xlabel("time (s)")
    axs[3].set_ylabel("m/s3")
    
    
    
    
    
    
    
    
    
    