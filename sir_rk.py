#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Tue Jun 16 14:30:12 2020

# @author: arunansupatra


import numpy as np
import matplotlib.pyplot as plt

def func_sdot(t, s, it):
    g = 1/14	#COVID19 infection lasts for 14 days
    r0 = 5	#Specify Reproduction Number
    b = g*r0
    return -(b*s*it/(s+it+g*it*t))

def func_idot(t, s, it):
    g = 1/14
    r0 = 5
    b = g*r0
    return ((b*s/(s+it+g*it*t)) - g)*it

def RK4(t0, tEnd, x0, y0, h):
    N = int((tEnd-t0)/h);

    T = range(t0,tEnd,h);
    Y = np.zeros((N,1));
    Y[0] = y0;
    X = np.zeros((N,1));
    X[0] = x0;
    
    for i in range(len(T)-1):
        k1_x = func_sdot(T[i], X[i], Y[i]);
        k1_y = func_idot(T[i], X[i], Y[i]);
        
        k2_x = func_sdot(T[i]+0.5*h, X[i]+0.5*h*k1_x, Y[i]+0.5*h*k1_y);
        k2_y = func_idot(T[i]+0.5*h, X[i]+0.5*h*k1_x, Y[i]+0.5*h*k1_y);
        
        k3_x = func_sdot(T[i]+0.5*h, X[i]+0.5*h*k2_x, Y[i]+0.5*h*k2_y);
        k3_y = func_idot(T[i]+0.5*h, X[i]+0.5*h*k2_x, Y[i]+0.5*h*k2_y);
        
        k4_x = func_sdot(T[i]+h, X[i]+h*k3_x, Y[i]+h*k3_y);
        k4_y = func_idot(T[i]+h, X[i]+h*k3_x, Y[i]+h*k3_y);
        
        X[i+1] = X[i] + (1/6)*(k1_x+2*k2_x+2*k3_x+k4_x)*h;
        Y[i+1] = Y[i] + (1/6)*(k1_y+2*k2_y+2*k3_y+k4_y)*h;
    return X, Y

t0 = 0; 
tend = 100;

h = 1; 

T = range(t0,tend,h);

s0=100;
i0=900; 
r0 = 5;
g = (1/14);
b = g*r0;

S, I = RK4(t0, tend, s0, i0, h);
R = -(S+I) + (s0+i0)

# plt.figure()
# plt.scatter(T, S)
# plt.title("Succeptible")

plt.figure()
plt.scatter(T, I)
plt.title("Infected")

log_I = np.log(I)

# plt.figure()
# plt.scatter(T, log_I)
# plt.title("lin-log Infected") 

# plt.figure()
# plt.scatter(T, R)
# plt.title("Removed")

# I_daily_np = np.zeros((len(I),), dtype = int)
# I_daily_np[0] = 0
# for i in range(len(I)-1):
#     I_daily_np[i+1] = I[i+1] + R[i+1] - I[i] - R[i]

# plt.figure()
# plt.scatter(T, I_daily_np)
# plt.title("Daily Infected")
