#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 22:08:35 2020

@author: arunansupatra
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_excel("/Users/arunansupatra/Desktop/Covid19/us-states_nyt_v2.xlsx", header=0, converters = {0: pd.to_datetime})
state_df = np.array(df1.iloc[:, 1])
arr = np.array(df1.iloc[:, 0])

us_states = "New York"

afg = np.zeros((1, 6))
for i in range(np.size(arr)):
    if df1.iloc[i,5] == us_states:
        afg = np.concatenate((afg, np.array((df1.iloc[i]), ndmin = 2)))

afg = afg[1:]
diff = np.zeros((1,1))

for i in range(np.size(afg[:, 0])-1):
    tmpp = np.zeros((1,1))
    tmpp = afg[i+1, 2] - afg[i, 2]
    diff = np.append(diff, tmpp)
    
log_diff = np.log(diff)
  
ny = afg[:, 2]
ny = np.array(ny, dtype = int)
log_ny = np.log(ny)

# plt.figure()
# plt.scatter(range(len(diff)), diff)
# plt.title("lin-lin ny diff")

# plt.figure()
# plt.scatter(range(len(diff)), log_diff)
# plt.title("lin-log ny diff")

no_precautions = log_diff[2:15]
social_distancing = log_diff[16:21]
stay_at_home = log_diff[22:46]
face_covering = log_diff[47:]

np_lin = diff
sd_lin = diff[16:]
sah_lin = diff[22:]
fc_lin = diff[47:]

# plt.figure()
# plt.scatter(range(len(diff)), np_lin, label = "No Precautions")
# plt.scatter(range(len(diff))[16:], sd_lin, label = "Social Distancing")
# plt.scatter(range(len(diff))[22:], sah_lin, label = "Stay At Home")
# plt.scatter(range(len(diff))[47:], fc_lin, label = "Face Covering")
# plt.title("lin-lin ny diff")
# plt.legend()

# plt.figure()
# plt.scatter(range(104), log_diff)
# plt.title("lin-log ny diff")



def func_sdot(t, s, it):
    g = 1/14
    r0 = 5
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

# plt.figure()
# for s0 in [100000, 500000, 1000000, 1500000, 2000000]:
#     #s0=3864900
#     i0=30 
    
#     S, I = RK4(t0, tend, s0, i0, h);
#     R = -(S+I) + (s0+i0)
    
    
#     I_daily_np = np.zeros((len(I),), dtype = int)
#     I_daily_np[0] = 0
#     for i in range(len(I)-1):
#         #I_cum[i+1] = I[i+1] + I_cum[i]
#         I_daily_np[i+1] = I[i+1] + R[i+1] - I[i] - R[i]
    
    
#     plt.plot(T, I_daily_np, label = "s0 = " + str(s0))
    
#     plt.legend()
    
#     # plt.plot(T, np.log(I_daily_np), label = "s0 = " + str(s0))  
#     # plt.title("Log Infected NY All Possibilities")
#     # plt.legend()
            
# plt.title("Daily infections in NY: for various S0")
# plt.scatter(range(len(diff)), np_lin, label = "No Precautions", c = 'lightskyblue')
# plt.scatter(range(len(diff))[16:], sd_lin, label = "Social Distancing", c = 'lightcoral')
# plt.scatter(range(len(diff))[22:], sah_lin, label = "Stay At Home", c = 'forestgreen')
# plt.scatter(range(len(diff))[47:], fc_lin, label = "Face Covering", c = 'firebrick')
# plt.legend()
# plt.xlabel("Days since 01 March 2020")
# plt.ylabel("Daily Infections")



#_____________________________________-
#no precautions fitting

s0=386490
i0=10 

S, I = RK4(t0, tend, s0, i0, h);
R = -(S+I) + (s0+i0)


I_daily_np = np.zeros((len(I),), dtype = int)
I_daily_np[0] = 0
for i in range(len(I)-1):
    #I_cum[i+1] = I[i+1] + I_cum[i]
    I_daily_np[i+1] = I[i+1] + R[i+1] - I[i] - R[i]
    
# plt.figure()
# plt.plot(I)
# plt.plot(R)
# plt.plot(S)
# plt.plot(s0)
# plt.plot(I_daily_np)
# plt.plot(I+R, c = 'black')


# plt.figure()
# plt.scatter(T, S)
# plt.title("Succeptible")

# plt.figure()
plt.figure()
# plt.plot(T[:16], I_daily_np[:16], label = "r0 = 5")
plt.title("Daily infections in NY: No precautions period")
plt.legend()

# plt.plot(T, np.log(I_daily_np), label = "s0 = " + str(s0))  
# plt.title("Log Infected NY All Possibilities")
# plt.legend()
        

plt.scatter(range(len(diff))[:16], np_lin[:16], label = "No Precautions", c = 'lightskyblue')
plt.scatter(range(len(diff))[16:], sd_lin, label = "Social Distancing", c = 'lightcoral')
plt.scatter(range(len(diff))[22:], sah_lin, label = "Stay At Home", c = 'forestgreen')
plt.scatter(range(len(diff))[47:], fc_lin, label = "Face Covering", c = 'firebrick')
plt.legend()
plt.xlabel("Days since 01 March 2020")
plt.ylabel("Daily Infections")

# plt.scatter(range(len(diff)), np.log(np_lin), label = "No Precautions")
# plt.scatter(range(len(diff))[16:], np.log(sd_lin), label = "Social Distancing")
# plt.scatter(range(len(diff))[22:], np.log(sah_lin), label = "Stay At Home")
# plt.scatter(range(len(diff))[47:], np.log(fc_lin), label = "Face Covering")

#______________________________________
#eu travel ban announcement fitting

s0=385948
i0=1012 

def func_sdot(t, s, it):
    g = 1/14
    r0 = 8
    b = g*r0
    return -(b*s*it/(s+it+g*it*t))

def func_idot(t, s, it):
    g = 1/14
    r0 = 8
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

S, I = RK4(t0, tend, s0, i0, h);
R = -(S+I) + (s0+i0)


I_daily_np = np.zeros((len(I),), dtype = int)
I_daily_np[0] = sd_lin[0]
for i in range(len(I)-1):
    #I_cum[i+1] = I[i+1] + I_cum[i]
    I_daily_np[i+1] = I[i+1] + R[i+1] - I[i] - R[i]
    

plt.figure()
plt.plot(T[16:], I_daily_np[:84], label = "r0 = 8")

def func_sdot(t, s, it):
    g = 1/14
    r0 = 9
    b = g*r0
    return -(b*s*it/(s+it+g*it*t))

def func_idot(t, s, it):
    g = 1/14
    r0 = 9
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

S, I = RK4(t0, tend, s0, i0, h);
R = -(S+I) + (s0+i0)


I_daily_np = np.zeros((len(I),), dtype = int)
I_daily_np[0] = sd_lin[0]
for i in range(len(I)-1):
    #I_cum[i+1] = I[i+1] + I_cum[i]
    I_daily_np[i+1] = I[i+1] + R[i+1] - I[i] - R[i]


plt.plot(T[16:], I_daily_np[:84], label = "r0 = 9")

plt.title("Daily infections in NY: Social distancing period")
plt.legend()

#(len(I_daily_np)-16)

# plt.plot(T, np.log(I_daily_np), label = "s0 = " + str(s0))  
# plt.title("Log Infected NY All Possibilities")
# plt.legend()
        

plt.scatter(range(len(diff)), np_lin, label = "No Precautions", c = 'lightskyblue')
plt.scatter(range(len(diff))[16:], sd_lin, label = "Social Distancing", c = 'lightcoral')
plt.scatter(range(len(diff))[22:], sah_lin, label = "Stay At Home", c = 'forestgreen')
plt.scatter(range(len(diff))[47:], fc_lin, label = "Face Covering", c = 'firebrick')
plt.legend()
plt.xlabel("Days since 01 March 2020")
plt.ylabel("Daily Infections")

#_____________________________________
#sd + sah fitting

s0=386490
i0=35000

def func_sdot(t, s, it):
    g = 1/14
    r0 = 2.5
    b = g*r0
    return -(b*s*it/(s+it+g*it*t))

def func_idot(t, s, it):
    g = 1/14
    r0 = 2.5
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

S, I = RK4(t0, tend, s0, i0, h);
R = -(S+I) + (s0+i0)


I_daily_np = np.zeros((len(I),), dtype = int)
I_daily_np[0] = 5711
for i in range(len(I)-1):
    #I_cum[i+1] = I[i+1] + I_cum[i]
    I_daily_np[i+1] = I[i+1] + R[i+1] - I[i] - R[i]
    
plt.figure()
plt.plot(T[22:], I_daily_np[:78], label = "r0 = 2.5")
plt.title("Daily infections in NY: Stay at home period")
plt.legend()

#(len(I_daily_np)-16)

# plt.plot(T, np.log(I_daily_np), label = "s0 = " + str(s0))  
# plt.title("Log Infected NY All Possibilities")
# plt.legend()
        

plt.scatter(range(len(diff)), np_lin, label = "No Precautions", c = 'lightskyblue')
plt.scatter(range(len(diff))[16:], sd_lin, label = "Social Distancing", c = 'lightcoral')
plt.scatter(range(len(diff))[22:], sah_lin, label = "Stay At Home", c = 'forestgreen')
plt.scatter(range(len(diff))[47:], fc_lin, label = "Face Covering", c = 'firebrick')
plt.legend()
plt.xlabel("Days since 01 March 2020")
plt.ylabel("Daily Infections")

#______________________________________________
#fc fitting

i0=163293
s0=386490-i0 #386490-i0
tEnd = 104

def func_sdot(t, s, it):
    g = 1/14
    r0 = 1
    b = g*r0
    return -(b*s*it/(s+it+g*it*t))

def func_idot(t, s, it):
    g = 1/14
    r0 = 1
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

S, I = RK4(t0, tend, s0, i0, h);
R = -(S+I) + (s0+i0)


I_daily_np = np.zeros((len(I),), dtype = int)
I_daily_np[0] = 7183
for i in range(len(I)-1):
    #I_cum[i+1] = I[i+1] + I_cum[i]
    I_daily_np[i+1] = I[i+1] + R[i+1] - I[i] - R[i]
    
plt.figure()
plt.plot(T[47:], I_daily_np[:53], label = "r0 = 1")
plt.title("Daily infections in NY: Face covering period")
plt.legend()


# plt.plot(T[47:103], np.log(I_daily_np[1:54]), label = "s0 = " + str(s0))  
# plt.title("Log Infected NY All Possibilities")
# plt.legend()
        

plt.scatter(range(len(diff)), np_lin, label = "No Precautions", c = 'lightskyblue')
plt.scatter(range(len(diff))[16:], sd_lin, label = "Social Distancing", c = 'lightcoral')
plt.scatter(range(len(diff))[22:], sah_lin, label = "Stay At Home", c = 'forestgreen')
plt.scatter(range(len(diff))[47:], fc_lin, label = "Face Covering", c = 'firebrick')
plt.legend()
plt.xlabel("Days since 01 March 2020")
plt.ylabel("Daily Infections")

# plt.scatter(range(len(diff)), np.log(np_lin), label = "No Precautions")
# plt.scatter(range(len(diff))[16:], np.log(sd_lin), label = "Social Distancing")
# plt.scatter(range(len(diff))[22:], np.log(sah_lin), label = "Stay At Home")
# plt.scatter(range(len(diff))[47:], np.log(fc_lin), label = "Face Covering")