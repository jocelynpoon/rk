#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:57:10 2024

@author: jocelynpoon
"""

import numpy as np
import matplotlib.pyplot as plt

t = 10  
n = 41  
dx = 40/(n-1)
dt = 1
m = 5
atol = .001
rtol = .001

#u = np.zeros((int(t/dt)+1, n))
u = np.zeros((10000, n))
#u_hat = np.zeros((int(t/dt)+1, n))
u_hat = np.zeros((10000, n))
x = np.linspace(0, 40, n)
#print(x)

u[0] = 0.5 * (1 + np.tanh(250 * (x - 20)))

u[:, 0] = 0
u[:, -1] = 1

gamma = [[0,0,0], [0,1,0], [-0.497531095840104, 1.384996869124138, 0], [1.010070514199942, 3.878155713328178, 0], [-3.196559004608766,-2.324512951813145, 1.642598936063715], [1.717835630267259, -0.514633322274467, 0.188295940828347]]
beta = [0, 0.075152045700771, 0.211361016946069, 1.100713347634329, 0.728537814675568, 0.393172889823198]
delta = [1, 0.081252332929194, -1.083849060586449, -1.096110881845602, 2.859440022030827, -0.655568367959557, -0.194421504490852]
beta_controller = [0.70, -0.40, 0] #PI 
q = 4
q_hat = 3
k = q_hat + 1
value = []
epsilon = [1, 1]
dt_old = 1
dt_list = [1]
j = 0

def rhs(u):
    dudx = np.zeros_like(u)
    dudx[1:] = (u[1:] - u[:-1]) / dx
    return -0.5 * dudx

def step_size(s1, s2, atol, rtol):
    value = []
    for i in range(0,n):
        error = s1[i] - s2[i]
        denominator = (atol +rtol * max(abs(s1[i]), abs(s2[i])))
        value.append((error/denominator)**2)
    w = (np.sum(value)/n)**(1/2)
    epsilon[:0] = [1/w]
    #print('ep', epsilon)
    dt = epsilon[0]**(beta_controller[0]/k) * epsilon[1]**(beta_controller[1]/k) * epsilon[2]**(beta_controller[2]/k) * dt_old
    #print(dt)
    epsilon.pop()
    return dt
s=0
while np.sum(dt_list) < t:
    s3 = u[j].copy()
    s1 = s3.copy()
    s2 = np.zeros_like(s3)
    for i in range(1, m+1):
        s2 += delta[i-1]*s1
        s1 = gamma[i][0]*s1 + gamma[i][1]*s2 + gamma[i][2]*s3 + beta[i-1] * dt * rhs(s1)
        s+=1
    s2 = (s2 + delta[m]*s1 + delta[m+1]*s3) / sum(delta[0:m+2])
    #print(sum(delta[0:m+2]))
    u[j+1] = s1
    u_hat[j+1] = s2
    if t - np.sum(dt_list) < 0.01:
        dt = t - np.sum(dt_list)
    else:
        dt = step_size(s1, s2, atol, rtol)
    dt_list.append(dt)
    dt_old = dt
    #print('dt', dt_list)
    #print(np.sum(dt_list))
    j += 1   
print(j)
         
plt.figure(figsize=(10, 6))
plt.plot(x, u[j], label='u')
plt.plot(x, u_hat[j], label='u_hat')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.title('RK4(3)5[3S*] Solution')
plt.grid(True)
plt.show()
