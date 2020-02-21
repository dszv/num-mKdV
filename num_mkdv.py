# -*- coding: utf-8 -*-
"""
Created on Mon Dec 2 2019

@author: Diego S and Harold Blas

Numerical analysis of the perturbed mkdv equation
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


e_1, e_2 = 1.2, 0.9                         #epsilon_1, epsilon_2
alpha = 4.0
a_1, a_2, a_3 = 1.0, 1.0, 1.0
b_1, b_2, b_3 = 0.0, 0.0, 0.0
c_1, c_2, c_3 = 0.0, 0.0, 0.0
k_1, k_2, k_3 = 0.75, 0.71, 0.67
d_1, d_2, d_3 = 90.0, 60.0, 52.0            #delta_1, delta_2, delta_3

S = 250.0                                   #length
x_0 = -80.0
h = 0.01
J = int(S/h)                                #N + 1 = J

T = 50.0                                    #duration 50
t_0 = 35.0
tau = 0.001
M = int(T/tau)                              #K + 1 = M

x = np.zeros((J) , dtype = float)
t = np.zeros((M) , dtype = float)
q = np.zeros((M , J) , dtype = float)       #q(t,x)
p = np.zeros((M , J) , dtype = float)       #p(t,x)
u = np.zeros((M , J) , dtype = float)       #u(t,x)

Al = np.zeros((J) , dtype = float)          #A[j , j - 1] j = 2,...,J
Ad = np.ones((J) , dtype = float)           #A[j , j]
Au = np.zeros((J - 1) , dtype = float)      #A[j , j + 1]
P = np.zeros((J) , dtype = float)
D = np.zeros((J) , dtype = float)           #AP = D
Ll = np.zeros((J) , dtype = float)          #L[j , j - 1] j = 2,...,J
Ud = np.zeros((J) , dtype = float)          #U[j , j] , Uu = Au
Y = np.zeros((J) , dtype = float)           #Y = UP


def omega(a,k):
    return (a**2 * k + (1 - e_1 ) * k**3 )/(a**2 - e_1 * k**2)

def q_0(a,k):
    return (3 * a**2 )/((a**2 + (1 - e_1 ) * k**2 )*(2 + e_2 ))

def dzeta(x,t,a,k,d):
    return k * x - omega(a,k) * t + d

def qq(x,t,a,b,c,k,d):
    return q_0(a,k) * ( np.log( np.cosh( dzeta(x,t,a,k,d) / ( 2 * a ) ) ) + b * dzeta(x,t,a,k,d) + c )

def pp(x,t,a,b,c,k,d):
     return - q_0(a,k) * ( ( omega(a,k) / ( 2 * a ) ) * np.tanh( dzeta(x,t,a,k,d) / ( 2 * a )  ) + b * omega(a,k) )

def uu(x,t,a,b,c,k,d):
    return q_0(a,k) * ( ( ( 2 * k * omega(a,k) ) / ( alpha * a**2 ) ) * np.cosh( dzeta(x,t,a,k,d) / ( 2 * a) )**(-2) )

def main():
    ("Calculating...")
    
    for j in range(J):
        x[j] = x_0 + j * h
    
    for m in range(M):
        t[m] = t_0 + m * tau
    
    #Initial conditions m=0,1 for p,q,u
    for j in range(J):
        q[0 , j] = qq(x[j],t[0],a_1,b_1,c_1,k_1,d_1) + qq(x[j],t[0],a_2,b_2,c_2,k_2,d_2) # + qq(x[j],t[0],a_3,b_3,c_3,k_3,d_3)
        q[1 , j] = qq(x[j],t[1],a_1,b_1,c_1,k_1,d_1) + qq(x[j],t[1],a_2,b_2,c_2,k_2,d_2) # + qq(x[j],t[1],a_3,b_3,c_3,k_3,d_3)
        p[0 , j] = pp(x[j],t[0],a_1,b_1,c_1,k_1,d_1) + pp(x[j],t[0],a_2,b_2,c_2,k_2,d_2) # + pp(x[j],t[0],a_3,b_3,c_3,k_3,d_3)
        p[1 , j] = pp(x[j],t[1],a_1,b_1,c_1,k_1,d_1) + pp(x[j],t[1],a_2,b_2,c_2,k_2,d_2) # + pp(x[j],t[1],a_3,b_3,c_3,k_3,d_3)
        u[0 , j] = uu(x[j],t[0],a_1,b_1,c_1,k_1,d_1) + uu(x[j],t[0],a_2,b_2,c_2,k_2,d_2) # + uu(x[j],t[0],a_3,b_3,c_3,k_3,d_3)
        u[1 , j] = uu(x[j],t[1],a_1,b_1,c_1,k_1,d_1) + uu(x[j],t[1],a_2,b_2,c_2,k_2,d_2) # + uu(x[j],t[1],a_3,b_3,c_3,k_3,d_3)
        
    #Find q[2 , -]
    for j in range(J):
        q[2 , j] = 2 * tau * p[1 ,j] + q[0 , j]
    
    #Find p[m+1 , -] m+1=2,3,...
    for m in range(1 , M - 1):
        #A = ...
        #LU Decomposition method
        #Determine L, U
        Ad[0] = 1/(2 * tau)
        Ad[1] = 1/(2 * tau)
        Ad[J - 2] = 1/(2 * tau)
        Ad[J - 1] = 1/(2 * tau)
        for j in range(2 , J - 2):
            Al[j] = -e_1/(2 * tau * h**2 )								                    #a_i=-e_1/2h^2 t
            Ad[j] = 1/(2 * tau) - e_2 * (q[m , j + 1] - 2 * q[m , j] + q[m , j - 1])/(tau * h**2 ) + e_1/(tau * h**2 )      #b^m_i
            Au[j] = -e_1/(2 * tau * h**2 )								                    #c_i=-e_1/2h^2 t
    
        Ud[0] = Ad[0]                                                                                                       #u_1=b_1
    
        for j in range(1 , J):
            Ll[j] = Al[j]/Ud[j - 1]                                                                                         #l_i=a_i/u_{i-1}
            Ud[j] = Ad[j] - Ll[j] * Au[j - 1]                                                                               #u_i=b_i-l_i*c_{i-1}
    
        #Solve LY = D (Y = UP); first define vector D
        D[0] = p[m , 0]/(2 * tau)                                                                   #d^m_0=p^{m-1}_0/2t
        D[1] = p[m , 1]/(2 * tau)                                                                   #d^m_1=p^{m-1}_1/2t
        D[J - 2] = p[m , J - 2]/(2 * tau)                                                           #d^m_{N-1}=p^{m-1}_{N-1}/2t
        D[J - 1] = p[m , J - 1]/(2 * tau)                                                           #d^m_N=p^{m-1}_N/2t
    
        for j in range(2 , J - 2):
            D[j] = p[m - 1 , j]/(2 * tau) - (p[m , j + 1] - p[m , j - 1])/(2 * h) - e_1 * (p[m - 1 , j + 1] - 2 * p[m - 1 , j] + p[m - 1 , j - 1])/(2 * tau * h**2 ) - e_2 * (q[m , j + 1] - 2 * q[m , j] + q[m , j - 1])*p[m - 1 , j]/(tau * h**2 ) + ((p[m , j + 1] - p[m , j - 1])**2 )/(h**2 ) - (1 - e_1) * (p[m , j + 2] - 2 * p[m , j + 1] + 2 * p[m , j - 1] - p[m , j - 2])/(2 * h**3 )        #d^m_i
    
        Y[0] = D[0]										    #y_1=d_1
    
        for j in range(1 , J):
            Y[j] = D[j] - Ll[j] * Y[j - 1] 							    #y_i=d_i-l_i*y_{i-1}
    
        #Solve UP = Y
        P[J - 1] = Y[J - 1]/Ud[J - 1]   	        					    #p_n=y_n/u_n
    
        for j in range(1 , J):
            P[J - 1 - j] = (Y[J - 1 - j] - Au[J - 1 - j] * P[J - j])/Ud[J - 1 - j]                  #p_{n-k}=(y_{n-k}-c_{n-k}*p_{n-k+1})/u_{n-k}
        for j in range(J):
            p[m + 1 , j] = P[j]
    
        #Find q[m+2,-] m+2=3,4,...
        if (m < M - 2):
            for j in range(J):
                q[m + 2 , j] = 2 * tau * p[m + 1 , j] + q[m , j]
    
        #Find u[m+1,-] m+1=2,3,...
        for j in range(1 , J - 1):
            u[m + 1 , j] = -4 * (p[m + 1 , j + 1] - p[m + 1 , j - 1])/(alpha * h)
    
        if (m % 50 == 0):
            print(m)
    
    #saving soliton data
    np.savetxt('soliton.txt',u,delimiter=",")
    
    #plotting
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x,u[2,:])
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    
    plt.subplot(1, 3, 2)
    plt.plot(x,u[int(M/2),:])
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    
    plt.subplot(1, 3, 3)
    plt.plot(x,u[M - 1,:])
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    
    plt.tight_layout()
    plt.savefig('collision.pdf')
    plt.show()

main()
