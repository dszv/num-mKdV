import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')


e_1, e_2 = 1.2, 0.9                         # epsilon_1, epsilon_2
alpha = 4.0
n = 2                                       # number of solitons
a = [1.0, 1.0]
b = [0.0, 0.0]
c = [0.0, 0.0]
k = [0.75, 0.71]
d = [90.0, 60.0]                            # delta_1, delta_2, ...

S = 250.0                                   # length
x_0 = -80.0
h = 0.1                                     # recommended: 0.01
J = int(S/h)                                # N + 1 = J

T = 30.0                                    # duration (recommended: 50)
t_0 = 45.0                                  # initial time (recommended: 35) 
tau = 0.01                                  # recommended: 0.001
M = int(T/tau)                              # K + 1 = M

x = np.linspace(x_0, x_0 + h*(J - 1), J)
t = np.linspace(t_0, t_0 + tau*(M - 1), M)
q = np.zeros((M , J) , dtype = float)       # q(t,x)
p = np.zeros((M , J) , dtype = float)       # p(t,x)
u = np.zeros((M , J) , dtype = float)       # u(t,x)

Al = np.zeros((J) , dtype = float)          # A[j , j - 1] j = 2,...,J
Ad = np.ones((J) , dtype = float)           # A[j , j]
Au = np.zeros((J - 1) , dtype = float)      # A[j , j + 1]
P = np.zeros((J) , dtype = float)
D = np.zeros((J) , dtype = float)           # AP = D
Ll = np.zeros((J) , dtype = float)          # L[j , j - 1] j = 2,...,J
Ud = np.zeros((J) , dtype = float)          # U[j , j] , Uu = Au
Y = np.zeros((J) , dtype = float)           # Y = UP


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
    print("Calculating...")
    
    # initial conditions m=0,1 for p,q,u
    for j in range(J):
        for i in range(n):
            for l in range(2):
                q[l, j] = q[l, j] + qq(x[j], t[l], a[i], b[i], c[i], k[i], d[i])
                p[l, j] = p[l, j] + pp(x[j], t[l], a[i], b[i], c[i], k[i], d[i])
                u[l, j] = u[l, j] + uu(x[j], t[l], a[i], b[i], c[i], k[i], d[i])
        
    # find q[2 , -]
    for j in range(J):
        q[2 , j] = 2 * tau * p[1 ,j] + q[0 , j]
    
    # find p[m+1 , -] m+1=2,3,...
    for m in range(1 , M - 1):
        # A = ...
        # LU decomposition method
        # determine L, U
        Ad[0] = 1/(2 * tau)
        Ad[1] = 1/(2 * tau)
        Ad[J - 2] = 1/(2 * tau)
        Ad[J - 1] = 1/(2 * tau)
        for j in range(2 , J - 2):
            Al[j] = -e_1/(2 * tau * h**2 )								                                                                  # a_i=-e_1/2h^2 t
            Ad[j] = 1/(2 * tau) - e_2 * (q[m , j + 1] - 2 * q[m , j] + q[m , j - 1])/(tau * h**2 ) + e_1/(tau * h**2 )      # b^m_i
            Au[j] = -e_1/(2 * tau * h**2 )								                                                                  # c_i=-e_1/2h^2 t
    
        Ud[0] = Ad[0]                                                                                                       # u_1=b_1
    
        for j in range(1 , J):
            Ll[j] = Al[j]/Ud[j - 1]                                                                                         # l_i=a_i/u_{i-1}
            Ud[j] = Ad[j] - Ll[j] * Au[j - 1]                                                                               # u_i=b_i-l_i*c_{i-1}
    
        # solve LY = D (Y = UP); first define vector D
        D[0] = p[m , 0]/(2 * tau)                                                                                           # d^m_0=p^{m-1}_0/2t
        D[1] = p[m , 1]/(2 * tau)                                                                                           # d^m_1=p^{m-1}_1/2t
        D[J - 2] = p[m , J - 2]/(2 * tau)                                                                                   # d^m_{N-1}=p^{m-1}_{N-1}/2t
        D[J - 1] = p[m , J - 1]/(2 * tau)                                                                                   # d^m_N=p^{m-1}_N/2t
    
        for j in range(2 , J - 2):
            D[j] = p[m - 1 , j]/(2 * tau) - (p[m , j + 1] - p[m , j - 1])/(2 * h) - e_1 * (p[m - 1 , j + 1] - 2 * p[m - 1 , j] + p[m - 1 , j - 1])/(2 * tau * h**2 ) - e_2 * (q[m , j + 1] - 2 * q[m , j] + q[m , j - 1])*p[m - 1 , j]/(tau * h**2 ) + ((p[m , j + 1] - p[m , j - 1])**2 )/(h**2 ) - (1 - e_1) * (p[m , j + 2] - 2 * p[m , j + 1] + 2 * p[m , j - 1] - p[m , j - 2])/(2 * h**3 )        #d^m_i
    
        Y[0] = D[0]										                                                                                      # y_1=d_1
    
        for j in range(1 , J):
            Y[j] = D[j] - Ll[j] * Y[j - 1] 							                                                                    # y_i=d_i-l_i*y_{i-1}
    
        # solve UP = Y
        P[J - 1] = Y[J - 1]/Ud[J - 1]   	        					                                                                # p_n=y_n/u_n
    
        for j in range(1 , J):
            P[J - 1 - j] = (Y[J - 1 - j] - Au[J - 1 - j] * P[J - j])/Ud[J - 1 - j]                                          # p_{n-k}=(y_{n-k}-c_{n-k}*p_{n-k+1})/u_{n-k}
        for j in range(J):
            p[m + 1 , j] = P[j]
    
        # find q[m+2,-] m+2=3,4,...
        if (m < M - 2):
            for j in range(J):
                q[m + 2 , j] = 2 * tau * p[m + 1 , j] + q[m , j]
    
        # find u[m+1,-] m+1=2,3,...
        for j in range(1 , J - 1):
            u[m + 1 , j] = -4 * (p[m + 1 , j + 1] - p[m + 1 , j - 1])/(alpha * h)

    print("Good job!")

    # saving n-soliton data
    np.savetxt('soliton.txt',u,delimiter=",")
    
    # plotting
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
