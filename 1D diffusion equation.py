"""
@ A program for solving 1D diffusion equation
@ author ZimoJupiter
@ w.zimo@outlook.com
@ date 25 Nov 2024
@ license MIT License
"""
import numpy as np
from numpy import pi, exp, sqrt
import matplotlib.pyplot as plt
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'

def FundamentalSolution():
    M = 1
    D = 0.1
    t = np.array([1/12*pi, 1/4*pi, 4/pi, pi, 2*pi])
    x = np.arange(-5, 5.01, 0.01)
    c = np.zeros((t.shape[0], x.shape[0]))

    for t_i in range(t.shape[0]):
        c[t_i, :] = M/sqrt(4*pi*D*t[t_i])*exp(-x**2/(4*D*t[t_i]))

    plt.figure()
    labels = [r'$1/12\pi$', r'$1/4\pi$', r'$4/\pi$', r'$\pi$', r'$2\pi$']
    colors = ['goldenrod', 'orange', 'yellowgreen', 'green', 'teal', 'r']
    for t_i in range(t.shape[0]):
        plt.plot(x, c[t_i, :], c=colors[t_i], label=labels[t_i])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$c(x, t, M, D)$')
    plt.legend()
    # plt.show()
    plt.savefig('Figures/FundamentalSolution.png')

def SourcesImageMethodSolution(n=1, pic=True):
    M = 1
    D = 0.1
    L = 2
    t = np.array([1/12*pi, 1/4*pi, 4/pi, pi, 2*pi, 80*pi])
    x = np.arange(-5, 5.01, 0.01)
    c = np.zeros((t.shape[0], x.shape[0]))

    for t_i in range(t.shape[0]):
        sources = 0
        for n_i in range(n):
            sources += exp(-(x+2*(n_i+1)*L)**2/(4*D*t[t_i])) + exp(-(x-2*(n_i+1)*L)**2/(4*D*t[t_i]))
        c[t_i, :] = M/sqrt(4*pi*D*t[t_i])*(exp(-x**2/(4*D*t[t_i])) + sources) \

    if pic == True:
        plt.figure()
        labels = [r'$1/12\pi$', r'$1/4\pi$', r'$4/\pi$', r'$\pi$', r'$2\pi$', r'$80\pi$']
        colors = ['goldenrod', 'orange', 'yellowgreen', 'green', 'teal', 'r']
        for t_i in range(t.shape[0]):
            plt.plot(x, c[t_i, :], c = colors[t_i], label=labels[t_i])
        plt.axvspan(-L, L, color='gray', alpha=0.2, linewidth=0)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$c(x, t, M, D)$')
        plt.legend()
        # plt.show()
        plt.savefig('Figures/SourcesImageMethodSolution.png')
        
    return x[int(x.shape[0]*3/10):int(x.shape[0]*7/10)], c[:,int(x.shape[0]*3/10):int(x.shape[0]*7/10)]

def SourcesImageMethodSolution_2():
    M = 1
    D = 0.1
    L = 2
    n = 5
    t = 80*pi
    x = np.arange(-5, 5.01, 0.01)
    c = np.zeros((n, x.shape[0]))

    for n_ii in range(n):
        sources = 0
        for n_i in range(n_ii+1):
            sources += exp(-(x+2*(n_i+1)*L)**2/(4*D*t)) + exp(-(x-2*(n_i+1)*L)**2/(4*D*t))
        c[n_ii, :] = M/sqrt(4*pi*D*t)*(exp(-x**2/(4*D*t)) + sources) \

    plt.figure()
    labels = [r'$n=2$', r'$n=4$', r'$n=6$', r'$n=8$', r'$n=10$']
    colors = ['goldenrod', 'orange', 'yellowgreen', 'green', 'teal', 'r']
    for n_i in range(n):
        plt.plot(x, c[n_i, :], c=colors[n_i], label=labels[n_i])
    plt.axvspan(-L, L, color='gray', alpha=0.2, linewidth=0)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$c(x, t, M, D)$')
    plt.legend(loc = 'lower right')
    # plt.show()
    plt.savefig('Figures/SourcesImageMethodSolution_2.png')

def SourcesImageMethodSolution_3():
    M = 1
    D = 0.1
    L = 2
    n = 10
    t = 80*pi
    x = np.arange(-5, 5.01, 0.01)
    c = np.zeros((n, x.shape[0]))

    for n_ii in range(n):
        sources = 0
        for n_i in range(n_ii+1):
            sources += exp(-(x+2*(n_i+1)*L)**2/(4*D*t)) + exp(-(x-2*(n_i+1)*L)**2/(4*D*t))
        c[n_ii, :] = M/sqrt(4*pi*D*t)*(exp(-x**2/(4*D*t)) + sources)
    
    m = np.zeros((n))
    for n_i in range(n):
        m[n_i] = sum(c[n_i, int(np.where(np.round(x,5) == -2)[0]):int(np.where(np.round(x,5) == 2)[0])])*(x[1]-x[0])

    plt.figure()
    plt.plot(np.arange(n), m, label='Mass conservation')
    plt.scatter(np.arange(n)[2], m[2], color='red', zorder=5)
    plt.annotate(r'$n = 4, \eta = 95.16\% $', 
             xy=(np.arange(n)[2], m[2]), 
             xytext=(np.arange(n)[2]+0.2, m[2]-0.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5))
    plt.scatter(np.arange(n)[3], m[3], color='red', zorder=5)
    plt.annotate(r'$n = 6, \eta = 98.89\% $', 
             xy=(np.arange(n)[3], m[3]), 
             xytext=(np.arange(n)[3]+0.3, m[3]-0.15),
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5))
    plt.scatter(np.arange(n)[4], m[4], color='red', zorder=5)
    plt.annotate(r'$n = 8, \eta = 99.81\% $', 
             xy=(np.arange(n)[4], m[4]), 
             xytext=(np.arange(n)[4]+0.4, m[4]-0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5))
    plt.xlabel(r'n/2')
    plt.ylabel(r'Mass conservation ratio')
    # plt.show()
    plt.savefig('Figures/SourcesImageMethodSolution_3.png')

def ImplicitScheme():
    dx = 0.01
    dt = 0.005
    L = 4
    D = 0.1
    alpha = D*dt/(dx**2)

    x = np.arange(-0.5*L, 0.5*L+dx, dx)
    t = np.arange(0, 2.5*pi, dt)

    c = np.zeros((x.shape[0], t.shape[0]))
    c[int(0.5*(x.shape[0]-1)),0] = 1/dx

    A = np.zeros((x.shape[0]-2, x.shape[0]-2))
    B = np.zeros((x.shape[0]-2))
    C = np.zeros((x.shape[0]-2))

    t_comp = np.array([1/12*pi, 1/4*pi, 4/pi, pi, 2*pi])
    t_comp_num = 0
    t_comp_save = np.zeros((t_comp.shape[0], x.shape[0]))

    for i in range(x.shape[0]-2):
        if i == 0:
            A[i,i] = 2 + alpha
            A[i,i+1] = -alpha
        elif i == x.shape[0]-3:
            A[i,i] = 2 + alpha
            A[i,i-1] = -alpha
        else:
            A[i,i] = 2*(1+alpha)
            A[i,i-1] = -alpha
            A[i,i+1] = -alpha

    for t_i in range(t.shape[0]-2):
        B[0] = (2-alpha)*c[1,t_i] + alpha*c[2,t_i]
        B[-1] = alpha*c[-3,t_i] + (2-alpha)*c[-2,t_i]
        B[1:-1] = alpha*c[1:-3,t_i] + 2*(1-alpha)*c[2:-2,t_i] + alpha*c[3:-1,t_i]
        C = np.linalg.inv(A).dot(B)
        c[1:-1,t_i+1] = C

        if t_comp_num <t_comp.shape[0]:
            if t[t_i] < t_comp[t_comp_num] and t[t_i+1] > t_comp[t_comp_num]:
                t_comp_save[t_comp_num,:] = c[:,t_i]
                t_comp_num += 1

    A_x, A_c = SourcesImageMethodSolution(n=20, pic=False)

    plt.figure()
    labels_n = [r'$1/12\pi$ (n)', r'$1/4\pi$ (n)', r'$4/\pi$ (n)', r'$\pi$ (n)', r'$2\pi$ (n)']
    labels_a = [r'$1/12\pi$ (a)', r'$1/4\pi$ (a)', r'$4/\pi$ (a)', r'$\pi$ (a)', r'$2\pi$ (a)']
    colors = ['goldenrod', 'orange', 'yellowgreen', 'green', 'teal', 'r']
    for t_i in range(t_comp.shape[0]):
        plt.plot(x[1:-1], t_comp_save[t_i, 1:-1], c=colors[t_i], label=labels_n[t_i])
    for t_i in range(t_comp.shape[0]):
        plt.scatter(A_x[::10], A_c[t_i, ::10], marker='x', c=colors[t_i], label=labels_a[t_i])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$c(x, t, M, D)$')
    plt.legend(ncol=1)
    # plt.show()
    plt.savefig('Figures/Implicit.png')

def ExplicitScheme():
    dx = 0.01
    dt = 0.0001
    L = 4
    D = 0.1
    alpha = D*dt/(dx**2)

    x = np.arange(-0.5*L, 0.5*L+dx, dx)
    t = np.arange(0, 2.5*pi, dt)

    c = np.zeros((x.shape[0], t.shape[0]))
    c[int(0.5*(x.shape[0]-1)),0] = 1/dx

    t_comp = np.array([1/12*pi, 1/4*pi, 4/pi, pi, 2*pi])
    t_comp_num = 0
    t_comp_save = np.zeros((t_comp.shape[0], x.shape[0]))

    for t_i in range(t.shape[0]-1):
        for x_i in range(1,x.shape[0]-1):
            if x_i == 1:
                c[x_i,t_i+1] = c[x_i,t_i] + alpha*(c[x_i+1,t_i] - c[x_i,t_i])
            elif x_i == x.shape[0]-2:
                c[x_i,t_i+1] = c[x_i,t_i] + alpha*(c[x_i-1,t_i] - c[x_i,t_i])
            else:
                c[x_i,t_i+1] = c[x_i,t_i] + alpha*(c[x_i+1,t_i] - 2*c[x_i,t_i] + c[x_i-1,t_i])

        if t_comp_num <t_comp.shape[0]:
            if t[t_i] < t_comp[t_comp_num] and t[t_i+1] > t_comp[t_comp_num]:
                t_comp_save[t_comp_num,:] = c[:,t_i]
                t_comp_num += 1

    A_x, A_c = SourcesImageMethodSolution(n=20, pic=False)

    plt.figure()
    labels_n = [r'$1/12\pi$ (n)', r'$1/4\pi$ (n)', r'$4/\pi$ (n)', r'$\pi$ (n)', r'$2\pi$ (n)']
    labels_a = [r'$1/12\pi$ (a)', r'$1/4\pi$ (a)', r'$4/\pi$ (a)', r'$\pi$ (a)', r'$2\pi$ (a)']
    colors = ['goldenrod', 'orange', 'yellowgreen', 'green', 'teal', 'r']
    for t_i in range(t_comp.shape[0]):
        plt.plot(x[1:-1], t_comp_save[t_i, 1:-1], c=colors[t_i], label=labels_n[t_i])
    for t_i in range(t_comp.shape[0]):
        plt.scatter(A_x[::10], A_c[t_i, ::10], marker='x', c=colors[t_i], label=labels_a[t_i])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$c(x, t, M, D)$')
    plt.legend(ncol=1)
    # plt.show()
    plt.savefig('Figures/Explicit.png')

if __name__ == '__main__':
    FundamentalSolution()
    SourcesImageMethodSolution(n=1, pic=True)
    SourcesImageMethodSolution_2()
    SourcesImageMethodSolution_3()
    ImplicitScheme()
    ExplicitScheme()
