# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:32:41 2018

@author: thmun
"""

import matplotlib.pyplot as plt;
import numpy as np;
import matplotlib.animation as animation;
from mpl_toolkits.mplot3d import Axes3D

def islambda(f):
    LAMBDA = lambda:0
    return isinstance(f, type(LAMBDA)) and f.__name__ == LAMBDA.__name__

# Definition of ODE solver by FDM
def fdm(A = 0, B = 0, C = 0, D = 0, y0 = 0, dydx = 1, h = 0.5, max_x = 5, drawplot = True):
    iterations = int(max_x / h);
    x = [];
    y = [];
    
    y.append(0);
    y.append(y[0] + dydx * h)
    
    x.append(0);
    x.append(h);
    
    for i in range(2, iterations):
        x.append(x[i - 1] + h)
        X = x[i]
        
        a = A(X) if islambda(A) else A
        b = B(X) if islambda(B) else B
        c = C(X) if islambda(C) else C
        d = D(X) if islambda(D) else D
            
        y.append((d - a * ((-2 * y[i - 1] + y[i - 2]) / h**2) + b * (y[i - 2] / (2 * h)) - c * y[i - 1]) * (2 * h**2 / (2 * a + h * b)))
    
    if(drawplot):
        plt.plot(x, y)
        plt.show()
    
    print("Number of iterations:", iterations)
    print("Last x value:", x[len(x) - 1])
    print("Last y value:", y[len(y) - 1])
    

# Test call of ODE solver with appropriate settings
#fdm(1, -5, 4, lambda x: x**2, 0, 1, 0.005, 5, True)

#fdm(1, 0, 3, 0, 0, 1, 0.005, 5, True)

def I(x):
    return 0;

def g(t):
    return -10*np.sin(t);

def h(t):
    return 20*np.sin(t);

def wave():
    c = 0.7;
    l = 10;
    dx = 0.05;
    dt = 0.05;
    C = c*dt/dx;
    
    max_y = 100;
    max_t = 50;
    
    t_res = int(max_t / dt);
    x_res = int(l / dx);
    
    frameinterval= 50;
    
    x_space = np.linspace(0, l, x_res);
    t_space = np.linspace(0, max_t, t_res);
    
    u = [[0 for t in range(len(x_space))] for x in range(len(t_space))]
    
    #Set initial conditions
    for i in range(0, len(x_space)):
        u[0][i] = I(x_space[i]);
    
    #Incorporate du/dt = 0
    for i in range(1, len(x_space) - 1):
        u[1][i] = u[0][i] - 1/2*C**2*(u[0][i+1] - 2*u[0][i] + u[0][i-1])
    
    # Enumeration trough definition-space to get both value and index
    for n, t in enumerate(t_space):
        # compute x for this time
        for i, x in enumerate(x_space):
            # boundary conditions
            u[n][0] = g(t);                # can be a function of t
            u[n][len(x_space)-1] = h(t);   # can be a function of t
            
            if n > 1 and n < len(t_space) - 1 and i < len(x_space) - 1:
                u[n+1][i] = C**2*(u[n][i+1] - 2*u[n][i] + u[n][i-1]) + 2*u[n][i] - u[n-1][i]
           
            
    
    def init():
        line.set_ydata(u[0])
        return line
        
    def animate(i):
        line.set_ydata(u[i])
        return line
    
    # Animated output
    fig, ax = plt.subplots()
    line, = ax.plot(u[0])
    plt.ylim(-max_y, max_y)
    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=frameinterval, blit=False, save_count=len(t_space))
    #ani.save("wave_smooth.avi")
    ani.save() # This throws an error because no filename, but the animation will not play without this
    
    #2D static output
#    for i in range(len(u)):
#        plt.plot(u[i])
#        
#    plt.show()
    
#    #3D static output
#    x = range(len(x_space));
#    y = range(len(t_space));
#    X, Y = np.meshgrid(x, y);
#    fig = plt.figure();
#    ax = fig.gca(projection='3d')
#    data = np.array(u);
#    ax.plot_surface(X, Y, data);


wave()







