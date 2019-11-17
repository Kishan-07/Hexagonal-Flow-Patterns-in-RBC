import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# Function that returns the time derivative of all the seven modes. Input requires an array of the modes values in the following order: [U011, U111, U122, T011, T111, T122, T002].
def f(x):
    global Ra, Pr
    
    dU011_dt = -np.sqrt(3.0/7)*x[1]*x[2] - np.sqrt(0.5)*Ra*Pr*x[3] - 2*Pr*x[0]
    dU111_dt = -0.5*np.sqrt(3.0/7)*x[2]*x[0] - np.sqrt(0.5)*Ra*Pr*x[4] - 2*Pr*x[1]
    dU122_dt = np.sqrt(3.0/7)*x[0]*x[1] - np.sqrt(3.0/7)*Ra*Pr*x[5] - 7*Pr*x[2]
    
    dT011_dt = -np.sqrt(0.5)*x[1]*x[5] - np.sqrt(2)*x[0]*x[6] - np.sqrt(0.5)*x[0] - 2*x[3]
    dT111_dt = -0.5*np.sqrt(0.5)*x[0]*x[5] - np.sqrt(2)*x[1]*x[6] - np.sqrt(0.5)*x[1] - 2*x[4]
    dT122_dt = 0.5*np.sqrt(0.5)*(x[0]*x[4] + x[1]*x[3]) - np.sqrt(3.0/7)*x[2] - 7*x[5]
    dT002_dt = 4*np.sqrt(2)*x[1]*x[4] + 2*np.sqrt(2)*x[0]*x[3] - 4*x[6]

    return np.array([dU011_dt, dU111_dt, dU122_dt, dT011_dt, dT111_dt, dT122_dt, dT002_dt])
    

# This function produces and saves a plot showing the temperature and velocity distribution at a given time. The input requires an array X of the modes values in the following order: [U011, U111, U122, T011, T111, T122, T002]. The second input 'i' is the index value of time in the array T at which the plot is being made.
def plot(X, i):
    global x, y, xv, yv, dt, no
    
    z = np.pi/3    # Plot the z = pi/3 plane
    
    # Defining T, u and v fields. u is the x-component of velocity and v is the y-component.
    T = 4*X[3]*np.cos(y)*np.sin(z) + 8*X[4]*np.cos(np.sqrt(3)*x/2)*np.cos(0.5*y)*np.sin(z) + 8*X[5]*np.cos(np.sqrt(3)*x/2)*np.cos(1.5*y)*np.sin(2*z) + 2*X[6]*np.sin(2*z)
    
    u = 2*np.sqrt(6)*X[1]*np.sin(np.sqrt(3)*xv/2)*np.cos(0.5*yv)*np.cos(z) + 8/np.sqrt(7)*X[2]*np.sin(np.sqrt(3)*xv/2)*np.cos(1.5*yv)*np.cos(2*z)
    
    v = 2*np.sqrt(2)*X[0]*np.sin(yv)*np.cos(z) +  2*np.sqrt(2)*X[1]*np.cos(np.sqrt(3)*xv/2)*np.sin(0.5*yv)*np.cos(z) + 8*np.sqrt(3.0/7)*X[2]*np.cos(np.sqrt(3)*xv/2)*np.sin(1.5*yv)*np.cos(2*z)
    
    # Plot the density plot for temperature field and quiver plot for velocity field
    plt.close()
    fig = plt.figure()
    ax = plt.gca()
    heatmap = ax.pcolormesh(x, y, T, cmap = cm.jet, vmin = -25, vmax = 25)
    cbar = plt.colorbar(heatmap, orientation='vertical')
    cbar.set_label(r'$\theta$', fontsize = 14, rotation = 0)
    cbar.ax.tick_params(labelsize=12)
    ax.quiver(xv, yv, u, v, scale = 3000)
    
    ax.set_title(r't = %.3f, $|\theta|_{max}$ = %.3f' %(i*dt, np.max(np.abs(T))), fontsize=14)
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$y$', fontsize=14)
    plt.xticks([0, 2*np.pi, 4*np.pi], fontsize=12)
    plt.yticks([0, 2*np.pi, 4*np.pi], fontsize=12)
    filename = '~/T_%03d.png' %(i/no)
    plt.savefig(filename)

    return


# Rayleigh and Prandtl no.
Ra = 40
Pr = 10

# Time range t and time-step dt
t = 20
dt = 0.0001
N = int(t/dt) + 1
no = (N-1)/250       # 'no' is the number of steps after which a plot is saved. A total of 251 frames will be saved to the user's computer.

ti = 10     # Time from which time series of U and T modes is to be plotted (in the end)
tf = 20     # Time till which time series of U and T modes is to be plotted (in the end)

X = np.zeros([7, N])  # Array to store the 7 modes for all the times from 0 to t.
T = np.linspace(0, t, N)  # Array for time
X[:, 0] = np.array([1, 0.1, 0, 1, 0.1, 0, 0])     # Initial values in the following order: [U011, U111, U122, T011, T111, T122, T002]

# Meshgrid for plotting the density plot for temperature.
x = np.linspace(0, 4*np.pi, 101)
y = np.linspace(0, 4*np.pi, 101)
x, y = np.meshgrid(x, y)

# Meshgrid for plotting the quiver plot for velocity.
xv = np.linspace(0, 4*np.pi, 11)
yv = np.linspace(0, 4*np.pi, 11)
xv, yv = np.meshgrid(xv, yv)

plot(X[:, 0], 0)

# Evaluate the modes for all times.
for i in range(1, N):
    k1 = f(X[:, i-1])
    k2 = f(X[:, i-1] + k1*dt/2)
    k3 = f(X[:, i-1] + k2*dt/2)
    k4 = f(X[:, i-1] + k3*dt)

    X[:, i] = X[:, i-1] + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    
    if i % no == 0:
        plot(X[:, i], i)

# Plot the time series for the U modes and save to user's computer. We start from t = 25 (to remove the transients)
plt.close()
fig = plt.figure()
ax = plt.gca()
ax.plot(T[ti*10000:tf*10000], X[0, ti*10000:tf*10000], 'r-', T[ti*10000:tf*10000], X[1, ti*10000:tf*10000], 'b-', T[ti*10000:tf*10000], X[2, ti*10000:tf*10000], 'g-')
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
ax.set_xlabel(r'$t$', fontsize = 14)
plt.legend([r'U$_{011}$', r'U$_{111}$', r'U$_{122}$'], loc='best')
ax.set_title(r'Ra = %d, Pr = %d' %(Ra, Pr), fontsize = 16)
ax.set_xlim([ti, tf])
plt.plot([ti, tf], [0, 0], 'k--')
filename = '~/U.png'
plt.savefig(filename)

# Plot the time series for the T modes and save to user's computer. We start from t = 25 (to remove the transients)
plt.close()
fig = plt.figure()
ax = plt.gca()
ax.plot(T[ti*10000:tf*10000], X[3, ti*10000:tf*10000], 'r-', T[ti*10000:tf*10000], X[4, ti*10000:tf*10000], 'b-', T[ti*10000:tf*10000], X[5, ti*10000:tf*10000], 'g-', T[ti*10000:tf*10000], X[6, ti*10000:tf*10000], 'k-')
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
ax.set_xlabel(r'$t$', fontsize = 14)
plt.legend([r'$\theta_{011}$', r'$\theta_{111}$', r'$\theta_{122}$', r'$\theta_{002}$'], loc='best')
ax.set_title(r'Ra = %d, Pr = %d' %(Ra, Pr), fontsize = 16)
ax.set_xlim([ti, tf])
plt.plot([ti, tf], [0, 0], 'k--')
filename = '~/T.png'
plt.savefig(filename)

plt.close()
