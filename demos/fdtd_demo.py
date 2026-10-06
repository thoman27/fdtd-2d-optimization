import os
import sys
import time
import math
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np

# Simulation Parameters
steps    = 200             # Total number of time steps
gridSize = 200                
dx = 0.1                   # Spatial step size (m) (Assume dz = dy = dx)

dt = dx / (math.sqrt(2) * 3.0e8) # Temporal step size based on Courant stability limit 
                                 # Wave cannot be faster than numerical info propogates (dt <= dx/(c * sqrt(Dimensions))

# Free Space With Infinite Boundaries
mu0 = 4e-7 * np.pi
eps0 = 8.854e-12

fig, ax = plt.subplots(figsize=(6, 5))

Ez = np.zeros((gridSize, gridSize))
Hx = np.zeros((gridSize, gridSize))
Hy = np.zeros((gridSize, gridSize))

nx = Ez.shape[0]
ny = Ez.shape[1]
sx, sy = nx // 2, ny // 2

for n in range(steps):
    # Update Hx field from Ez spatial derivatives
    for i in range(nx - 1):
        for j in range(ny - 1):
            Hx[i, j] = Hx[i, j] - (dt / (dx * mu0)) * (Ez[i, j+1] - Ez[i, j])

    # Update Hy field from Ez spatial derivatives
    for i in range(nx - 1):
        for j in range(ny - 1):
            Hy[i, j] = Hy[i, j] + (dt / (dx * mu0)) * (Ez[i+1, j] - Ez[i, j])

    # Source (sinusoidal wave)
    #Use simple line source
    Ez[sx, sy] += np.sin(2 * np.pi * 0.05 * n)

    # Update Ez field from Hx and Hy spatial derivatives
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            Ez[i, j] = Ez[i, j] + (dt / (dx * eps0)) * (
                (Hy[i, j] - Hy[i-1, j]) - (Hx[i, j] - Hx[i, j-1])
            )

    # Real-time visualization update every 10 steps
    if n % 10 == 0:
        ax.clear()
        ax.imshow(Ez, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f"Step {n}")
        plt.pause(0.01)

ax.imshow(Ez, cmap='RdBu', vmin=-1, vmax=1)
ax.set_title(f"Step {n}")
plt.show()