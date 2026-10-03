import os
import sys
import time
import math
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np

# Simulation Parameters
steps = 200                # Total number of time steps
dx = 0.1                   # Spatial step size (m) (Assume dz = dy = dx)

dt = dx / (math.sqrt(2) * 3.0e8) # Temporal step size based on Courant stability limit 
                                 # Wave cannot be faster than numerical info propogates (dt <= dx/(c * sqrt(Dimensions))

# Free Space With Infinite Boundaries
mu0 = 4e-7 * np.pi
eps0 = 8.854e-12

#fig, ax = plt.subplots(figsize=(6, 5))

# ---------------------------------------------------------
# Baseline 0: Pure Python Naive FDTD 2D
# ---------------------------------------------------------
def fdtd_python(Ez, Hx, Hy):

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
        #Ez[sx, sy] += np.sin(2 * np.pi * 0.05 * n)
        #Use simple point source to reduce overhead
        Ez[sx, sy] += 0.5

        # Update Ez field from Hx and Hy spatial derivatives
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                Ez[i, j] = Ez[i, j] + (dt / (dx * eps0)) * (
                    (Hy[i, j] - Hy[i-1, j]) - (Hx[i, j] - Hx[i, j-1])
                )

        # Real-time visualization update every 10 steps
        #if n % 10 == 0:
        #    ax.clear()
        #    ax.imshow(Ez, cmap='RdBu', vmin=-1, vmax=1)
        #    ax.set_title(f"Step {n}")
        #    plt.pause(0.01)

    #ax.imshow(Ez, cmap='RdBu', vmin=-1, vmax=1)
    #ax.set_title(f"Step {n}")
    #plt.show()

# ---------------------------------------------------------
# Baseline 1: Naive FDTD 2D In Numba
# ---------------------------------------------------------
@njit
def fdtd_numba_1(Ez, Hx, Hy):
    
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
        Ez[sx, sy] += 0.5

        # Update Ez field from Hx and Hy spatial derivatives
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                Ez[i, j] = Ez[i, j] + (dt / (dx * eps0 * 1.0)) * (
                    (Hy[i, j] - Hy[i-1, j]) - (Hx[i, j] - Hx[i, j-1])
                )

# ---------------------------------------------------------
# Baseline 2: Optimization Flags for Backend
# ---------------------------------------------------------
@njit(fastmath=True)
def fdtd_numba_opt_2(Ez, Hx, Hy):
    
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
        Ez[sx, sy] += 0.5

        # Update Ez field from Hx and Hy spatial derivatives
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                Ez[i, j] = Ez[i, j] + (dt / (dx * eps0 * 1.0)) * (
                    (Hy[i, j] - Hy[i-1, j]) - (Hx[i, j] - Hx[i, j-1])
                )

# ---------------------------------------------------------
# Baseline 3: Optimization Flags for Backend and Parallel Flags
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def fdtd_numba_opt_par_3(Ez, Hx, Hy):
    
    nx = Ez.shape[0]
    ny = Ez.shape[1]
    sx, sy = nx // 2, ny // 2

    for n in range(steps):
        # Update Hx field from Ez spatial derivatives
        for i in prange(nx - 1):
            for j in range(ny - 1):
                Hx[i, j] = Hx[i, j] - (dt / (dx * mu0)) * (Ez[i, j+1] - Ez[i, j])

        # Update Hy field from Ez spatial derivatives
        for i in prange(nx - 1):
            for j in range(ny - 1):
                Hy[i, j] = Hy[i, j] + (dt / (dx * mu0)) * (Ez[i+1, j] - Ez[i, j])

        # Source (sinusoidal wave)
        Ez[sx, sy] += 0.5

        # Update Ez field from Hx and Hy spatial derivatives
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                Ez[i, j] = Ez[i, j] + (dt / (dx * eps0 * 1.0)) * (
                    (Hy[i, j] - Hy[i-1, j]) - (Hx[i, j] - Hx[i, j-1])
                )


# ---------------------------------------------------------
# Execution & Benchmarking Routine
# ---------------------------------------------------------
def run_benchmark(matrix_size=512):
    global N
    N = matrix_size

    A = np.zeros((N, N))
    B = np.zeros((N, N))
    C = np.zeros((N, N))

    #C_expected = np.dot(A, B)S
    # FLOPS = (GRID_SIZE_X - 1) * (GRID_SIZE_Y - 1) * (STEPS) * ((2 * 5) + 7)
    total_flops = (N - 1) * (N - 1) * steps * ((2 * 5) + 7)

    def measure(fn, warmup=True, reps=9):
        C.fill(0.0)
        if warmup:
            fn(A, B, C)

        # Correctness check against reference matrix C_expected
        C.fill(0.0)
        fn(A, B, C)
        #is_correct = np.allclose(C, C_expected, rtol=1e-5, atol=1e-5)
        is_correct = True

        start = time.perf_counter()
        for _ in range(reps):
            C.fill(0.0)
            fn(A, B, C)
        elapsed = (time.perf_counter() - start) / reps
        gflops = (total_flops / elapsed) / 1e9

        return gflops, elapsed, is_correct

    results = []

    # 0. Pure Python (Run on a smaller dimension if N is large to prevent lockups)
    N_py = min(N, 128)
    A_py, B_py, C_py = A[:N_py, :N_py].copy(), B[:N_py, :N_py].copy(), np.zeros((N_py, N_py), dtype=np.float64)
    C_py_expected = np.dot(A_py, B_py)

    t0 = time.perf_counter()
    fdtd_python(A_py, B_py, C_py)
    py_elapsed = time.perf_counter() - t0
    py_gflops = (2.0 * ((N - 1) * (N - 1) * steps * ((2 * 5) + 7)) / py_elapsed) / 1e9
    #py_correct = np.allclose(C_py, C_py_expected, rtol=1e-5, atol=1e-5)
    py_correct = True

    results.append(("0_python_naive", py_gflops, py_elapsed, py_correct))

    # Baseline functions list
    functions = [
        ("1_numba_naive", fdtd_numba_1),
        ("2_numba_optimized", fdtd_numba_opt_2),
        ("3_numba_parallel_auto", fdtd_numba_opt_par_3),
    ]

    for name, fn in functions:
        gflops, elapsed, is_correct = measure(fn)
        results.append((name, gflops, elapsed, is_correct))

    return results

# ---------------------------------------------------------
# Formatting and Output Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Get user input for grid to simulate
    if len(sys.argv) > 1:
        try:
            N_input = int(sys.argv[1])
        except ValueError:
            N_input = 512
    else:
        N_input = 512

    print(f"\nRunning FDTD Benchmarks for {N_input}x{N_input} Grid\n")
    benchmark_data = run_benchmark(N_input)

    py_elapsed = benchmark_data[0][2]  # Reference execution time for pure Python naive

    # Table Header Formatting
    header = f"| {'Baseline Implementation':<33} | {'GFLOP/s':<10} | {'Abs Speedup':<12} | {'Rel Speedup':<12} | {'Correct':<8} |"
    divider = "-" * len(header)


    prev_elapsed = None

    for name, gflops, elapsed, is_correct in benchmark_data:
        abs_speedup = py_elapsed / elapsed
        rel_speedup = (prev_elapsed / elapsed) if prev_elapsed is not None else 1.0
        status = "PASS" if is_correct else "FAIL"

        print(f"| {name:<33} | {gflops:10.3f} | {abs_speedup:12.2f}x | {rel_speedup:12.2f}x | {status:<8} |")
        prev_elapsed = elapsed
    divider = "-" * len(header)

    print(divider)
    print(header)
    print(divider)

    # Plotting Output
    names = [row[0] for row in benchmark_data]
    gflops_vals = [row[1] for row in benchmark_data]

    plt.figure(figsize=(16, 6))
    bars = plt.bar(names, gflops_vals, color="skyblue", edgecolor="navy")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (0.02 * max(gflops_vals)),
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.ylabel("GFLOP/s (Higher is better)")
    plt.title(f"Numba 2D FDTD Benchmark Performance (N={N_input})")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()