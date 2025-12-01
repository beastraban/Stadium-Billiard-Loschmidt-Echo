"""
Stadium Billiard - Collision Table Method
==========================================

Direct translation of MATLAB 2D gas approach as written by Ira Wolfson (2025)
One big collision table. That's it.
Written with AI Claude assistance
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import curve_fit
import time


def initialize_positions(M, R, a, seed=None):
    """Random positions inside stadium, velocities with |v|=1."""
    if seed is not None:
        np.random.seed(seed)
    
    pos = np.zeros((M, 2))
    vel = np.zeros((M, 2))
    
    count = 0
    while count < M:
        x = np.random.uniform(-a - R, a + R)
        y = np.random.uniform(-R, R)
        
        # Check inside stadium
        inside = False
        if abs(x) <= a and abs(y) <= R:
            inside = True
        elif x < -a and (x + a)**2 + y**2 <= R**2:
            inside = True
        elif x > a and (x - a)**2 + y**2 <= R**2:
            inside = True
        
        if inside:
            pos[count, 0] = x
            pos[count, 1] = y
            theta = np.random.uniform(0, 2*np.pi)
            vel[count, 0] = np.cos(theta)
            vel[count, 1] = np.sin(theta)
            count += 1
    
    return pos, vel


def compute_wall_time(x, y, vx, vy, wall, R, a):
    """
    Compute time to hit a specific wall.
    
    Walls: 0=top, 1=bottom, 2=left_circle, 3=right_circle
    """
    if wall == 0:  # top: y = R
        if vy > 1e-14:
            t = (R - y) / vy
            x_hit = x + vx * t
            if t > 1e-12 and abs(x_hit) < a:
                return t
    
    elif wall == 1:  # bottom: y = -R
        if vy < -1e-14:
            t = (-R - y) / vy
            x_hit = x + vx * t
            if t > 1e-12 and abs(x_hit) < a:
                return t
    
    elif wall == 2:  # left circle: center (-a, 0)
        dx, dy = x + a, y
        A = vx**2 + vy**2
        B = 2 * (dx*vx + dy*vy)
        C = dx**2 + dy**2 - R**2
        disc = B**2 - 4*A*C
        if disc >= 0 and A > 1e-14:
            sqrt_disc = np.sqrt(disc)
            t1 = (-B - sqrt_disc) / (2*A)
            t2 = (-B + sqrt_disc) / (2*A)
            for t in [t1, t2]:
                if t > 1e-12:
                    x_hit = x + vx * t
                    if x_hit <= -a + 1e-10:
                        return t
    
    elif wall == 3:  # right circle: center (a, 0)
        dx, dy = x - a, y
        A = vx**2 + vy**2
        B = 2 * (dx*vx + dy*vy)
        C = dx**2 + dy**2 - R**2
        disc = B**2 - 4*A*C
        if disc >= 0 and A > 1e-14:
            sqrt_disc = np.sqrt(disc)
            t1 = (-B - sqrt_disc) / (2*A)
            t2 = (-B + sqrt_disc) / (2*A)
            for t in [t1, t2]:
                if t > 1e-12:
                    x_hit = x + vx * t
                    if x_hit >= a - 1e-10:
                        return t
    
    return np.nan


def reflect_velocity(x, y, vx, vy, wall, R, a):
    """Reflect velocity off wall."""
    if wall == 0 or wall == 1:  # top or bottom
        return vx, -vy
    elif wall == 2:  # left circle
        nx = (x + a) / R
        ny = y / R
        dot = vx*nx + vy*ny
        return vx - 2*dot*nx, vy - 2*dot*ny
    else:  # right circle
        nx = (x - a) / R
        ny = y / R
        dot = vx*nx + vy*ny
        return vx - 2*dot*nx, vy - 2*dot*ny


def evolve_to_time(pos, vel, t_start, t_end, R, a):
    """
    Evolve ensemble from t_start to t_end.
    Run collisions until next collision > t_end, then advance to exactly t_end.
    
    Returns: pos, vel at t_end
    """
    M = pos.shape[0]
    pos = pos.copy()
    vel = vel.copy()
    
    t = t_start
    
    # Wall collision table: [particle, wall, time]
    wallHitTable = np.zeros((M * 4, 3))
    
    # Initialize
    count = 0
    for n in range(M):
        for w in range(4):
            wallHitTable[count, 0] = n
            wallHitTable[count, 1] = w
            dt = compute_wall_time(pos[n, 0], pos[n, 1], vel[n, 0], vel[n, 1], w, R, a)
            if dt is not None and not np.isnan(dt) and dt > 0:
                wallHitTable[count, 2] = t + dt
            else:
                wallHitTable[count, 2] = np.nan
            count += 1
    
    # Main loop: process collisions until next one is beyond t_end
    while True:
        # Find next collision
        if np.all(np.isnan(wallHitTable[:, 2])):
            break
        
        idx = np.nanargmin(wallHitTable[:, 2])
        next_time = wallHitTable[idx, 2]
        
        if np.isnan(next_time) or next_time > t_end:
            break
        
        particle = int(wallHitTable[idx, 0])
        wall = int(wallHitTable[idx, 1])
        
        # Advance all positions to collision time
        dt = next_time - t
        pos = pos + vel * dt
        t = next_time
        
        # Reflect velocity
        vel[particle, 0], vel[particle, 1] = reflect_velocity(
            pos[particle, 0], pos[particle, 1],
            vel[particle, 0], vel[particle, 1],
            wall, R, a
        )
        
        # Clear this collision
        wallHitTable[idx, 2] = np.nan
        
        # Update collision times for this particle
        for w in range(4):
            row = particle * 4 + w
            dt_new = compute_wall_time(
                pos[particle, 0], pos[particle, 1],
                vel[particle, 0], vel[particle, 1],
                w, R, a
            )
            if dt_new is not None and not np.isnan(dt_new) and dt_new > 0:
                wallHitTable[row, 2] = t + dt_new
            else:
                wallHitTable[row, 2] = np.nan
    
    # Advance to exactly t_end
    dt = t_end - t
    pos = pos + vel * dt
    
    return pos, vel


def loschmidt_echo(pos0, vel0, T_values, R, a, epsilon):
    """
    Loschmidt echo experiment.
    
    For each T:
    1. Forward evolve from 0 to T with stadium R
    2. Flip velocities
    3. Backward evolve from T to 2T with stadium R*(1+epsilon)
    4. Measure distance from initial state at 2T
    """
    M = pos0.shape[0]
    n_times = len(T_values)
    distances = np.zeros((M, n_times))
    
    R_pert = R * (1 + epsilon)
    
    for j, T in enumerate(T_values):
        # Forward evolution: 0 → T
        pos_fwd, vel_fwd = evolve_to_time(pos0.copy(), vel0.copy(), 0, T, R, a)
        
        # Flip velocities
        vel_back = -vel_fwd
        
        # Backward evolution: T → 2T with perturbed stadium
        pos_back, vel_back = evolve_to_time(pos_fwd, vel_back, T, 2*T, R_pert, a)
        
        # Flip velocities again
        vel_back = -vel_back
        
        # Distance from initial state
        distances[:, j] = np.sqrt(
            (pos_back[:, 0] - pos0[:, 0])**2 + 
            (pos_back[:, 1] - pos0[:, 1])**2 +
            (vel_back[:, 0] - vel0[:, 0])**2 + 
            (vel_back[:, 1] - vel0[:, 1])**2
        )
    
    return distances


def compute_fidelity(sup_distances, delta):
    return np.mean(sup_distances < delta, axis=0)


def sigmoid(t, t_c, sigma):
    return 0.5 * erfc((t - t_c) / (np.sqrt(2) * sigma))

def exponential(t, gamma):
    return np.exp(-gamma * t)


def fit_models(t, fidelity):
    t_max = t[-1]
    idx = np.argmin(np.abs(fidelity - 0.5))
    t_half = t[idx] if 0 < idx < len(t) - 1 else t_max / 2
    
    try:
        popt, _ = curve_fit(sigmoid, t, fidelity, p0=[t_half, t_max/10],
                            bounds=([0, 0.1], [t_max*2, t_max]), maxfev=5000)
        t_c, sigma = popt
    except:
        t_c, sigma = t_half, t_max/10
    
    res_sig = np.sum((fidelity - sigmoid(t, t_c, sigma))**2)
    
    try:
        popt, _ = curve_fit(exponential, t, fidelity, 
                            p0=[np.log(2)/max(t_half, 0.1)],
                            bounds=([0.001], [20/t_max]), maxfev=5000)
        gamma = popt[0]
    except:
        gamma = np.log(2) / max(t_half, 0.1)
    
    res_exp = np.sum((fidelity - exponential(t, gamma))**2)
    ratio = res_exp / res_sig if res_sig > 1e-15 else 1.0
    
    return {'t_c': t_c, 'sigma': sigma, 'gamma': gamma, 'ratio': ratio}


def main():
    print("Stadium Billiard - Collision Table Method")
    print("=" * 60)
    
    R, a = 1.0, 1.0
    epsilon = 0.01
    
    # Dense 30x30 grid
    M_values = np.logspace(np.log10(50), np.log10(5000), 30).astype(int)
    M_values = list(np.unique(M_values))  # Remove duplicates from rounding
    delta_values = list(np.logspace(np.log10(0.02), np.log10(2.0), 30))
    
    # Time values
    T_max = 25.0
    T_values = np.linspace(1.0, T_max, 25)
    
    print(f"R={R}, a={a}, ε={epsilon}")
    print(f"M values: {M_values}")
    print(f"δ values: {delta_values}")
    print(f"T_max={T_max}")
    print()
    
    # Store fidelity: shape (n_M, n_delta, n_T)
    fidelity_grid = np.zeros((len(M_values), len(delta_values), len(T_values)))
    
    for i, M in enumerate(M_values):
        print(f"Running M={M}...", end=" ", flush=True)
        start = time.time()
        
        pos0, vel0 = initialize_positions(M, R, a, seed=42)
        distances = loschmidt_echo(pos0, vel0, T_values, R, a, epsilon)
        
        for j, delta in enumerate(delta_values):
            fidelity_grid[i, j, :] = np.mean(distances < delta, axis=0)
        
        print(f"Done in {time.time() - start:.1f}s")
    
    print()
    
    # Save data
    np.savez('stadium_grid.npz', 
             T_values=T_values, M_values=M_values, delta_values=delta_values,
             fidelity_grid=fidelity_grid, epsilon=epsilon)
    print("Saved: stadium_grid.npz")
    
    # 3D Plot: x=T, y=log(M), z=Fidelity for fixed delta
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    
    # Pick middle delta for 3D plot
    delta_idx = len(delta_values) // 2
    delta_plot = delta_values[delta_idx]
    
    T_mesh, M_mesh = np.meshgrid(T_values, M_values)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(T_mesh, np.log10(M_mesh), fidelity_grid[:, delta_idx, :],
                           cmap=cm.viridis, edgecolor='none', alpha=0.9)
    
    ax.set_xlabel('Time T', fontsize=12, labelpad=10)
    ax.set_ylabel('log₁₀(M)', fontsize=12, labelpad=10)
    ax.set_zlabel('Fidelity', fontsize=12, labelpad=10)
    ax.set_title(f'Loschmidt Echo (δ={delta_plot}, ε={epsilon})', fontsize=14)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Fidelity')
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig('stadium_3d_M.png', dpi=300, bbox_inches='tight')
    print("Saved: stadium_3d_M.png")
    
    # 3D Plot: x=T, y=log(delta), z=Fidelity for fixed M
    M_idx = len(M_values) // 2
    M_plot = M_values[M_idx]
    
    T_mesh2, delta_mesh = np.meshgrid(T_values, delta_values)
    
    fig2 = plt.figure(figsize=(12, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    surf2 = ax2.plot_surface(T_mesh2, np.log10(delta_mesh), fidelity_grid[M_idx, :, :],
                             cmap=cm.viridis, edgecolor='none', alpha=0.9)
    
    ax2.set_xlabel('Time T', fontsize=12, labelpad=10)
    ax2.set_ylabel('log₁₀(δ)', fontsize=12, labelpad=10)
    ax2.set_zlabel('Fidelity', fontsize=12, labelpad=10)
    ax2.set_title(f'Loschmidt Echo (M={M_plot}, ε={epsilon})', fontsize=14)
    
    fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='Fidelity')
    ax2.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig('stadium_3d_delta.png', dpi=300, bbox_inches='tight')
    print("Saved: stadium_3d_delta.png")
    
    # 2D grid: t_c(M, delta)
    t_c_grid = np.zeros((len(M_values), len(delta_values)))
    
    for i, M in enumerate(M_values):
        for j, delta in enumerate(delta_values):
            fidelity = fidelity_grid[i, j, :]
            fits = fit_models(T_values, fidelity)
            t_c_grid[i, j] = fits['t_c']
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    im = ax3.imshow(t_c_grid, aspect='auto', origin='lower', cmap=cm.viridis,
                    extent=[np.log10(delta_values[-1]), np.log10(delta_values[0]),
                            np.log10(M_values[0]), np.log10(M_values[-1])])
    
    ax3.set_xlabel('log₁₀(δ)', fontsize=12)
    ax3.set_ylabel('log₁₀(M)', fontsize=12)
    ax3.set_title('Critical Time t_c(M, δ)', fontsize=14)
    
    cbar = fig3.colorbar(im, ax=ax3)
    cbar.set_label('t_c', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('stadium_tc_grid.png', dpi=300, bbox_inches='tight')
    print("Saved: stadium_tc_grid.png")
    
    # Save to CSV files
    import pandas as pd
    
    # Save T values
    pd.DataFrame({'T': T_values}).to_csv('stadium_T_values.csv', index=False)
    
    # Save M values
    pd.DataFrame({'M': M_values}).to_csv('stadium_M_values.csv', index=False)
    
    # Save delta values
    pd.DataFrame({'delta': delta_values}).to_csv('stadium_delta_values.csv', index=False)
    
    # Save t_c grid
    pd.DataFrame(t_c_grid, 
                 index=[f'M={m}' for m in M_values],
                 columns=[f'd={d:.4f}' for d in delta_values]).to_csv('stadium_tc_grid.csv')
    
    # Save full fidelity data as single CSV (flattened)
    rows = []
    for i, M in enumerate(M_values):
        for j, delta in enumerate(delta_values):
            for k, T in enumerate(T_values):
                rows.append({'M': M, 'delta': delta, 'T': T, 
                            'fidelity': fidelity_grid[i, j, k]})
    pd.DataFrame(rows).to_csv('stadium_fidelity_full.csv', index=False)
    
    print("Saved: CSV files")


if __name__ == "__main__":
    main()
