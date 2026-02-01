"""
Orbital Transfer Solver - Core Functionality
Finds initial velocity to transfer between two positions in given time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, PillowWriter
import spiceypy as spy
import pymcel as pc
import warnings
import time
import os
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib to non-interactive backend to avoid Qt issues
rcParams['backend'] = 'Agg'

# ============================================================================
# UNIT CONVERSION FUNCTIONS
# ============================================================================

def get_canonical_units(time_user_years):
    """Convert time from years to canonical units."""
    mu = 1  # Gravitational parameter
    UL = 1 * pc.constantes.au   # astronomical unit
    UM = 1 * pc.constantes.M_sun # solar mass
    G = pc.constantes.G # gravitational constant
    
    UT = np.sqrt(UL**3 / (G * UM)) # time unit constant
    time_canonical = time_user_years * pc.constantes.año / UT # convert years to canonical units
    
    return mu, UT, time_canonical

def canonical_to_years(time_canonical):
    """Convert time from canonical units back to years."""
    UL = 1 * pc.constantes.au
    UM = 1 * pc.constantes.M_sun
    G = pc.constantes.G
    
    UT = np.sqrt(UL**3 / (G * UM))
    time_years = time_canonical * UT / pc.constantes.año
    
    return time_years

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def objective_function(r_target, r_start, velocity_guess, mu, time_prop, 
                       velocity_history=None, orbital_elements_history=None):
    """Objective function: computes position error after propagation."""
    initial_state = np.concatenate((r_start, velocity_guess))
    final_state = spy.prop2b(mu, initial_state, time_prop)
    
    position_error = np.linalg.norm(final_state[:3] - r_target)
    
    if velocity_history is not None:
        velocity_history.append(velocity_guess.copy())
    
    if orbital_elements_history is not None:
        orbital_elements = spy.oscelt(final_state, 0, mu)
        orbital_elements_history.append(orbital_elements)
    
    return position_error

def nelder_mead_optimizer(objective_func, initial_guess, r_target, r_start, mu, time_prop,
                          velocity_history=None, orbital_elements_history=None,
                          step=0.1, tolerance=1e-8, max_no_improve=20):
    """Nelder-Mead optimization algorithm."""
    '''source https://github.com/fchollet/nelder-mead/'''
    dim = len(initial_guess)
    
    best_score = objective_func(r_target, r_start, initial_guess, mu, time_prop,
                                velocity_history, orbital_elements_history)
    no_improve = 0
    simplex = [[initial_guess, best_score]]
    
    # Initialize simplex
    for i in range(dim):
        point = initial_guess.copy()
        point[i] += step
        score = objective_func(r_target, r_start, point, mu, time_prop,
                               velocity_history, orbital_elements_history)
        simplex.append([point, score])
    
    iter_count = 0
    while True:
        simplex.sort(key=lambda x: x[1])
        current_best = simplex[0][1]
        
        # Check for improvement
        if current_best < best_score - tolerance:
            no_improve = 0
            best_score = current_best
        else:
            no_improve += 1
        
        if no_improve >= max_no_improve:
            return simplex[0][0], simplex[0][1]
        
        # Compute centroid
        centroid = np.zeros(dim)
        for point, _ in simplex[:-1]:
            centroid += point / (len(simplex) - 1)
        
        # Reflection
        worst = simplex[-1][0]
        reflected = centroid + 1.0 * (centroid - worst)
        rscore = objective_func(r_target, r_start, reflected, mu, time_prop,
                                velocity_history, orbital_elements_history)
        
        if simplex[0][1] <= rscore < simplex[-2][1]:
            simplex[-1] = [reflected, rscore]
            continue
        
        # Expansion
        if rscore < simplex[0][1]:
            expanded = centroid + 2.0 * (centroid - worst)
            escore = objective_func(r_target, r_start, expanded, mu, time_prop,
                                    velocity_history, orbital_elements_history)
            if escore < rscore:
                simplex[-1] = [expanded, escore]
            else:
                simplex[-1] = [reflected, rscore]
            continue
        
        # Contraction
        contracted = centroid + 0.5 * (centroid - worst)
        cscore = objective_func(r_target, r_start, contracted, mu, time_prop,
                                velocity_history, orbital_elements_history)
        if cscore < simplex[-1][1]:
            simplex[-1] = [contracted, cscore]
            continue
        
        # Reduction
        best_point = simplex[0][0]
        new_simplex = []
        for point, score in simplex:
            reduced = best_point + 0.5 * (point - best_point)
            new_score = objective_func(r_target, r_start, reduced, mu, time_prop,
                                       velocity_history, orbital_elements_history)
            new_simplex.append([reduced, new_score])
        simplex = new_simplex
        
        iter_count += 1

# ============================================================================
# ANIMATION FUNCTION FOR GIF GENERATION (WITH LARGER SUN)
# ============================================================================

def create_orbit_animation(initial_state, mu, duration_canonical, duration_years, 
                           filename_prefix, num_frames=100, fps=20):
    """
    Create an animated GIF of the orbital transfer.
    
    Parameters:
    -----------
    initial_state : np.array
        Initial state vector [x, y, z, vx, vy, vz]
    mu : float
        Gravitational parameter
    duration_canonical : float
        Propagation duration in canonical units
    duration_years : float
        Propagation duration in years
    filename_prefix : str
        Prefix for output filename
    num_frames : int
        Number of frames in animation
    fps : int
        Frames per second for GIF
    
    Returns:
    --------
    str : Path to saved GIF file
    """
    print(f"Generating orbit animation...")
    
    # Generate trajectory data
    times_canonical = np.linspace(0, duration_canonical, num_frames)
    positions = np.zeros((num_frames, 3))
    
    for i, t in enumerate(times_canonical):
        state = spy.prop2b(mu, initial_state, t)
        positions[i, :] = state[:3]
    
    # Convert times to years for display
    times_years = canonical_to_years(times_canonical)
    
    # Create figure
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up static elements - SUN (LARGER AND PROMINENT)
    sun_size = 400  # Larger size for the Sun
    sun_color = 'gold'
    sun_edge = 'darkorange'
    
    # Plot the Sun prominently
    ax.scatter(0, 0, 0, color=sun_color, s=sun_size, label='Sun', 
               edgecolors=sun_edge, linewidth=2, zorder=10)
    
    # Add a glow effect around the Sun
    ax.scatter(0, 0, 0, color=sun_color, s=sun_size*1.5, alpha=0.3, 
               edgecolors='none', zorder=9)
    
    # Start and end points (static markers)
    start_size = 150
    end_size = 150
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               color='red', s=start_size, label='Start', 
               edgecolors='darkred', linewidth=1.5, zorder=8)
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               color='limegreen', s=end_size, label='End', 
               edgecolors='darkgreen', linewidth=1.5, zorder=8)
    
    # Set labels and title
    ax.set_xlabel('X [AU]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y [AU]', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z [AU]', fontsize=14, fontweight='bold')
    ax.set_title('Orbital Transfer Animation', fontsize=16, fontweight='bold', pad=25)
    ax.legend(loc='upper right', fontsize=12, markerscale=0.8)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_range = np.max([np.ptp(positions[:, 0]), 
                        np.ptp(positions[:, 1]), 
                        np.ptp(positions[:, 2])]) / 2.0
    
    # Add 10% padding to ensure Sun is always visible
    padding = max_range * 0.1
    max_range_with_padding = max_range + padding
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range_with_padding, mid_x + max_range_with_padding)
    ax.set_ylim(mid_y - max_range_with_padding, mid_y + max_range_with_padding)
    ax.set_zlim(mid_z - max_range_with_padding, mid_z + max_range_with_padding)
    
    # Initialize empty trajectory line
    line, = ax.plot([], [], [], 'cyan', linewidth=3, alpha=0.9, label='Trajectory')
    point, = ax.plot([], [], [], 'blue', marker='o', markersize=10, 
                     alpha=1.0, label='Spacecraft')
    
    # Time text (in years)
    time_text = ax.text2D(0.02, 0.96, '', transform=ax.transAxes,
                         fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Progress text
    progress_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes,
                            fontsize=11,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Mission info text
    mission_text = ax.text2D(0.02, 0.83, '', transform=ax.transAxes,
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    mission_text.set_text(f'Total Mission Time: {duration_years:.3f} years')
    
    # Sun label (positioned near the Sun)
    sun_label = ax.text(0, 0, 0, '  SUN', color='darkorange', fontsize=12, 
                       fontweight='bold', ha='left', va='center', zorder=11)
    
    # Start and end labels
    start_label = ax.text(positions[0, 0], positions[0, 1], positions[0, 2], 
                         '  START', color='darkred', fontsize=11, 
                         fontweight='bold', ha='left', va='center', zorder=7)
    end_label = ax.text(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                       '  END', color='darkgreen', fontsize=11, 
                       fontweight='bold', ha='left', va='center', zorder=7)
    
    def init():
        """Initialize animation."""
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        time_text.set_text('')
        progress_text.set_text('')
        return line, point, time_text, progress_text, mission_text, sun_label, start_label, end_label
    
    def animate(i):
        """Update animation frame."""
        # Update trajectory line
        line.set_data(positions[:i, 0], positions[:i, 1])
        line.set_3d_properties(positions[:i, 2])
        
        # Update current position point
        point.set_data([positions[i, 0]], [positions[i, 1]])
        point.set_3d_properties([positions[i, 2]])
        
        # Update time display (in years)
        current_time_years = times_years[i]
        time_text.set_text(f'Time: {current_time_years:.3f} years\n'
                          f'Progress: {100*i/(num_frames-1):.1f}%')
        
        # Update progress
        progress = i / (num_frames - 1)
        progress_text.set_text(f'Position: ({positions[i, 0]:.2f}, '
                              f'{positions[i, 1]:.2f}, {positions[i, 2]:.2f}) AU')
        
        return line, point, time_text, progress_text, mission_text, sun_label, start_label, end_label
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=num_frames,
                        init_func=init, blit=True,
                        interval=1000/fps, repeat=True)
    
    # Save as GIF
    gif_filename = f"{filename_prefix}_animation.gif"
    
    # Use PillowWriter for GIF creation
    writer = PillowWriter(fps=fps)
    anim.save(gif_filename, writer=writer, dpi=120)
    
    plt.close(fig)
    
    print(f"✓ Animation saved: {gif_filename}")
    return gif_filename

# ============================================================================
# PLOTTING FUNCTIONS WITH AUTO-SAVE (NOW IN YEARS, WITH LARGER SUN)
# ============================================================================

def plot_trajectory_3d(initial_state, mu, duration_canonical, duration_years, 
                       filename_prefix, num_points=300):
    """Plot 3D trajectory and save to file."""
    # Generate trajectory in canonical units
    times_canonical = np.linspace(0, duration_canonical, num_points)
    positions = np.zeros((num_points, 3))
    
    for i, t in enumerate(times_canonical):
        state = spy.prop2b(mu, initial_state, t)
        positions[i, :] = state[:3]
    
    # Convert times to years for plotting
    times_years = canonical_to_years(times_canonical)
    
    # Create plot
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the Sun prominently (LARGER)
    sun_size = 500  # Larger size
    sun_color = 'gold'
    sun_edge = 'darkorange'
    
    ax.scatter(0, 0, 0, color=sun_color, s=sun_size, label='Sun', 
               edgecolors=sun_edge, linewidth=2, zorder=10)
    
    # Add a glow effect around the Sun
    ax.scatter(0, 0, 0, color=sun_color, s=sun_size*1.5, alpha=0.3, 
               edgecolors='none', zorder=9)
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'cyan', linewidth=3, alpha=0.9, label='Transfer Orbit', zorder=5)
    
    # Start and end points
    start_size = 150
    end_size = 150
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               color='red', s=start_size, label='Start', 
               edgecolors='darkred', linewidth=1.5, zorder=8)
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               color='limegreen', s=end_size, label='End', 
               edgecolors='darkgreen', linewidth=1.5, zorder=8)
    
    # Labels
    ax.set_xlabel('X [AU]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y [AU]', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z [AU]', fontsize=14, fontweight='bold')
    ax.set_title('Orbital Transfer Trajectory', fontsize=16, fontweight='bold', pad=25)
    ax.legend(loc='upper right', fontsize=12, markerscale=0.8)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect with padding for Sun visibility
    max_range = np.max([np.ptp(positions[:, 0]), 
                        np.ptp(positions[:, 1]), 
                        np.ptp(positions[:, 2])]) / 2.0
    
    # Add padding to ensure Sun is always visible
    padding = max_range * 0.1
    max_range_with_padding = max_range + padding
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range_with_padding, mid_x + max_range_with_padding)
    ax.set_ylim(mid_y - max_range_with_padding, mid_y + max_range_with_padding)
    ax.set_zlim(mid_z - max_range_with_padding, mid_z + max_range_with_padding)
    
    # Add labels near key points
    ax.text(0, 0, 0, '  SUN', color='darkorange', fontsize=12, 
           fontweight='bold', ha='left', va='center', zorder=11)
    ax.text(positions[0, 0], positions[0, 1], positions[0, 2], 
           '  START', color='darkred', fontsize=11, 
           fontweight='bold', ha='left', va='center', zorder=7)
    ax.text(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
           '  END', color='darkgreen', fontsize=11, 
           fontweight='bold', ha='left', va='center', zorder=7)
    
    # Add info box with time in years
    total_time_years = duration_years
    info_text = (f"Mission Information:\n"
                 f"• Total Time: {total_time_years:.3f} years\n"
                 f"• Trajectory Points: {num_points}\n"
                 f"• Sun at origin (0,0,0)")
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
              fontsize=11, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    
    # Save file
    filename = f"{filename_prefix}_trajectory.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Trajectory plot saved: {filename}")
    return filename

def plot_time_progression(initial_state, mu, duration_canonical, duration_years, filename_prefix):
    """Plot position vs time during the transfer."""
    # Generate trajectory data
    num_points = 200
    times_canonical = np.linspace(0, duration_canonical, num_points)
    positions = np.zeros((num_points, 3))
    
    for i, t in enumerate(times_canonical):
        state = spy.prop2b(mu, initial_state, t)
        positions[i, :] = state[:3]
    
    # Convert times to years
    times_years = canonical_to_years(times_canonical)
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    components = ['X', 'Y', 'Z']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # More vibrant colors
    
    for i, (ax, comp, color) in enumerate(zip(axes, components, colors)):
        ax.plot(times_years, positions[:, i], color=color, linewidth=3, alpha=0.8)
        
        # Mark start and end times
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Start')
        ax.axvline(x=duration_years, color='green', linestyle='--', alpha=0.7, linewidth=2, label='End')
        
        # Mark Sun's position (0,0,0) with a horizontal line
        ax.axhline(y=0, color='gold', linestyle=':', alpha=0.5, linewidth=1.5, label='Sun (0)')
        
        ax.set_xlabel('Time [years]', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'{comp} Position [AU]', fontsize=13, fontweight='bold')
        ax.set_title(f'Position {comp}-Component vs Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11)
        
        # Set x-axis limits with some padding
        ax.set_xlim(-0.05 * duration_years, 1.05 * duration_years)
        
        # Add a background sun icon in the corner
        ax.text(0.98, 0.05, '☉', transform=ax.transAxes, fontsize=24, 
               color='gold', alpha=0.2, ha='right', va='bottom')
    
    plt.suptitle('Position vs Time During Transfer', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save file
    filename = f"{filename_prefix}_position_vs_time.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Position vs time plot saved: {filename}")
    return filename

def plot_convergence_history(velocity_history, orbital_elements_history, filename_prefix):
    """Plot convergence history and save to files."""
    if not velocity_history:
        return
    
    velocities = np.array(velocity_history)
    elements = np.array(orbital_elements_history)
    
    # Plot 1: Velocity convergence
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 12))
    components = ['X', 'Y', 'Z']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (ax, comp, color) in enumerate(zip(axes1, components, colors)):
        ax.plot(velocities[:, i], color=color, linewidth=2.5)
        final_val = velocities[-1, i]
        ax.axhline(y=final_val, color=color, linestyle='--', alpha=0.7,
                  label=f'Final: {final_val:.6f}', linewidth=2)
        
        ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'v{comp} [AU/yr]', fontsize=13, fontweight='bold')
        ax.set_title(f'Velocity {comp}-Component Convergence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
    
    plt.suptitle('Velocity Convergence History', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    filename1 = f"{filename_prefix}_velocity_convergence.png"
    plt.savefig(filename1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"✓ Velocity convergence plot saved: {filename1}")
    
    # Plot 2: Orbital elements convergence
    if len(elements) > 0:
        element_names = ['Perifocal Distance [AU]', 'Eccentricity', 'Inclination [rad]',
                        'Long. Ascending Node [rad]', 'Arg. Periapsis [rad]', 'Mean Anomaly [rad]']
        
        fig2, axes2 = plt.subplots(3, 2, figsize=(16, 14))
        axes2 = axes2.flatten()
        
        colors2 = plt.cm.Set2(np.linspace(0, 1, 6))
        
        for i, (ax, name, color) in enumerate(zip(axes2, element_names, colors2)):
            ax.plot(elements[:, i], color=color, linewidth=2)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel(name, fontsize=12)
            ax.set_title(name.split(' [')[0], fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Orbital Elements Convergence', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        filename2 = f"{filename_prefix}_orbital_elements.png"
        plt.savefig(filename2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"✓ Orbital elements plot saved: {filename2}")
    
    return filename1

# ============================================================================
# MAIN SOLVER (WITH ANIMATION AND PROPER TIME CONVERSION)
# ============================================================================

def solve_orbit_transfer(r_start_au, r_target_au, transfer_time_years):
    """
    Main solver function for orbital transfer problem.
    
    Parameters:
    -----------
    r_start_au : list or np.array
        Starting position [x, y, z] in AU
    r_target_au : list or np.array
        Target position [x, y, z] in AU
    transfer_time_years : float
        Transfer time in years
        
    Returns:
    --------
    dict : Solution results
    """
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = f"orbit_transfer_{timestamp}"
    
    print("\n" + "=" * 70)
    print("ORBITAL TRANSFER SOLVER")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Start: {r_start_au} AU")
    print(f"Target: {r_target_au} AU")
    print(f"Time: {transfer_time_years} years")
    print("=" * 70)
    
    # Start timing
    start_time = time.time()
    
    # Convert to numpy arrays
    r_start = np.array(r_start_au, dtype=float)
    r_target = np.array(r_target_au, dtype=float)
    
    # Get canonical units
    mu, UT, time_canonical = get_canonical_units(transfer_time_years)
    
    print(f"\nConverting to canonical units...")
    print(f"Gravitational parameter: {mu} AU³/yr²")
    print(f"Time unit (UT): {UT:.6f} years")
    print(f"Time in canonical units: {time_canonical:.6f}")
    print(f"Original time in years: {transfer_time_years}")
    print(f"Reconverted time: {canonical_to_years(time_canonical):.6f} years (verification)")
    
    # Initialize trackers
    velocity_history = []
    orbital_elements_history = []
    
    # Initial guess (can be adjusted)
    initial_velocity_guess = np.array([0.5, 0.5, 0.5])
    
    # Run optimization
    print(f"\nStarting Nelder-Mead optimization...")
    
    optimal_velocity, position_error = nelder_mead_optimizer(
        objective_function,
        initial_velocity_guess,
        r_target,
        r_start,
        mu,
        time_canonical,
        velocity_history,
        orbital_elements_history
    )
    
    # Create final state
    optimal_initial_state = np.concatenate((r_start, optimal_velocity))
    final_state = spy.prop2b(mu, optimal_initial_state, time_canonical)
    orbital_elements = spy.oscelt(final_state, 0, mu)
    
    # Computation time
    computation_time = time.time() - start_time
    
    # Create results dictionary
    result = {
        'timestamp': timestamp,
        'r_start': r_start,
        'r_target': r_target,
        'transfer_time': transfer_time_years,
        'optimal_velocity': optimal_velocity,
        'velocity_magnitude': np.linalg.norm(optimal_velocity),
        'position_error': position_error,
        'final_state': final_state,
        'orbital_elements': orbital_elements,
        'velocity_history': velocity_history,
        'orbital_elements_history': orbital_elements_history,
        'computation_time': computation_time,
        'filename_prefix': filename_prefix,
        'mu': mu,
        'time_canonical': time_canonical,
        'initial_state': optimal_initial_state
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Optimal Initial Velocity: {optimal_velocity} AU/yr")
    print(f"Velocity Magnitude: {np.linalg.norm(optimal_velocity):.6f} AU/yr")
    print(f"Final Position Error: {position_error:.2e} AU")
    print(f"Orbital Eccentricity: {orbital_elements[1]:.6f}")
    print(f"Inclination: {np.degrees(orbital_elements[2]):.2f}°")
    print(f"Number of Iterations: {len(velocity_history)}")
    print(f"Computation Time: {computation_time:.2f} seconds")
    print("=" * 70)
    
    # Generate and save plots
    print(f"\nGenerating plots and animation...")
    
    # Plot trajectory (static) - ALWAYS SHOWS SUN PROMINENTLY
    trajectory_file = plot_trajectory_3d(
        optimal_initial_state,
        mu,
        time_canonical,
        transfer_time_years,  # Pass years for display
        filename_prefix
    )
    
    # Create animation (GIF) - ALWAYS SHOWS SUN PROMINENTLY
    gif_file = create_orbit_animation(
        optimal_initial_state,
        mu,
        time_canonical,
        transfer_time_years,  # Pass years for display
        filename_prefix,
        num_frames=80,  # Reasonable number of frames
        fps=15          # Smooth animation
    )
    
    # Plot position vs time
    time_plot_file = plot_time_progression(
        optimal_initial_state,
        mu,
        time_canonical,
        transfer_time_years,
        filename_prefix
    )
    
    # Plot convergence history
    if velocity_history:
        convergence_file = plot_convergence_history(
            velocity_history,
            orbital_elements_history,
            filename_prefix
        )
    
    result['plot_files'] = {
        'trajectory': trajectory_file,
        'animation': gif_file,
        'position_vs_time': time_plot_file,
        'convergence': convergence_file if 'convergence' in locals() else None
    }
    
    print(f"\n✓ All files saved with prefix: {filename_prefix}")
    print(f"✓ Solution completed in {computation_time:.2f} seconds")
    
    return result

# ============================================================================
# INTERACTIVE MODE (ONLY USER-DEFINED SIMULATION)
# ============================================================================

def get_user_input():
    """Get orbital transfer parameters from user input."""
    print("\n" + "=" * 70)
    print("ORBITAL TRANSFER PARAMETERS INPUT")
    print("=" * 70)
    print("Enter the initial and target positions in Astronomical Units (AU)")
    print("and the desired transfer time in years.")
    print("=" * 70)
    
    # Get user input
    print("\n--- INITIAL POSITION ---")
    r_start = []
    for coord in ['X', 'Y', 'Z']:
        while True:
            try:
                value = float(input(f"  {coord} coordinate (AU): "))
                r_start.append(value)
                break
            except ValueError:
                print("    Please enter a valid number.")
    
    print("\n--- TARGET POSITION ---")
    r_target = []
    for coord in ['X', 'Y', 'Z']:
        while True:
            try:
                value = float(input(f"  {coord} coordinate (AU): "))
                r_target.append(value)
                break
            except ValueError:
                print("    Please enter a valid number.")
    
    print("\n--- TRANSFER TIME ---")
    while True:
        try:
            transfer_time = float(input("  Transfer time (years): "))
            if transfer_time <= 0:
                print("    Time must be positive.")
                continue
            break
        except ValueError:
            print("    Please enter a valid number.")
    
    return r_start, r_target, transfer_time

def confirm_inputs(r_start, r_target, transfer_time):
    """Display and confirm user inputs."""
    print("\n" + "=" * 70)
    print("CONFIRM INPUTS")
    print("=" * 70)
    print(f"Start Position: {r_start} AU")
    print(f"Target Position: {r_target} AU")
    print(f"Transfer Time: {transfer_time} years")
    print("=" * 70)
    
    confirm = input("\nProceed with calculation? (y/n): ").lower()
    return confirm == 'y'

def main():
    """Main execution function - only runs user-defined simulations."""
    print("\n" + "=" * 70)
    print("ORBITAL TRANSFER VELOCITY SOLVER")
    print("=" * 70)
    print("Finds initial velocity for orbital transfer between two points")
    print("using Nelder-Mead optimization (no Lambert theorem).")
    print("=" * 70)
    print("\nThis solver will calculate the optimal initial velocity")
    print("to transfer between your specified positions in the given time.")
    print("All results will be saved as PNG/GIF files with timestamp.")
    print("All time displays will be in years.")
    print("The Sun will be prominently displayed in all 3D visualizations.")
    print("=" * 70)
    
    while True:
        # Get user input
        r_start, r_target, transfer_time = get_user_input()
        
        # Confirm inputs
        if not confirm_inputs(r_start, r_target, transfer_time):
            print("\nCalculation cancelled. Starting over...")
            continue
        
        # Run solver
        try:
            result = solve_orbit_transfer(r_start, r_target, transfer_time)
            
            # Ask for another simulation
            print("\n" + "=" * 70)
            another = input("Run another simulation? (y/n): ").lower()
            if another != 'y':
                print("\nExiting...")
                break
                
        except Exception as e:
            print(f"\n✗ Error during calculation: {e}")
            retry = input("\nTry again with new inputs? (y/n): ").lower()
            if retry != 'y':
                print("\nExiting... ")
                break

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()