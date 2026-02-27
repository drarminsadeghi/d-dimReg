import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.colors import LinearSegmentedColormap

#######################################################################################################################


# Assigning values
n = 1
q = 0.3
d_values = np.arange(4, 12)

#######################################################################################################################


# Defining the equation to solve for m
def equation(m, r, d, n, q):
    exponent1 = (d - 3)
    exponent2 = ((d - 2) / 2) * (3 / n - 2)
    exponent3 = (3 * (d - 2)) / (2 * n)
    exponent4 = (-2 * n) / 3 * ((d - 1) / (d - 2))

    term1 = m ** exponent1 / r ** exponent1
    term2 = 1 + (m ** exponent2 * q ** (d - 2)) / r ** exponent3

    return 1 - term1 * term2 ** exponent4

#######################################################################################################################


# Function to safely solve for m
def safe_solve_m(r, d, n, q):
    try:
        initial_guess = np.array([max(1, r / d)])  # array instead of scalar
        m_sol = fsolve(equation, x0=initial_guess, args=(r, d, n, q))[0]

        if np.isfinite(m_sol) and m_sol > 0:
            return m_sol
        else:
            return np.nan
    except:
        return np.nan

#######################################################################################################################


# Defining the function T(r); i.e. the temperature
def T_function_safe(r, d, n, q):
    m_sol = safe_solve_m(r, d, n, q)

    if np.isnan(m_sol):
        return np.nan

    exponent2 = ((d - 2) / 2) * (3 / n - 2)
    exponent3 = (3 * (d - 2)) / (2 * n)

    term_num = (d - 3) - 2 * m_sol ** exponent2 * q ** (d - 2) * r ** (-exponent3)
    term_den = 1 + m_sol ** exponent2 * q ** (d - 2) * r ** (-exponent3)

    T_val = 1 / (4 * np.pi * r) * (term_num / term_den)

    return T_val if T_val >= 0 else np.nan

#######################################################################################################################


# Defining T function for q=0 case; i.e. Schwarzschild
def T_function_q_zero(r, d, n):
    return T_function_safe(r, d, n, q=0)

#######################################################################################################################


# Defining r range for plotting
r_values = np.linspace(0.001,1.5, 1000)

#######################################################################################################################


# Custom coloring
color_sequence_hex = [
    '#000000',  # Black
    '#8800C7',  # purple
    '#3500C7',  # Blue
    '#0091D9',  # cyan
    '#56E600',  # light green
    '#00731E',  # Green
    '#EDA900',  # orange
    '#FF3838',  # Red
]

custom_cmap = LinearSegmentedColormap.from_list('custom_8color', color_sequence_hex)

colors = custom_cmap(np.linspace(0, 1, len(d_values)))

#######################################################################################################################

# Preparing the plot
plt.figure(figsize=(12, 7))

for i, d in enumerate(d_values):
    # Computing T values for q=0.3
    T_values = [T_function_safe(r, d, n, q) for r in r_values]
    plt.plot(r_values, T_values, label=f'd={d}', color=colors[i])

    # Computing T values for q=0
    y_values_q_zero = np.array([T_function_q_zero(r, d, n) for r in r_values])
    y_values_q_zero[y_values_q_zero < 0] = np.nan  # Replace negative values with NaN

    # Plotting for q=0 with dashed lines
    plt.plot(r_values, y_values_q_zero, color=colors[i], linestyle='--', alpha=0.7)


plt.axvline(0, color='gray', linestyle='-', linewidth=1)  # Darker horizontal line at y=0

# Set axis limits
plt.ylim(0, 1.6)
plt.xlim(0, 1.5)

#######################################################################################################################

# Labels,grids and Legends

# Secondary legend for line styles (solid/dashed)
solid_line = plt.Line2D([0], [0], color='black', linestyle='-', label=r'Regular ($\tilde{q}$ ≠ 0)')
dashed_line = plt.Line2D([0], [0], color='black', linestyle='--', label=r'ST ($\tilde{q}$ = 0)')
legend2 = plt.legend(handles=[solid_line, dashed_line], loc='upper right', bbox_to_anchor=(1,1))
plt.gca().add_artist(legend2)  # Add the secondary legend to the plot

plt.xlabel(r'${r}_h$')
plt.ylabel(r'${T}$')
plt.title('n=1 (Hayward)')
plt.legend( loc='upper right', bbox_to_anchor=(1, 0.9))
plt.grid(True, which='major', color='lightgray', linestyle='-', linewidth=0.8)  # Major grid (light gray)
plt.grid(True, which='minor', color='#d3d3d3', linestyle='--', linewidth=0.5)  # Minor grid (lighter gray)
plt.minorticks_on()

plt.tick_params(which='major', length=10, width=1, direction='inout')
plt.tick_params(which='minor', length=3, width=1, direction='in')

#######################################################################################################################

plt.savefig('Temperature.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.show()
