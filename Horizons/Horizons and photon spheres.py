#libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sympy import symbols, solve, Eq, lambdify
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap

#######################################################################################################################

# Defining symbols
q, r, d = symbols('q r d')

#######################################################################################################################

# Assigning values
m = 1
n = 1

#######################################################################################################################

# Defining the horizon equation
equation = Eq(1 - (m / r ** (d - 3)) * (1 + (m ** (3 / n - 2) * q ** (d - 2)) / r ** ((3 * (d - 2)) / (2 * n))) ** (
        (-2 * n) / 3 * ((d - 1) / (d - 2))), 0)

#######################################################################################################################

# Solving the equation for q in terms of r and d
q_solution = solve(equation, q)[0]

#######################################################################################################################

# Converting the solution to a numerical function
q_func = lambdify((r, d), q_solution, modules='numpy')

#######################################################################################################################

# Defining the expression and its derivative numerically
def expr(r, q, d, n):
    return 1 / r ** 2 * (1 - 1 / r ** (d - 3) * (1 + q ** (d - 2) / r ** ((3 * (d - 2)) / (2 * n))) ** (
            (-2 * n * (d - 1)) / (3 * (d - 2))))

def dexpr_num(r, q, d, n):
    h = 1e-5
    return (expr(r + h, q, d, n) - expr(r, q, d, n)) / h

#######################################################################################################################

# Improved function to find q with better initial guesses and handling numerical issues
def find_q(d_val, r_val, n_val, q_guess=1):
    func = lambda q: dexpr_num(r_val, q, d_val, n_val)
    q_sol, info, ier, msg = fsolve(func, q_guess, full_output=True)
    if ier == 1 and q_sol[0] >= 0:
        return q_sol[0]
    else:
        return np.nan

#######################################################################################################################

# Defining the q formula directly
def q_formula(d):
    return ((d - 3) / 2) ** (1 / (d - 2)) * (2 / (d - 1)) ** ((d - 1) / ((d - 3) * (d - 2)))

# Define the r value for the last point in dotted curve
def r_last_point(d):
    return 2 ** (-1 / (-3 + d)) * (-1 + d) ** (1 / (-3 + d))

# Define the new equation to solve for minimum r
def find_r_min(d):
    def equation(r):
        term = (4 ** (1 / (6 - 5 * d + d ** 2)) *
                (-3 + d) ** (1 / (-2 + d)) *
                (1 / (-1 + d)) ** ((-1 + d) / (6 - 5 * d + d ** 2)))
        expr_value = 1 - r ** (3 - d) * (1 + term ** (-2 + d) * r ** (-3 * (-2 + d) / (2 * n))) ** (
                -2 * (-1 + d) * n / (3 * (-2 + d)))
        return expr_value

    try:
        r_min, info, ier, msg = fsolve(equation, x0=0.5, full_output=True)
        if ier == 1 and r_min[0] >= 0:
            return r_min[0]
        else:
            print(f"Warning: Minimum r not found or invalid for d={d}.")
            return np.nan
    except Exception as e:
        print(f"Error finding minimum r for d={d}: {e}")
        return np.nan

#######################################################################################################################

# Generating a range of r values for each d
d_vals = range(4, 12)
num_r_points = 400

#######################################################################################################################

# Custom coloring
color_sequence_hex = [
    '#000000',  # Black
    '#8800C7',  # purple
    '#3500C7',  #  Blue
    '#0091D9',  # cyan
    '#56E600',  # light green
    '#00731E',  #  Green
    '#EDA900',  # orange
    '#FF3838',  # Red
]

custom_cmap = LinearSegmentedColormap.from_list('custom_8color', color_sequence_hex)

colors = custom_cmap(np.linspace(0, 1, len(d_vals)))

#######################################################################################################################

# Preparing the plot
plt.figure(figsize=(12, 8))

#######################################################################################################################

# Computing and plotting q for each d using the symbolic method
for idx, d_val in enumerate(d_vals):
    r_values_full = np.linspace(0.00001, 1, num_r_points)  # Full range for solid curves
    q_values_sympy = q_func(r_values_full, d_val)
    plt.plot(r_values_full, q_values_sympy, color=colors[idx], linestyle='-', label=f'd={d_val}')


# Initialize lists to store minimum r values for dashed line
dashed_r_min = []

#######################################################################################################################

# Computing and plottng q for each d using the numerical method
for idx, d_val in enumerate(d_vals):
    q_threshold = q_formula(d_val)
    r_last = r_last_point(d_val)

    # Generating r values based on the d values
    if d_val == 5:
        r_values_num = np.linspace(0.74, r_last, num_r_points)
    elif d_val == 9:
        r_values_num = np.linspace(0.84, r_last, num_r_points)
    else:
        r_lower_bound = find_r_min(d_val)
        if np.isnan(r_lower_bound):
            continue  # Skip if r_lower_bound is not found
        r_values_num = np.linspace(r_lower_bound, r_last, num_r_points)

    q_values_num = []
    q_guess = 1.0
    for r_val in r_values_num:
        try:
            q_sol = find_q(d_val, r_val,n, q_guess)
            q_values_num.append(q_sol)
            q_guess = q_sol if not np.isnan(q_sol) else 1.0  # Use the last solution as the next guess
        except:
            q_values_num.append(np.nan)

    # Separating r and q values for above and below threshold
    r_values_above = [r for r, q in zip(r_values_num, q_values_num) if q > q_threshold]
    q_values_above = [q for q in q_values_num if q > q_threshold]
    r_values_below = [r for r, q in zip(r_values_num, q_values_num) if q <= q_threshold]
    q_values_below = [q for q in q_values_num if q <= q_threshold]

    # Append the last point to the dotted curve
    r_values_below.append(r_last)
    q_values_below.append(0)

    # Plot the parts above and below the threshold q separately
    plt.plot(r_values_above, q_values_above, color=colors[idx], linestyle='dotted')
    plt.plot(r_values_below, q_values_below, color=colors[idx], linestyle='--', dashes=(6, 3))

    # Store the minimum r values for dashed line
    if len(r_values_above) > 0:
        min_r = np.min(r_values_above)
        dashed_r_min.append((d_val, min_r))

        # Plot a dot at the minimum r point
        plt.scatter(min_r, q_threshold, color=colors[idx], s=40, zorder=5)  # Adjust 's' for dot size

        # Add q_ext label for minimum r on dashed line
        plt.text(min_r, q_threshold + 0.01, r'$\tilde{q}_{{ext}}$', color=colors[idx], ha='right', fontsize=8)

    # Find and label the maximum q point on the dashed line as q_dis
    if len(r_values_above) > 0:
        max_q_dis = np.max(q_values_above)
        max_r_dis = r_values_above[q_values_above.index(max_q_dis)]
        plt.scatter(max_r_dis, max_q_dis, color=colors[idx], s=50, marker='o', zorder=5)  # Dot for maximum q_dis
        plt.text(max_r_dis, max_q_dis + 0.01, r'$\tilde{q}_{{deg}}$', color=colors[idx], ha='right', fontsize=8)

#######################################################################################################################

# Labeling axes in the plot
plt.xlabel(r'$\tilde{r}$')
plt.ylabel(r'$\tilde{q}$')
plt.title('n=1 (Hayward)')

#######################################################################################################################

# Creating the legend (for d values)
legend1 = plt.legend(loc='upper right')
plt.gca().add_artist(legend1)

# Creating the second legend (for line styles)
solid_line = mlines.Line2D([], [], color='black', linestyle='-', label='Horizon')
dashed_line = mlines.Line2D([], [], color='black', linestyle='dotted', label='COPS')
dashdot_line = mlines.Line2D([], [], color='black', linestyle='--', dashes=(6, 3), label='BHPS')
legend2 = plt.legend(handles=[solid_line, dashed_line, dashdot_line], loc='upper left')

plt.grid(True, zorder=0)

#######################################################################################################################

# Save and show the plot
plt.savefig('HVeffn1.eps', format='eps', bbox_inches='tight', pad_inches=0)
plt.show()
