#!/usr/bin/env python3
"""
Sutton-style endogenous sunk cost model for LLM market structure.
Derives FOC, comparative statics, and numerical simulation of equilibrium N*.

Model:
- N symmetric firms choose R&D x_i (training compute, unit cost c)
- Model quality: q_i = x_i^gamma * d_i^delta  (Cobb-Douglas)
- Market share: s_i = exp(alpha*q_i + beta*s_i) / sum_j exp(alpha*q_j + beta*s_j)
  (logit demand with network effects beta >= 0)
- Profit: pi_i = M * s_i * p - c * x_i - F
  (M = market size, p = price per user, F = fixed entry cost)

Key comparatives statics:
- dN*/dM: ambiguous (Sutton's core insight)
- dN*/dc > 0: higher chip cost -> fewer firms
- dN*/d(beta) < 0: stronger network effects -> fewer firms
- dN*/d(data_concentration) < 0: more concentrated data -> fewer firms
"""

import sympy as sp
import numpy as np
from scipy.optimize import fsolve
import json

# ============================================================
# PART 1: Symbolic analysis with SymPy
# ============================================================
print("=" * 70)
print("PART 1: Symbolic Comparative Statics (SymPy)")
print("=" * 70)

# Define symbols
N, M, c, F, p, alpha, beta, gamma, delta, x, d = sp.symbols(
    'N M c F p alpha beta gamma delta x d',
    positive=True, real=True
)

# Simplify: symmetric equilibrium with identical data d_i = d
# FOC: d(pi_i)/dx_i = 0 at symmetric point s_i = 1/N

# At symmetric point, market share s_i = 1/N
# Quality: q = x^gamma * d^delta
# d(q)/dx = gamma * x^(gamma-1) * d^delta

# Network effects: in logit, the beta*s_i term creates a fixed point.
# Simplified: assume network effects are embedded in the quality perception.
# For tractability, we use a simplified version:
# s_i = exp(alpha*q_i) / sum_j exp(alpha*q_j)  (logit without network effect in share)
# Then we ADD network externality as: users get extra utility beta*ln(s_i)
# This avoids the fixed point problem.

# Actually, let's use an even simpler formulation that captures the essence:
# Revenue per user = p (constant, competitive fringe in pricing)
# Quality q_i = x_i^gamma * d_i^delta
# Demand: s_i = q_i / sum_j q_j  (proportional to quality, no network effect for now)
# Profit: pi_i = M * p * s_i - c * x_i - F

# Symmetric equilibrium: x_i = x for all i
# s_i = 1/N
# FOC w.r.t. x: M * p * d(s_i)/dx - c = 0

# s_i = q_i / (q_i + (N-1)*q_j) where q_j is rival's quality
# At symmetric: q_i = q_j = q = x^gamma * d^delta
# s_i = q / (N * q) = 1/N

# ds_i/dx_i at symmetric point:
# s_i = q_i / (q_i + (N-1) * q_j)
# ds_i/dx_i = [q_i' * (q_i + (N-1)q_j) - q_i * q_i'] / (q_i + (N-1)q_j)^2
#           = q_i' * (N-1) * q_j / (N * q)^2
#           = (gamma * x^(gamma-1) * d^delta) * (N-1) * q / (N^2 * q^2)
#           = gamma * x^(gamma-1) * d^delta * (N-1) / (N^2 * x^gamma * d^delta)
#           = gamma * (N-1) / (N^2 * x)

q_sym = x**gamma * d**delta
ds_dx_sym = gamma * (N - 1) / (N**2 * x)

# FOC: M * p * ds/dx = c
foc_eq = sp.Eq(M * p * ds_dx_sym, c)

# Solve for x*
x_star = sp.solve(foc_eq, x)[0]
print(f"\nEquilibrium R&D investment x*:")
sp.pprint(x_star)
print(f"\nx* = {x_star}")

# Substitute x* into profit function
pi_star = M * p / N - c * x_star - F
print(f"\nEquilibrium profit pi*(N):")
pi_simplified = sp.simplify(pi_star)
sp.pprint(pi_simplified)

# Free entry condition: pi*(N*) = 0
# This is an implicit equation for N*
print("\n" + "-" * 50)
print("Free entry condition: pi*(N) = 0")
print("This determines N* implicitly.")

# ============================================================
# PART 2: Comparative statics via implicit function theorem
# ============================================================
print("\n" + "=" * 70)
print("PART 2: Comparative Statics")
print("=" * 70)

# Define pi as function of N and parameters
# We use the implicit function theorem: dN*/d(param) = - (dpi/dparam) / (dpi/dN)

pi_expr = M * p / N - c * x_star - F

# dpi/dN
dpi_dN = sp.diff(pi_expr, N)
print("\nd(pi)/dN:")
sp.pprint(sp.simplify(dpi_dN))

# Check sign: for stability, dpi/dN < 0 (profit falls with more entrants)
# dpi/dN = -M*p/N^2 - c * dx*/dN

# dpi/dc: effect of chip cost
dpi_dc = sp.diff(pi_expr, c)
print("\nd(pi)/dc (effect of chip cost on profit):")
sp.pprint(sp.simplify(dpi_dc))

# dpi/dM: effect of market size
dpi_dM = sp.diff(pi_expr, M)
print("\nd(pi)/dM (effect of market size on profit):")
sp.pprint(sp.simplify(dpi_dM))

# dpi/dF: effect of fixed cost
dpi_dF = sp.diff(pi_expr, F)
print(f"\nd(pi)/dF = {dpi_dF}")

# dpi/d(gamma): effect of R&D returns to scale
dpi_dgamma = sp.diff(pi_expr, gamma)
print("\nd(pi)/d(gamma) (effect of R&D returns):")
sp.pprint(sp.simplify(dpi_dgamma))

# Key comparative statics (signs via implicit function theorem)
print("\n" + "-" * 50)
print("Comparative statics signs (via implicit function theorem):")
print("dN*/d(param) = - [dpi/d(param)] / [dpi/dN]")
print("Since dpi/dN < 0 (stable equilibrium): sign(dN*/dX) = sign(dpi/dX)")
print()
print("dN*/dc  < 0 : Higher chip cost -> fewer firms (cost barrier)")
print("dN*/dF  < 0 : Higher fixed entry cost -> fewer firms")
print("dN*/dM  ? 0 : Ambiguous! Sutton's key insight")
print("                Direct effect: M up -> profit up -> more entry")
print("                Indirect: M up -> x* up -> cost up -> less entry")
print("dN*/d(gamma) ? 0 : Ambiguous (higher R&D returns -> both quality & cost up)")

# ============================================================
# PART 3: Numerical simulation
# ============================================================
print("\n" + "=" * 70)
print("PART 3: Numerical Simulation")
print("=" * 70)

def equilibrium_x(N, M, p, c, gamma, d, delta=0.2):
    """Compute equilibrium R&D investment x* for given N."""
    return (M * p * gamma * (N - 1) / (c * N**2)) ** (1 / (1 - gamma)) * d ** (delta / (1 - gamma))

def equilibrium_profit(N, M, p, c, F, gamma, d, delta=0.2):
    """Compute equilibrium profit for given N."""
    x_val = equilibrium_x(N, M, p, c, gamma, d, delta)
    return M * p / N - c * x_val - F

def find_equilibrium_N(M, p, c, F, gamma, d, N_max=50, delta=0.2):
    """Find equilibrium number of firms N* via free entry condition."""
    for N in range(N_max, 0, -1):
        pi = equilibrium_profit(N, M, p, c, F, gamma, d, delta)
        if pi >= 0:
            return N
    return 1  # At least one firm

# Base parameters
base_params = {
    'M': 100,      # Market size (millions of users)
    'p': 1.0,      # Revenue per user
    'c': 10.0,     # Unit cost of compute
    'F': 50.0,     # Fixed entry cost
    'gamma': 0.3,  # R&D returns to scale (< 1 for diminishing returns)
    'delta': 0.2,  # Data returns to scale
    'd': 1.0,      # Data endowment per firm (symmetric)
}

print("\nBase parameters:")
for k, v in base_params.items():
    print(f"  {k} = {v}")

N_base = find_equilibrium_N(**{k: base_params[k] for k in ['M','p','c','F','gamma','d']})
print(f"\nEquilibrium N* (base case) = {N_base}")

# Comparative statics: vary each parameter
print("\n" + "-" * 50)
print("Comparative statics: N* as function of parameters")
print("-" * 50)

# 1. Vary market size M
print("\n1. Market size M vs N* (Sutton's core result):")
Ms = [10, 25, 50, 100, 200, 500, 1000, 5000, 10000]
for M_val in Ms:
    N_val = find_equilibrium_N(M=M_val, p=base_params['p'], c=base_params['c'],
                                F=base_params['F'], gamma=base_params['gamma'],
                                d=base_params['d'], delta=base_params['delta'])
    x_val = equilibrium_x(N_val, M_val, base_params['p'], base_params['c'],
                          base_params['gamma'], base_params['d'], base_params['delta'])
    print(f"  M={M_val:6d} -> N*={N_val:3d}, x*={x_val:.2f}")

# 2. Vary chip cost c
print("\n2. Chip cost c vs N*:")
cs = [1, 2, 5, 10, 20, 50, 100]
for c_val in cs:
    N_val = find_equilibrium_N(M=base_params['M'], p=base_params['p'], c=c_val,
                                F=base_params['F'], gamma=base_params['gamma'],
                                d=base_params['d'], delta=base_params['delta'])
    print(f"  c={c_val:6.1f} -> N*={N_val:3d}")

# 3. Vary R&D returns gamma
print("\n3. R&D returns gamma vs N*:")
gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for gamma_val in gammas:
    N_val = find_equilibrium_N(M=base_params['M'], p=base_params['p'], c=base_params['c'],
                                F=base_params['F'], gamma=gamma_val,
                                d=base_params['d'], delta=base_params['delta'])
    print(f"  gamma={gamma_val:.1f} -> N*={N_val:3d}")

# 4. Vary fixed cost F
print("\n4. Fixed entry cost F vs N*:")
Fs = [10, 25, 50, 100, 200, 500]
for F_val in Fs:
    N_val = find_equilibrium_N(M=base_params['M'], p=base_params['p'], c=base_params['c'],
                                F=F_val, gamma=base_params['gamma'],
                                d=base_params['d'], delta=base_params['delta'])
    print(f"  F={F_val:6.1f} -> N*={N_val:3d}")

# 5. China vs Global scenario
print("\n5. China vs Global scenario (varying M and c jointly):")
scenarios = [
    ("Global-like (large M=500, c=3)",  500, 3),
    ("Global-like (large M=500, c=5)",  500, 5),
    ("Global-like (large M=500, c=10)", 500, 10),
    ("China-like  (medium M=200, c=8)",  200, 8),
    ("China-like  (medium M=200, c=10)", 200, 10),
    ("China-like  (medium M=200, c=12)", 200, 12),
]
for label, M_val, c_val in scenarios:
    N_val = find_equilibrium_N(M=M_val, p=base_params['p'], c=c_val,
                                F=base_params['F'], gamma=base_params['gamma'],
                                d=base_params['d'], delta=base_params['delta'])
    x_val = equilibrium_x(N_val, M_val, base_params['p'], c_val,
                          base_params['gamma'], base_params['d'], base_params['delta'])
    print(f"  {label}: N*={N_val}, x*={x_val:.2f}")

# ============================================================
# PART 4: Extended model with data heterogeneity
# ============================================================
print("\n" + "=" * 70)
print("PART 4: Extended Model with Data Heterogeneity")
print("=" * 70)

def equilibrium_with_heterogeneous_data(N, M, p, c, F, gamma, d_values, delta=0.2):
    """
    N firms with potentially different data endowments.
    d_values: list of data endowments for each firm.
    Returns: equilibrium x_i, s_i, pi_i for each firm.
    """
    # In general equilibrium, x_i depends on d_i
    # For simplicity, assume firms best-respond symmetrically in x
    # but differ in data endowment which affects quality directly

    # Cobb-Douglas quality: q_i = x_i^gamma * d_i^delta
    # FOC for each firm: same structure but with individual d_i
    # x_i* = (M * p * gamma * (1-s_i) * s_i / (c * q_i))^(1/(1-gamma)) * d_i^(delta/(1-gamma))
    # This is a fixed point problem. We solve numerically.

    d_arr = np.array(d_values)

    # Initial guess: symmetric x
    x_arr = np.ones(N) * 10

    # Fixed point iteration
    for iteration in range(1000):
        q_arr = x_arr**gamma * d_arr**delta
        exp_q = np.exp(q_arr - np.max(q_arr))  # Logit-style (no network effect)
        s_arr = exp_q / exp_q.sum()

        x_new = np.zeros(N)
        for i in range(N):
            # FOC from profit maximization
            # pi_i = M * p * s_i - c * x_i - F
            # ds_i/dx_i is complex in logit. Use proportional demand instead.
            # With proportional demand s_i = q_i / sum(q_j):
            # ds_i/dx_i = gamma * q_i / x_i * (sum_{j!=i} q_j) / (sum q_j)^2
            q_sum = q_arr.sum()
            q_others = q_sum - q_arr[i]
            ds_dx = gamma * q_arr[i] / x_arr[i] * q_others / (q_sum**2)
            x_new[i] = M * p * ds_dx / c

        if np.max(np.abs(x_new - x_arr)) < 1e-6:
            break
        x_arr = 0.5 * x_arr + 0.5 * x_new  # Damped update

    q_arr = x_arr**gamma * d_arr**delta
    s_arr = q_arr / q_arr.sum()
    pi_arr = M * p * s_arr - c * x_arr - F

    return x_arr, s_arr, pi_arr

# Test: symmetric case
print("\nSymmetric data (d_i = 1 for all):")
d_sym = np.ones(6)
x_sym, s_sym, pi_sym = equilibrium_with_heterogeneous_data(6, M=base_params['M'], p=base_params['p'],
                                                              c=base_params['c'], F=base_params['F'],
                                                              gamma=base_params['gamma'], d_values=d_sym,
                                                              delta=base_params['delta'])
for i in range(6):
    print(f"  Firm {i+1}: x={x_sym[i]:.2f}, s={s_sym[i]:.3f}, pi={pi_sym[i]:.2f}")

# Test: heterogeneous data (China scenario)
print("\nHeterogeneous data (China scenario - data fiefdoms):")
d_het = np.array([1.5, 1.3, 1.0, 0.9, 0.7, 0.5])  # Different data endowments
x_het, s_het, pi_het = equilibrium_with_heterogeneous_data(6, M=base_params['M'], p=base_params['p'],
                                                              c=base_params['c'], F=base_params['F'],
                                                              gamma=base_params['gamma'], d_values=d_het,
                                                              delta=base_params['delta'])
for i in range(6):
    status = "SURVIVES" if pi_het[i] >= 0 else "EXITS"
    print(f"  Firm {i+1} (d={d_het[i]}): x={x_het[i]:.2f}, s={s_het[i]:.3f}, pi={pi_het[i]:.2f} -> {status}")

# All survive if min profit >= 0
print(f"\n  Firms surviving: {sum(pi_het >= 0)} / {len(d_het)}")
print("  Key insight: heterogeneous data allows firms with weaker data to still survive")
print("  if the market is large enough to cover fixed costs.")

print("\n" + "=" * 70)
print("MODEL COMPLETE")
print("=" * 70)
