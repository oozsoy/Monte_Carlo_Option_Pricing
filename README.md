# Monte Carlo for Option Pricing

This repository explores Monte Carlo simulation techniques for pricing financial derivatives, with a particular focus on **exotic options**. These are options with more complex features than standard European-style ("vanilla") options — for example, path-dependence, barriers, or early exercise features — where analytical solutions are often unavailable.

## 📌 Motivation

Monte Carlo (MC) simulation is a **flexible and powerful tool** in financial engineering, particularly well-suited for pricing **complex and path-dependent options**. Its appeal lies in its generality — you can simulate almost any stochastic process — but this comes at a cost: **high computational expense**.

This project is dedicated to **systematically studying variance reduction methods** that make Monte Carlo practical for **real-world applications**, especially in high-dimensional and exotic pricing scenarios. These methods aim to reduce the **number of simulations needed** to achieve a given level of accuracy, thereby improving performance and feasibility in production environments.

### 🧠 Why Variance Reduction?

In MC simulation, we generate many random paths to estimate expected values under the risk-neutral measure. The standard error of the estimate decreases slowly — at a rate of $1/\sqrt{N}$, where $N$ is the number of paths. For **exotic options**, the payoff depends on more than just the terminal price (e.g., average price, hitting barriers), requiring even more paths for stable estimation.

This challenge is **magnified in multi-dimensional problems** — for instance, pricing a basket option where multiple underlying assets evolve simultaneously. Here, each "dimension" refers to one asset or source of uncertainty. As dimensions increase, the number of paths required to cover the space grows **exponentially**, often called the "curse of dimensionality."

Hence, we investigate **variance reduction techniques** like:
- **Antithetic variates**
- **Control variates** (including delta- and gamma-based methods)
- **Quasi-random sequences** (e.g., Sobol, Halton)

These methods aim to reduce the **variance of the estimator** while keeping the mean unbiased, improving **efficiency** (i.e., smaller standard error for the same number of paths).

## 📂 Structure

This repo will evolve over time. Currently, it includes:

### `notebooks/`

- `quasi_randnum.ipynb`:  
  Compares convergence and variance for vanilla and exotic options using pseudo-random vs quasi-random number sequences (Sobol/Latin Hypercube).
  
- `variance_reduc.ipynb`:  
  Demonstrates antithetic variates, control variates (delta/gamma-based), and their combinations. Includes benchmarking of **standard error vs. computation time**.

### `scripts/` *(coming soon)*

Reusable modules and utilities for simulations, payoff functions, and experimental tracking.

## 📘 Topics Covered

- Monte Carlo pricing under the risk-neutral measure
- Variance reduction techniques: theory and implementation
  - Antithetic variates
  - Control variates
  - Quasi-random numbers (low-discrepancy sequences)
- Trade-off analysis: variance vs. computation time
- **Planned**: Path-dependent and multi-asset exotic options
