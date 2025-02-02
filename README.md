# Supplier Selection Application
## Overview
This application is a Supplier Selection Optimization tool built using **Streamlit** and **PuLP** for linear programming. It helps businesses optimize supplier selection based on cost, lead time, and quality while ensuring demand fulfillment.

## Features
- Optimization using Linear Programming
- Customizable Weights for Cost, Lead Time, and Quality
- Automatic Demand Generation
- Interactive Visualizations (Supplier Allocations, Cost Distribution, Trade-offs)
- Streamlit-Based User Interface

👉 **Try the app here:** [Supplier Selection App](https://supplier-selection-application-5hhlomtv722jufn4z4wioe.streamlit.app/)

---

## Optimization Model

The optimization model aims to minimize the weighted sum of normalized costs, lead times, and quality scores:

### Indexes
- $s$: Supplier index (1, 2, ..., $S$).
- $t$: Week index (1,2,..., $T$).

### Decision Variable
- $x_{s,t}$: fraction of demand fulfilled by supplier $s$ at time $t$ (Continuous variable).
- $y_{s,t}$: binary variable indicating if a supplier is active in a given week.

### Parameters
- $D_t$: Weekly demand at week $t$.
- $C_s$: The cost of procuring from supplier $s$ per unit.
- $L_s$: The time required for delivery per supplier $s$.
- $Q_s$: Quality score per supplier $s$.
- Weights for Cost $w1$, Lead Time $w2$, and Quality $w3$
- Number of Suppliers (S) – Total number of suppliers considered.
- $Capacity_s$: The maximum units a supplier can provide.
- $OrderRequirement_s$: The smallest quantity that can be ordered.
 
### Objective Function:
Minimize the total cost:

$$
\min Z = \sum_{s=1}^{S} \sum_{t=1}^{T} x_{s,t} \cdot D_t (w1 \cdot C_s + w2 \cdot L_s + w3 \cdot Q_s)
$$

### Constraints:

1. **Demand Satisfaction**:  

$$
\sum_{s=1}^{S} x_{s,t} \cdot D_t = D_t \forall t
$$

2. **Capacity Limits**:  

$$ 
x_{s,t} \cdot D_t \leq capacity_s, \forall s,t 
$$

3. **Minimum Order Requirement**:  

$$ 
x_{s,t} \cdot D_t \leq OrderRequirement_s \cdot y_{s,t} \forall s,t
$$

4. **Binary Supplier Activation**

$$
x_{s,t} \leq y_{s,t} \forall s,t
$$
   
### Solving the Model:
The problem is solved using PuLP's LpProblem method, which uses available solvers (e.g., CBC) to find the optimal solution.

## How to Use
### Inputs
Users can define the following parameters via the Streamlit UI:
- Number of Weeks (Planning Horizon)
- Weights for Cost, Lead Time, and Quality
- Number of Suppliers & Their Properties
- Weekly Demand (Manual or Randomized)

### Output
- Optimal Supplier Allocation per Week

### Visualizations
- **Table:** Presents the supplier allocation and their costs per week
- **Stacked Bar Chart:** Displays supplier allocation per week
- **Pie Chart:** Shows cost contribution by supplier
- **Scatter Plot:** Illustrates cost vs. quality trade-offs

## Requirements:
- Python
- Streamlit
- PuLP
- Plotly
