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
- $t$: Week index (1,2,...,$T$).

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
\sum_{s=1}^{S} x_{s,t} \cdot D_t \eq D_t forall T
$$

$$ 
E_i \cdot working_hours + O_i + U_i \geq D_i \cdot service_rate
$$

4. **Hiring and Firing Caps**:  
Limit hiring and firing per week:  

$$ 
H_i \leq maxh, \quad F_i \leq maxf 
$$

5. **Overtime Limit**:  
Restrict overtime hours to a percentage of total working hours:  

$$ 
O_i \leq E_i \cdot overtime_rate 
$$

6. **Unmet Demand Limit**
Ensure the Unmet Demand is larger than or equal to the remaining demand after the working hours and the overtime:

$$
U_i \geq D_i - E_i \cdot wokring_hours - O[i]
$$

8. **Budget Constraint**:  
Ensure total costs do not exceed the budget:  

$$ 
\sum_{i=1}^{m} H_i \cdot hiring_cost + F_i \cdot firing_cost + E_i \cdot salary_cost + O_i \cdot overtime_cost \leq budget 
$$
   
### Solving the Model:
The problem is solved using PuLP's LpProblem method, which uses available solvers (e.g., CBC) to find the optimal solution.

## How to Use:
1. Input the parameters
2. Click on "optimize" button

## Requirements:
- Python 3.x
- Streamlit
- Pandas
- PuLP
- Plotly

## Acknowledgments
- PuLP for Linear Programming formulation.
- Plotly for map visualization.
- Stramlit for web app deployment.
