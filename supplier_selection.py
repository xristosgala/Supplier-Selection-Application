import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, LpStatus
import random

def solve_supplier_selection_problem(num_weeks, w1, w2, w3, num_suppliers, suppliers, costs, lead_times, quality_scores, 
                                     capacities, min_order, num_active_suppliers, weekly_demand, service_rate):

    # Create optimization model
    model = LpProblem("Supplier_Selection_Optimization", LpMinimize)

    # Decision variables: Fraction of demand fulfilled by each supplier per week
    x = {(s, t): LpVariable(f"Supplier_{s}_Week_{t}", lowBound=0, upBound=1, cat='Continuous')
        for s in suppliers for t in range(num_weeks)}

    # Binary variables for whether a supplier is active in a given week
    y = {(s, t): LpVariable(f"Active_Supplier_{s}_Week_{t}", cat='Binary')
        for s in suppliers for t in range(num_weeks)}

    # Objective function: Weighted sum (normalized)
    cost_min, cost_max = min(costs.values()), max(costs.values())
    lead_min, lead_max = min(lead_times.values()), max(lead_times.values())
    quality_min, quality_max = min(quality_scores.values()), max(quality_scores.values())

    def safe_normalization(value, min_val, max_val, epsilon=1e-6):
        return (value - min_val) / (max_val - min_val + epsilon)
    
    def safe_normalization_quality(value, min_val, max_val, epsilon=1e-6):
        return (max_val - value) / (max_val - min_val + epsilon)
    
    normalized_costs = {s: safe_normalization(costs[s], cost_min, cost_max) for s in suppliers}
    normalized_lead = {s: safe_normalization(lead_times[s], lead_min, lead_max) for s in suppliers}
    normalized_quality = {s: safe_normalization_quality(quality_scores[s], quality_min, quality_max) for s in suppliers}

    model += lpSum([
        x[s, t] * weekly_demand[t] * (w1 * normalized_costs[s] + w2 * normalized_lead[s] + w3 * normalized_quality[s]) 
        for s in suppliers for t in range(num_weeks)
    ])

    # Constraints
    for t in range(num_weeks):
        model += lpSum([x[s, t] * weekly_demand[t] for s in suppliers]) == weekly_demand[t]  # Meet weekly demand
        model += lpSum([y[s, t] for s in suppliers]) == num_active_suppliers  # Set number of active suppliers per week
        for s in suppliers:
            model += x[s, t] * weekly_demand[t] <= capacities[s]  # Capacity constraint
            model += x[s, t] * weekly_demand[t] >= min_order[s] * y[s, t]  # Minimum order constraint (only if active)
            model += x[s, t] <= y[s, t]  # Ensure y[s, t] is 1 if any quantity is ordered

    # Solve the model
    model.solve()

    # Output selected suppliers and allocations per week
    selected_allocations = {(s, t): x[s, t].varValue * weekly_demand[t] for s in suppliers for t in range(num_weeks) if x[s, t].varValue > 0}

    model_result = LpStatus[model.status]
      
    # Create a new list to hold details with added cost info
    detailed_results = []
    for (s, t), allocation in selected_allocations.items():
        allocation_cost = allocation * costs[s]  # Cost for this allocation
        detailed_results.append({
            "Week": t + 1,  # Week number
            "Demand": weekly_demand[t],  # Weekly demand
            "Supplier": s + 1,  # Supplier index (adjusted for 1-based index)
            "Allocation": round(allocation, 0),  # Rounded allocation
            "Cost": round(allocation_cost, 2)  # Rounded cost for that allocation
        })

    return detailed_results, model_result


# Streamlit App
st.title("Supplier Selection Application")

# Input Fields
st.sidebar.header("Input Parameters")
num_weeks = st.sidebar.number_input("Number of Weeks", min_value=1, max_value=52, value=6)
w1 = st.sidebar.number_input("Cost Weight", min_value=0.0, max_value=1.0, value=0.5)
w2 = st.sidebar.number_input("Lead Time Weight", min_value=0.0, max_value=1.0, value=0.3)
w3 = st.sidebar.number_input("Quality Weight", min_value=0.0, max_value=1.0, value=0.2)
num_suppliers = st.sidebar.number_input("Number of Suppliers", min_value=1, max_value=10, value=3)
suppliers =  [i for i in range(num_suppliers)]
costs = {supplier: st.sidebar.number_input(f"Cost for Supplier {supplier+1}", min_value=1, max_value=100, value=10) for supplier in suppliers}
lead_times = {supplier: st.sidebar.number_input(f"Lead Time for Supplier {supplier+1}", min_value=1, max_value=100, value=12) for supplier in suppliers}
quality_scores = {supplier: st.sidebar.number_input(f"Quality Score for Supplier {supplier+1}", min_value=1, max_value=100, value=8) for supplier in suppliers}
capacities = {supplier: st.sidebar.number_input(f"Capacity for Supplier {supplier+1}", min_value=1, max_value=1000, value=100) for supplier in suppliers}
min_order = {supplier: st.sidebar.number_input(f"Minimum Order for Supplier {supplier+1}", min_value=10, max_value=100, value=10) for supplier in suppliers}
num_active_suppliers = st.sidebar.number_input("Minimum Number of Active Suppliers", min_value=1, max_value=10, value=2)
demand_range = st.sidebar.slider("Demand Range", min_value=10, max_value=1000, value=(20, 100))
random_demand = st.sidebar.checkbox("Generate Random Demand", value=True)
service_rate = st.sidebar.slider("Service Rate", min_value=0.00, max_value=1.00, value=0.95)

if random_demand:
    weekly_demand = [random.randint(demand_range[0], demand_range[1]) for _ in range(num_weeks)]
else:
    weekly_demand = [st.sidebar.number_input(f"Demand for Week {i+1}", min_value=1, value=random.randint(1,1000)) for i in range(num_weeks)]

# Solve and Display Results
if st.button("Optimize"):
    results, model_result= solve_supplier_selection_problem(num_weeks, w1, w2, w3, num_suppliers, suppliers, costs, lead_times, quality_scores, 
                                     capacities, min_order, num_active_suppliers, weekly_demand, service_rate)

    st.subheader("Optimization Results")
    if model_result=='Optimal':
        st.success("Status: Optimal")
      
        # Convert to DataFrame for better formatting
        df = pd.DataFrame(results)
        
        # Display the DataFrame in Streamlit
        st.write("Results in a Tabular Form:")
      
        if not df.empty:
            st.dataframe(df.style.format({"Allocation": "{:.0f}", "Cost": "${:.2f}"}))
    else:
        st.warning("No feasible solution found!")

