import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, LpStatus
import random
import matplotlib.pyplot as plt
import seaborn as sns

def solve_supplier_selection_problem(num_weeks, w1, w2, w3, num_suppliers, suppliers, costs, lead_times, quality_scores, 
                                     capacities, min_order, weekly_demand):

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

    def safe_normalization(value, min_val, max_val):
        return 0 if max_val == min_val else (value - min_val) / (max_val - min_val)
    
    def safe_normalization_quality(value, min_val, max_val):
        return 0 if max_val == min_val else (max_val - value) / (max_val - min_val)
    
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
    total_cost = 0  # Variable to store the total cost
    for t in range(num_weeks):
        week_data = {"Week": t + 1, "Demand": weekly_demand[t]}  # Initialize week data with demand
        for s in suppliers:
            allocation = selected_allocations.get((s, t), 0)  # Get allocation for supplier s in week t
            week_data[f"Supplier {s +  1} Allocation"] = round(allocation, 2) # Add allocation for this supplier in this week
            allocation_cost = allocation * costs[s]  # Cost for this allocation
            week_data[f"Supplier {s + 1} Cost"] = round(allocation_cost, 2)  # Add cost for this supplier in this week
            total_cost += allocation_cost
        detailed_results.append(week_data)

    return detailed_results, model_result, total_cost

def plot_supply_chain_graphs(df, suppliers, costs, quality_scores):
    # Stacked Bar Chart: Weekly Supplier Allocation
    allocation_cols = [f"Supplier {s + 1} Allocation" for s in suppliers]
    df_plot = df.melt(id_vars=["Week"], value_vars=allocation_cols, var_name="Supplier", value_name="Allocation")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Week", y="Allocation", hue="Supplier", data=df_plot, ax=ax)
    ax.set_ylabel("Allocation Amount")
    ax.set_title("Supplier Allocation per Week")
    st.pyplot(fig)
    
    # Pie Chart: Total Cost Contribution by Supplier
    total_cost_per_supplier = {f"Supplier {s + 1}": df[f"Supplier {s + 1} Cost"].sum() for s in suppliers}
    
    fig, ax = plt.subplots()
    ax.pie(total_cost_per_supplier.values(), labels=total_cost_per_supplier.keys(), autopct='%1.1f%%', startangle=50)
    ax.set_title("Total Cost Contribution")
    st.pyplot(fig)
    
    # Scatter Plot: Quality vs. Cost Trade-Off
    fig, ax = plt.subplots()
    
    for s in suppliers:
        ax.scatter(costs[s], quality_scores[s], label=f"Supplier {s + 1}", s=50)
    
    ax.set_xlabel("Cost per Unit")
    ax.set_ylabel("Quality Score")
    ax.set_title("Quality vs. Cost Trade-Off")
    ax.legend()
    st.pyplot(fig)

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
lead_times = {supplier: st.sidebar.number_input(f"Lead Time for Supplier {supplier+1}", min_value=1, max_value=100, value=2) for supplier in suppliers}
quality_scores = {supplier: st.sidebar.number_input(f"Quality Score for Supplier {supplier+1}", min_value=1, max_value=10, value=8) for supplier in suppliers}
capacities = {supplier: st.sidebar.number_input(f"Capacity for Supplier {supplier+1}", min_value=1, max_value=1000, value=100) for supplier in suppliers}
min_order = {supplier: st.sidebar.number_input(f"Minimum Order for Supplier {supplier+1}", min_value=10, max_value=100, value=10) for supplier in suppliers}
demand_range = st.sidebar.slider("Demand Range", min_value=10, max_value=1000, value=(20, 100))
random_demand = st.sidebar.checkbox("Generate Random Demand", value=True)

if random_demand:
    weekly_demand = [random.randint(demand_range[0], demand_range[1]) for _ in range(num_weeks)]
else:
    weekly_demand = [st.sidebar.number_input(f"Demand for Week {i+1}", min_value=1, value=100) for i in range(num_weeks)]

# Solve and Display Results
if st.button("Optimize"):
    detailed_results, model_result, total_cost = solve_supplier_selection_problem(num_weeks, w1, w2, w3, num_suppliers, suppliers, costs, lead_times, quality_scores, 
                                     capacities, min_order, weekly_demand)

    st.subheader("Optimization Results")
    if model_result=='Optimal':
        st.success("Status: Optimal")
      
        # Convert to DataFrame for better formatting
        df = pd.DataFrame(detailed_results)
        
        # Create a dictionary to format the Allocation and Cost columns
        format_dict = {}
        for s in suppliers:
            format_dict[f"Supplier {s + 1} Allocation"] = "{:.2f}"
            format_dict[f"Supplier {s + 1} Cost"] = "${:.2f}"
        
        # Display the Total Cost
        st.write(f"**Total Cost: ${total_cost:,.2f}**")
        
        # Display the DataFrame in Streamlit with the formatting
        st.write("Results in a Tabular Form:")
        if not df.empty:
            st.dataframe(df.style.format(format_dict))

            # Inside the Streamlit App (After Optimization Results)
            plot_supply_chain_graphs(df, suppliers, costs, quality_scores)
  
    else:
        st.warning("No feasible solution found!")
