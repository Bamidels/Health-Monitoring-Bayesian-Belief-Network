#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
import pickle
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Define the structure of the Bayesian Network
model = BayesianNetwork([
    ("HealthCondition", "HeartRate"),
    ("Exercise", "HeartRate"),
    ("Exercise", "EnergyLevel"),
    ("HeartRate", "BreathingRate"),
    ("EnergyLevel", "BreathingRate"),
])

# Step 2: Define the parameters (CPDs) of the Bayesian Network
cpd_HealthCondition = TabularCPD(variable='HealthCondition', variable_card=2, values=[[0.8], [0.2]])
cpd_Exercise = TabularCPD(variable='Exercise', variable_card=2, values=[[0.7], [0.3]])
cpd_HeartRate = TabularCPD(variable='HeartRate', variable_card=2, values=[[0.8, 0.9, 0.7, 0.1], [0.2, 0.1, 0.3, 0.9]],
                           evidence=['HealthCondition', 'Exercise'], evidence_card=[2, 2])
cpd_EnergyLevel = TabularCPD(variable='EnergyLevel', variable_card=2, values=[[0.8, 0.3], [0.2, 0.7]],
                             evidence=['Exercise'], evidence_card=[2])
cpd_BreathingRate = TabularCPD(variable='BreathingRate', variable_card=2, values=[[0.8, 0.4, 0.6, 0.2], [0.2, 0.6, 0.4, 0.8]],
                               evidence=['HeartRate', 'EnergyLevel'], evidence_card=[2, 2])

# Step 3: Add the parameters to the model
model.add_cpds(cpd_HealthCondition, cpd_Exercise, cpd_HeartRate, cpd_EnergyLevel, cpd_BreathingRate)

# Step 4: Validate the model
assert model.check_model()

# Step 1: Store model
with open('health_bayesian_network_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Step 2: Load model
with open('health_bayesian_network_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Visualize the Bayesian Network with a fixed layout
G = nx.DiGraph()
G.add_edges_from(model.edges())

# here i will specify the node positions manually to avoid the random_state issue
pos = {
    'HealthCondition': (0, 0),
    'Exercise': (1, 0),
    'HeartRate': (0.5, -1),
    'EnergyLevel': (1.5, -1),
    'BreathingRate': (1, -2),
}

nx.draw_networkx(G, pos, with_labels=True, node_size=800, node_color='skyblue')
nx.draw_networkx_labels(G, pos)
plt.show()

# Variable elimination algorithm
from pgmpy.inference import VariableElimination

# Step 1: Initialize the Variable Elimination method
ve_infer = VariableElimination(model)

# (a) Query P(HeartRate|Exercise=T, EnergyLevel=T)
result_a = ve_infer.query(variables=['HeartRate'], evidence={"Exercise": 1, "EnergyLevel": 1})
print(result_a)


# In[2]:


# (a) Query P(HeartRate|Exercise=T, EnergyLevel=T)
result_a = ve_infer.query(variables=['HeartRate'], evidence={"Exercise": 1, "EnergyLevel": 1})
print("Query (a) - P(HeartRate|Exercise=T, EnergyLevel=T):\n", result_a)

# (b) Query P(HealthCondition|Exercise=T)
result_b = ve_infer.query(variables=['HealthCondition'], evidence={"Exercise": 1})
print("Query (b) - P(HealthCondition|Exercise=T):\n", result_b)

# (c) Query P(HeartRate|HealthCondition=T, Exercise=T)
result_c = ve_infer.query(variables=['HeartRate'], evidence={"HealthCondition": 1, "Exercise": 1})
print("Query (c) - P(HeartRate|HealthCondition=T, Exercise=T):\n", result_c)

# (d) Query P(EnergyLevel|Exercise=T, BreathingRate=T)
result_d = ve_infer.query(variables=['EnergyLevel'], evidence={"Exercise": 1, "BreathingRate": 1})
print("Query (d) - P(EnergyLevel|Exercise=T, BreathingRate=T):\n", result_d)

# (e) Query P(BreathingRate|HeartRate=T, EnergyLevel=T)
result_e = ve_infer.query(variables=['BreathingRate'], evidence={"HeartRate": 1, "EnergyLevel": 1})
print("Query (e) - P(BreathingRate|HeartRate=T, EnergyLevel=T):\n", result_e)

# (f) Query P(HealthCondition|HeartRate=T, BreathingRate=T)
result_f = ve_infer.query(variables=['HealthCondition'], evidence={"HeartRate": 1, "BreathingRate": 1})
print("Query (f) - P(HealthCondition|HeartRate=T, BreathingRate=T):\n", result_f)


# In[3]:


from tabulate import tabulate

# Define the queries and their corresponding results
queries = [
    ("Query (a) - P(HeartRate|Exercise=T, EnergyLevel=T)", result_a),
    ("Query (b) - P(HealthCondition|Exercise=T)", result_b),
    ("Query (c) - P(HeartRate|HealthCondition=T, Exercise=T)", result_c),
    ("Query (d) - P(EnergyLevel|Exercise=T, BreathingRate=T)", result_d),
    ("Query (e) - P(BreathingRate|HeartRate=T, EnergyLevel=T)", result_e),
    ("Query (f) - P(HealthCondition|HeartRate=T, BreathingRate=T)", result_f),
]

# Create a table of the queries and results
table = tabulate([(query, str(result)) for query, result in queries], headers=["Query", "Result"], tablefmt="grid")

# Print the table
print(table)


# In[ ]:




