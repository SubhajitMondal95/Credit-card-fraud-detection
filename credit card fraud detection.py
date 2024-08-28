from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import GibbsSampling
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import logging

# Suppress only the specific warning about probability values
logging.getLogger('pgmpy').setLevel(logging.ERROR)

# Define the Bayesian Network 
model = BayesianNetwork([('T', 'FP'), ('OD', 'OP'), ('FP', 'F'), ('OP', 'F')])

# Conditional Probability Tables (CPTs)
cpd_T = TabularCPD(variable='T', variable_card=2, values=[[0.05], [0.95]])  # P(Travel)
cpd_OD = TabularCPD(variable='OD', variable_card=2, values=[[0.7], [0.3]])  # P(OwnsDevice)

cpd_FP = TabularCPD(variable='FP', variable_card=2, values=[[0.88, 0.0001], [0.12, 0.9999]],  # P(ForeignPurchase | Travel)
                    evidence=['T'], evidence_card=[2])

cpd_OP = TabularCPD(variable='OP', variable_card=2, values=[[0.4, 0.05], [0.6, 0.95]],  # P(OnlinePurchase | OwnsDevice)
                    evidence=['OD'], evidence_card=[2])

cpd_F = TabularCPD(variable='F', variable_card=2,   # P(Fraud| Foreign Purchase, Online Purchase)
                   values=[[0.005, 0.2,  0.15, 0.25],
                           [0.995, 0.8, 0.85, 0.75]],
                   evidence=['FP', 'OP'],
                   evidence_card=[2, 2])

# Add nodes and edges to the model
model.add_cpds(cpd_T, cpd_OD, cpd_FP, cpd_OP, cpd_F)

# Check if the model is valid
if model.check_model():
    print("Network structure and CPDs are correctly defined. The probabilities in the columns sum to 1")

# Display CPDs
print("Showing all the CPDs one by one:")
for cpd in model.get_cpds():
    print(cpd)

print("Can also access them like this:")
c = model.get_cpds()
print(c[0])
print(model.get_cpds('F'))
print("Number of values F can take on. The cardinality of F is:", model.get_cardinality('F'))

# Plot the Bayesian Network
G = nx.DiGraph()
G.add_edges_from([('T', 'FP'), ('OD', 'OP'), ('FP', 'F'), ('OP', 'F')])

# Manually specify node positions
pos = {'T': (0.13, 0.81), 'OD': (0.4, 0.81), 'FP': (0.185, 0.484), 'OP': (0.38, 0.484), 'F': (0.28, 0.114)}

# Plot the graph
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_size=5000, node_color="lightgreen", font_size=10, arrows=True)

# Annotate nodes with CPT information
cpt_annotations = {'T': cpd_T, 'OD': cpd_OD, 'FP': cpd_FP, 'OP': cpd_OP, 'F': cpd_F}
pos = {'T': (0.13, 0.71), 'OD': (0.4, 0.71), 'FP': (0.15, 0.384), 'OP': (0.4, 0.384), 'F': (0.28, 0.22)}
nx.draw_networkx_labels(G, pos, labels=cpt_annotations, font_size=10, font_color='red', verticalalignment='bottom')

plt.show()

def print_probability(query, result):
    print(f"Query: {query}")
    print(f"Probability: {result:.4f}")

# --- Variable Elimination ---
def variable_elimination(query_variable, evidence={}):
    ve = VariableElimination(model)
    marginal = ve.query(variables=[query_variable], evidence=evidence)
    return marginal.values[0]

print("# --- Variable Elimination ---")
# (a) Prior probability of fraudulent transaction
query_variable = "F"
result = variable_elimination(query_variable)
print_probability("(a) Prior probability of fraudulent transaction Prior P(F)", result)

# (b) Probability of fraud given owning a device (OD=True)
evidence = {"OD": True}
result = variable_elimination(query_variable, evidence)
print_probability("(b) Probability of fraud given owning a device (OD=True) P(F | OD=True)", result)

# (c) Probability of fraud given traveling (T=True)
evidence = {"T": True}
result = variable_elimination(query_variable, evidence)
print_probability("(c) Probability of fraud given traveling (T=True) P(F | T=True)", result)

# -- Gibbs Sampling --
print("# -- Gibbs Sampling--")

# Initialize the Gibbs Sampling object with your Bayesian Network model
gibbs_sampler = GibbsSampling(model) 

# Query (a): What is the prior probability of a fraudulent transaction?
query_a = gibbs_sampler.sample(size=1000)  # Sample a large number of times
prior_prob_fraud = round(np.mean(query_a['F']), 4)   # Round the probability to 4 decimal places
print(f"(a) Prior probability of a fraudulent transaction: {prior_prob_fraud}")

# Query (b): Probability of fraud when customer owns a smartphone
query_b = gibbs_sampler.sample(size=1000)  # Sample without evidence first
query_b.loc[query_b['OD'] == 0, 'F'] = 0  # Set F=0 where OD=0
prob_fraud_given_phone = round(query_b['F'].mean(), 4)  # Round the probability to 4 decimal places
print(f"(b) Probability of fraud given that customer owns a smartphone: {prob_fraud_given_phone}")

# Query (c): Probability of fraud when customer is traveling (T=True)
query_c = gibbs_sampler.sample(size=1000)  # Sample without evidence first
query_c.loc[query_c['T'] == 0, 'F'] = 0  # Set F=0 where T=0
prob_fraud_given_travel = round(query_c['F'].mean(), 4)  # Round the probability to 4 decimal places
print(f"(c) Probability of fraud given that customer is traveling: {prob_fraud_given_travel}")

