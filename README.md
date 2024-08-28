# Credit-card-fraud-detection

n this code, a CPT is created for each variable in the network:
● T (Travel): This CPT specifies the probability of a customer traveling (0.05) or not traveling (0.95).
● OD (Owns Device): This CPT specifies the probability of a customer owning a device (0.7) or not owning a device (0.3).
● FP (Foreign Purchase): This CPT defines the probability of a foreign purchase given whether the customer traveled or not. (e.g., 0.88 probability of foreign purchase if traveled, 0.12 probability if not)
● OP (Online Purchase): This CPT defines the probability of an online purchase given whether the customer owns a device or not. (e.g., 0.4 probability of online purchase if owns device, 0.6 probability if not)
● F (Fraud): This CPT defines the probability of fraud considering both foreign purchase and online purchase. (e.g., 0.005 probability of fraud if no foreign or online purchase, 0.25 probability if both foreign and online purchase)


Variable Elimination:
This code implements a Variable Elimination algorithm to answer probabilistic queries in a Bayesian network. It works by iteratively eliminating irrelevant variables from the network and summing out their probabilities.
● Model Definition: The code defines the Bayesian network structure and CPTs for each variable using the pgmpy library.
● Query and Evidence: The code specifies the variable of interest (e.g., Fraud) and any known evidence (e.g., Owns Device = True).
● Variable Elimination Object: It creates a VariableElimination object from pgmpy to perform the computations.
● Marginalization: The object queries the network for the desired variable, considering the evidence. It eliminates irrelevant variables one by one, summing out their probabilities based on the CPTs.
● Result: Finally, it returns the marginal probability of the query variable after eliminating all irrelevant variables.

Gibbs Sampling:
This code implements a Gibbs Sampling algorithm to approximate probabilities in a Bayesian network. It works by iteratively sampling the state of each variable based on the current state of its neighbors.
● Model Definition: Similar to Variable Elimination, the code defines the Bayesian network structure and CPTs.
● Gibbs Sampling Object: It creates a GibbsSampling object from pgmpy to perform the sampling.
● Sample Generation: The object samples the state (True/False) of each variable in the network multiple times. It considers the current state of its neighbors in the network and the CPTs to determine the probability of each state.
● Averaging: After a sufficient number of samples, it calculates the average value for the variable of interest (e.g., the proportion of samples where Fraud is True). This provides an approximation of the true probability.
