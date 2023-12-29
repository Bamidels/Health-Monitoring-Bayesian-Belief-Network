# Health-Monitoring-Bayesian-Belief-Network
Modeling a health system with BBN to infer health from exercise, heart rate, energy, and breathing data.
Objective
To design and implement a BBN that encapsulates the complexities of a health monitoring system. The network incorporates multiple random variables, each representing a critical aspect of health. The project involves:

Clearly defining a health-related problem.
Representing the problem through a BBN.
Constructing a directed acyclic graph (DAG) for the BBN structure.
Estimating local conditional probabilities.
Encoding the network in a Python script.
Formulating and executing probabilistic queries to infer health conditions.
Problem Statement
We address the challenge of modeling a health monitoring system using a BBN. The system evaluates an individual's health condition based on observable data like exercise frequency, heart rate, energy level, and breathing rate.

Bayesian Belief Network Structure
Random Variables
HealthCondition (HC): The general health condition of the individual, either 'Good' or 'Poor'.
Exercise (Ex): Indicates if the individual has engaged in exercise, with possible states 'Yes' or 'No'.
HeartRate (HR): Categorized as 'Normal' or 'Elevated', affected by both exercise and overall health.
EnergyLevel (EL): The individual's energy level, either 'High' or 'Low', influenced by exercise.
BreathingRate (BR): The rate of breathing, 'Normal' or 'Elevated', dependent on heart rate and energy level.
Dependencies
The network captures the following dependencies:

Health condition potentially influenced by other non-modeled factors.
Exercise affecting heart rate and energy level.
Heart rate depending on health condition and exercise.
Energy level influenced by exercise.
Breathing rate dependent on heart rate and energy level.
Directed Acyclic Graph (DAG)
The DAG representing the BBN is included in the repository to illustrate the variable dependencies visually.

Probability Estimates
The project specifies the conditional probabilities for each variable, grounded in the assumption that exercise and overall health are significant determinants of the other variables. These probabilities are detailed in the repository.

Implementation
The implementation of the BBN is carried out in a Python script named MyBBN.py. This script encodes the network, including definitions for the nodes, edges, and conditional probability tables.

Probabilistic Queries and Analysis
The script executes six distinct probabilistic queries, each querying a different aspect of the health monitoring system. The results are captured and analyzed, providing a coherent interpretation of how each variable contributes to the health assessment.

Summary and Insights
The project demonstrates the utility of BBNs in modeling complex systems. Through this model, we gain valuable insights into how regular exercise and an individual's baseline health condition can influence physiological indicators such as heart rate, energy level, and breathing rate.
