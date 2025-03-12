# Bayesian Machine Learning with Generative AI Applications
## Predicting Box Office Success: A Bayesian Approach Using Movie Metadata
Win 25 | The Matrix Masters 🏆

Authors: Ritai Na, Dazhou Wu, Yidi Wang, Lucia Liu, Yiyang Yao

## **Abstract**
This project explores predicting box office success using a Bayesian approach with movie metadata. 
By integrating Bayesian Networks, Hierarchical Bayesian Regression, Hidden Markov Models, and MCMC sampling, we develop a robust probabilistic framework for revenue forecasting.
Our methodology provides credible intervals for revenue estimates, assisting investors and studios in optimizing financial decision-making.

## Table of Contents
- [Overview](#Overview)
- [Data Processing and EDA](#Data-Processing-and-EDA)
- [Bayesian Network](#Bayesian-Network)
- [Hidden Markov Model](#Hidden-Markov-Model)
- [Hierarchical Bayesian Model with HMM](#Hierarchical-bayesian-model-with-HMM)
- [MCMC Sampling](#MCMC-Sampling)
- [Future Work](#future-work)

## **Overview**
This project explores the application of **Hidden Markov Models (HMM)** and **Hierarchical Bayesian Modeling (HBM)** in analyzing trends in the movie industry. By leveraging machine learning and probabilistic modeling, we extract latent patterns from historical movie data.

## **Data Processing and EDA**

## **Bayesian Network**

## **Hierarchical Bayesian Modeling (HBM)**


## Hidden Markov Model 

This model outlines the use of a Gaussian Hidden Markov Model (HMM) to capture latent market dynamics in movie box office data. The HMM model should be helpful combing with the Hierarchical Bayesian Model. 

### 1. HMM Model Definition
A **Hidden Markov Model (HMM)** is a statistical model for time series data where the system is assumed to be a Markov process with unobserved (hidden) states. 
In our application:
- **Hidden States** represent the underlying market conditions (e.g., "high revenue" vs. "low revenue" periods) that are not directly observed.
- **Emission Probabilities** assume that the observed data (e.g., log-transformed box office numbers) are generated from a probability distribution (in our case, a Gaussian) specific to each hidden state.
- **Transition Matrix** describes the probabilities of transitioning from one hidden state to another between successive movies (or time periods). This matrix is key in understanding the dynamics of market state changes.


### 2. Feature Engineering for HMM
Effective feature engineering is crucial for accurately capturing the latent dynamics in movie box office data. Our approach involves processing various types of features, but it is important to note that for the HMM model, we only use numerical variables since there may be
- **Temporal Ordering**: Numerical variables such as log-transformed production budget, domestic gross, worldwide gross, opening weekend, max theaters, and weeks run naturally have a time order when arranged sequentially by movie release timing. This temporal ordering is essential for HMMs to capture market dynamics.
```python
log_transform_cols = [
    'Production Budget (USD)', 'Domestic Gross (USD)', 'Worldwide Gross (USD)',
    'Opening Weekend (USD)', 'Max Theaters', 'Weeks Run'
]
```
```python
df_sorted = df.sort_values(by=['Release_Year', 'Release_Month', 'Release_DayOfWeek'])
```

- **Model Compatibility**: Gaussian HMMs assume that observations are continuous and approximately normally distributed within each hidden state. 
- **Avoiding High-Dimensional Noise**: Textual data requires additional NLP processing to convert them into numerical form. However, text data typically lack inherent sequential order that reflects market dynamics.

### 3. Model Building
```python
X = df_features.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_2_states = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
model_2_states.fit(X_scaled)
```


### 4. Model Result
![HMM Transition Matrix](./images/hmm_tran_matrix.png)
This matrix tells us that if the market is in state 0 (e.g., a high revenue state), there is a 62.3% chance that it will remain in state 0 and a 37.7% chance of switching to state 1 (e.g., a lower revenue state) in the next period.




## **Hierarchical Bayesian Model with HMM**

## Usage
1. **Load and preprocess data** using `pandas`.
2. **Train the HMM model** and analyze state transitions.
3. **(Upcoming) Implement Bayesian modeling** for deeper analysis.
4. **Interpret movie success patterns** and gain industry insights.


  
## MCMC sampling


## Results & Visualization
The following visualizations illustrate the effectiveness of our models:
1. **Hidden States Over Time**: A time series showing how movie states transition.
2. **State Feature Distributions**: Comparing movie attributes across hidden states.
3. **Predicted vs Actual Movie Performance**: Evaluating HMM’s predictive power.

_(To be included: Graphs and illustrations to enhance understanding.)_

## Future Work
### Next Steps
- Improve Bayesian modeling to enhance revenue prediction accuracy.
- Incorporate visualization tools to better interpret HMM transitions.
- Extend the model to predict **future movie success probabilities**.







## Contact
For inquiries, feel free to reach out or contribute to the project!
