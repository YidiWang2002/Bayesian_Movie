## Bayesian Machine Learning with Generative AI Applications
### Predicting Box Office Success: A Bayesian Approach Using Movie Metadata
Win 25 | The Matrix Masters 🏆

Authors: Ritai Na, Dazhou Wu, Yidi Wang, Lucia Liu, Yiyang Yao

### Abstract:
This project explores predicting box office success using a Bayesian approach with movie metadata. 
By integrating Bayesian Networks, Hierarchical Bayesian Regression, Hidden Markov Models, and MCMC sampling, we develop a robust probabilistic framework for revenue forecasting.
Our methodology provides credible intervals for revenue estimates, assisting investors and studios in optimizing financial decision-making.

# HMM with HBM for Movie Industry Analysis

## Overview
This project explores the application of **Hidden Markov Models (HMM)** and **Hierarchical Bayesian Modeling (HBM)** in analyzing trends in the movie industry. By leveraging machine learning and probabilistic modeling, we extract latent patterns from historical movie data.

## Installation
To set up the required environment, install the following dependencies:
```bash
pip install pymc numpy pandas arviz scikit-learn hmmlearn
```

## Dataset
The project utilizes `Movie Final Data.csv`, containing attributes such as:
- **Financial Metrics**: Budget, Gross Revenue, Opening Weekend Sales
- **Theatrical Performance**: Opening Theaters, Maximum Theaters, Weeks in Run
- **Release Information**: Year, Month, Day of the Week

## Methodology
### **1. Data Preprocessing**
- Handled missing values using **median imputation** and **KNN imputation**.
- Applied **log transformation** to normalize skewed financial data.
- Sorted movies chronologically based on their release dates.

### **2. Hidden Markov Model (HMM) Implementation**
- Selected key features and applied **StandardScaler** for normalization.
- Trained a **2-state Gaussian HMM** using `hmmlearn`.
- Extracted **state transition probabilities**, **state means**, and **covariances**.
- Predicted hidden states for movies, categorizing them into different revenue-performance groups.

#### **HMM Transition Matrix**
```plaintext
| From | To State 0 | To State 1 |
|------|-----------|-----------|
| **State 0** | 0.623 | 0.377 |
| **State 1** | 0.474 | 0.526 |
```

**Interpretation:**
- **State 0**: Represents movies with **higher revenue and longer theatrical runs**.
- **State 1**: Represents movies with **lower revenue and shorter runs**.

### **3. Hierarchical Bayesian Modeling (HBM) (Work in Progress)**
- The next step is to implement **Hierarchical Bayesian Modeling** using `PyMC`.
- The goal is to further model **uncertainties** in movie success and extract robust financial indicators.

## Results & Visualization
The following visualizations illustrate the effectiveness of our models:
1. **Hidden States Over Time**: A time series showing how movie states transition.
2. **State Feature Distributions**: Comparing movie attributes across hidden states.
3. **Predicted vs Actual Movie Performance**: Evaluating HMM’s predictive power.

_(To be included: Graphs and illustrations to enhance understanding.)_

## Usage
1. **Load and preprocess data** using `pandas`.
2. **Train the HMM model** and analyze state transitions.
3. **(Upcoming) Implement Bayesian modeling** for deeper analysis.
4. **Interpret movie success patterns** and gain industry insights.

## Next Steps
- Improve Bayesian modeling to enhance revenue prediction accuracy.
- Incorporate visualization tools to better interpret HMM transitions.
- Extend the model to predict **future movie success probabilities**.

## Contact
For inquiries, feel free to reach out or contribute to the project!

---

This README is designed for **presentation purposes**, providing a structured overview of the methodology, findings, and next steps.

