# White-Box-vs-Black-Box-Models-in-Credit-Risk-Prediction
comparison of logistic regression and machine learning methods using the Give Me Some Credit dataset

#  Credit Risk Scoring & Model Benchmarking

This project implements a credit scoring pipeline in **R** to predict the **Probability of Default (PD)**. It benchmarks traditional **Logistic Regression** against **Regularized Regression (LASSO)** and **Decision Trees**, providing a framework for comparing interpretability vs. predictive power.

##  Table of Contents
- [Overview](#-overview)
- [Methodology](#-methodology)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Key Results](#-key-results)
- [Visualizations](#-visualizations)

##  Overview

The script (`credit_scoring.R`) performs an end-to-end analysis:
1.  **Data Ingestion:** Loads the "Give Me Some Credit" training dataset.
2.  **Preprocessing:** Handles missing values (`NA`), filters outliers, and creates a balanced subsample.
3.  **Modeling:** Trains three distinct classifiers:
    * Logistic Regression (GLM)
    * LASSO (L1 Regularization)
    * Decision Tree (CART)
4.  **Evaluation:** Compares performance using **ROC Curves** and **AUC** metrics.

##  Methodology

### 1. Data Cleaning
To ensure model stability, specific outliers are removed prior to training:
* **Utilization Check:** Rows where `RevolvingUtilizationOfUnsecuredLines > 10` are dropped.
* **Debt Burden:** Rows where `DebtRatio > 5000` are dropped.
* **Subsampling:** A random seed (`123`) is used to select 20,000 records for rapid prototyping.

### 2. Modeling Strategy

#### A. Logistic Regression
The industry standard for scorecards. We check for multicollinearity using the **Variance Inflation Factor (VIF)**.

#### B. LASSO Regression (glmnet)
Used for automated feature selection. The objective function minimizes the negative log-likelihood with an L1 penalty:

$$\min_{\beta} \left\{ - \frac{1}{N} \text{LogLikelihood}(\beta) + \lambda \| \beta \|_1 \right\}$$

* **Optimization:** Uses `cv.glmnet` to find the optimal $\lambda$ (lambda) that maximizes AUC.

#### C. Decision Tree (CART)
A non-parametric approach to visualize risk segments.
* **Hyperparameters:** `cp = 0.001` (Complexity Parameter) to control tree depth.
