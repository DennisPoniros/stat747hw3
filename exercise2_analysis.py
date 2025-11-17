#!/usr/bin/env python3
"""
Exercise 2: Applied Regression Learning
Comprehensive comparison of SVR vs Tree-based methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("EXERCISE 2: APPLIED REGRESSION LEARNING")
print("Comparing Tree-based Methods with Support Vector Regression")
print("=" * 80)

# ==============================================================================
# PART A: DATA EXPLORATION AND PREPARATION (15 points)
# ==============================================================================

print("\n" + "=" * 80)
print("PART A: DATA EXPLORATION AND PREPARATION")
print("=" * 80)

# 1. Data Understanding (5 points)
print("\n1. Data Understanding")
print("-" * 40)

# Load data (using relative path for cross-platform compatibility)
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()
data_path = os.path.join(script_dir, 'datafls.csv')
df = pd.read_csv(data_path)

# Create output directory for plots
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
print(f"Dataset shape: {df.shape}")
print(f"Number of features: {df.shape[1] - 1}")
print(f"Number of observations: {df.shape[0]}")

# Summary statistics
print("\nSummary Statistics for Response Variable (y):")
print(df['y'].describe())

print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check data types
print("\nData types:")
print(df.dtypes.value_counts())

# b) Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Distribution of response variable
axes[0, 0].hist(df['y'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution of Response Variable (y)')
axes[0, 0].set_xlabel('y')
axes[0, 0].set_ylabel('Frequency')

# Correlation heatmap for top correlated features
correlations = df.corr()['y'].abs().sort_values(ascending=False)[1:11]
top_features = correlations.index.tolist()
corr_matrix = df[['y'] + top_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[0, 1])
axes[0, 1].set_title('Top 10 Correlated Features with y')

# Scatter plots for top 4 predictors
for idx, (i, feature) in enumerate(zip([2, 0, 1, 2], top_features[:4])):
    row = 0 if idx < 2 else 1
    col = idx % 2 + 1 if row == 0 else idx % 2
    if row == 1 and col == 0:
        col = 0
    axes[row, col].scatter(df[feature], df['y'], alpha=0.5)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('y')
    axes[row, col].set_title(f'y vs {feature}')
    
    # Add regression line
    z = np.polyfit(df[feature], df['y'], 1)
    p = np.poly1d(z)
    axes[row, col].plot(df[feature], p(df[feature]), "r--", alpha=0.8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'exploration.png'), dpi=100, bbox_inches='tight')
plt.show()

# c) Check for missing values and outliers
print("\nc) Missing Values and Outliers:")
print("-" * 40)
print("Missing values per column:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values found!")
else:
    print(missing[missing > 0])

# Detect outliers using IQR method for response variable
Q1 = df['y'].quantile(0.25)
Q3 = df['y'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['y'] < Q1 - 1.5 * IQR) | (df['y'] > Q3 + 1.5 * IQR)]
print(f"\nOutliers in response variable (using IQR method): {len(outliers)} observations")
print(f"Percentage of outliers: {len(outliers)/len(df)*100:.2f}%")

# ==============================================================================
# 2. Feature Engineering (5 points)
# ==============================================================================

print("\n2. Feature Engineering")
print("-" * 40)

# Prepare features and target
X = df.drop('y', axis=1)
y = df['y']

# a) Create interaction terms for top correlated features
print("Creating interaction terms...")
top_3_features = correlations.index[:3].tolist()
for i in range(len(top_3_features)):
    for j in range(i+1, len(top_3_features)):
        feat1, feat2 = top_3_features[i], top_3_features[j]
        X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
        print(f"  Created interaction: {feat1} × {feat2}")

# Create polynomial features for top feature
top_feature = correlations.index[0]
X[f'{top_feature}_squared'] = X[top_feature] ** 2
print(f"  Created polynomial: {top_feature}²")

print(f"\nFeature matrix shape after engineering: {X.shape}")

# b) Standardize features for SVR
print("\nb) Standardizing features for SVR:")
print("  Why standardization is crucial for SVR:")
print("    - SVR uses distance-based kernels (RBF, polynomial)")
print("    - Features with larger scales dominate distance calculations")
print("    - Regularization parameter C applies uniformly across dimensions")
print("    - Kernel parameters (γ for RBF) are scale-sensitive")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# c) Split data
print("\nc) Train-Validation-Test Split:")
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42  # This gives us 15% validation
)

print(f"  Training set: {X_train.shape[0]} samples (70%)")
print(f"  Validation set: {X_val.shape[0]} samples (15%)")
print(f"  Test set: {X_test.shape[0]} samples (15%)")

# ==============================================================================
# 3. Baseline Models (5 points)
# ==============================================================================

print("\n3. Baseline Models")
print("-" * 40)

# a) Linear Regression baseline (using same scaled data as SVR)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred_val = lr.predict(X_val)
lr_pred_test = lr.predict(X_test)
lr_rmse_val = np.sqrt(mean_squared_error(y_val, lr_pred_val))
lr_r2_val = r2_score(y_val, lr_pred_val)
lr_rmse_test = np.sqrt(mean_squared_error(y_test, lr_pred_test))
lr_r2_test = r2_score(y_test, lr_pred_test)

print("a) Linear Regression Baseline:")
print(f"  Validation RMSE: {lr_rmse_val:.6f}")
print(f"  Validation R²: {lr_r2_val:.6f}")
print(f"  Test RMSE: {lr_rmse_test:.6f}")
print(f"  Test R²: {lr_r2_test:.6f}")

# b) Single Decision Tree (no pruning)
dt_unpruned = DecisionTreeRegressor(random_state=42)
dt_unpruned.fit(X_train, y_train)
dt_pred_val = dt_unpruned.predict(X_val)
dt_rmse_val = np.sqrt(mean_squared_error(y_val, dt_pred_val))
dt_r2_val = r2_score(y_val, dt_pred_val)

print("\nb) Single Decision Tree (unpruned):")
print(f"  Tree depth: {dt_unpruned.get_depth()}")
print(f"  Number of leaves: {dt_unpruned.get_n_leaves()}")
print(f"  Validation RMSE: {dt_rmse_val:.6f}")
print(f"  Validation R²: {dt_r2_val:.6f}")

# ==============================================================================
# PART B: SUPPORT VECTOR REGRESSION (15 points)
# ==============================================================================

print("\n" + "=" * 80)
print("PART B: SUPPORT VECTOR REGRESSION")
print("=" * 80)

# 1. Model Training (5 points)
print("\n1. Model Training with Different Kernels")
print("-" * 40)

# Define parameter grids for each kernel
param_grids = {
    'linear': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'epsilon': [0.001, 0.01, 0.1, 0.2]
    },
    'poly': {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2],
        'degree': [2, 3, 4],
        'coef0': [0, 1]
    },
    'rbf': {
        'C': [0.1, 1, 10, 100, 1000],
        'epsilon': [0.001, 0.01, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
}

# Store results
svr_models = {}
svr_results = {}

# Train and tune each kernel
for kernel_name, params in param_grids.items():
    print(f"\nTraining SVR with {kernel_name} kernel:")
    
    # Create base model
    if kernel_name == 'poly':
        svr = SVR(kernel=kernel_name)
    else:
        svr = SVR(kernel=kernel_name)
    
    # Grid search with cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        svr, params, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
    )
    
    # Fit model
    grid_search.fit(X_train, y_train)
    
    # Store best model
    svr_models[kernel_name] = grid_search.best_estimator_
    
    # Get validation predictions
    val_pred = grid_search.best_estimator_.predict(X_val)
    
    # Calculate metrics
    svr_results[kernel_name] = {
        'best_params': grid_search.best_params_,
        'cv_score': -grid_search.best_score_,
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'val_mae': mean_absolute_error(y_val, val_pred),
        'val_r2': r2_score(y_val, val_pred)
    }
    
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Cross-validation RMSE: {np.sqrt(-grid_search.best_score_):.6f}")
    print(f"  Validation RMSE: {svr_results[kernel_name]['val_rmse']:.6f}")
    
    # Count support vectors
    if hasattr(grid_search.best_estimator_, 'support_vectors_'):
        n_sv = len(grid_search.best_estimator_.support_)
        print(f"  Number of support vectors: {n_sv} ({n_sv/len(X_train)*100:.1f}% of training data)")

# Plot validation performance vs hyperparameters for RBF kernel
print("\nDetailed RBF hyperparameter analysis:")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Test different C values with best gamma
C_values = [0.01, 0.1, 1, 10, 100, 1000]
best_gamma = svr_models['rbf'].gamma
scores_C = []

for C in C_values:
    model = SVR(kernel='rbf', C=C, gamma=best_gamma)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='neg_mean_squared_error')
    scores_C.append(np.sqrt(-cv_scores.mean()))

axes[0].semilogx(C_values, scores_C, 'b-o')
axes[0].set_xlabel('C (Regularization Parameter)')
axes[0].set_ylabel('CV RMSE')
axes[0].set_title('RBF SVR: Effect of C Parameter')
axes[0].grid(True, alpha=0.3)

# Test different gamma values with best C
gamma_values = [0.001, 0.01, 0.1, 1, 10]
best_C = svr_models['rbf'].C
scores_gamma = []

for gamma in gamma_values:
    model = SVR(kernel='rbf', C=best_C, gamma=gamma)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='neg_mean_squared_error')
    scores_gamma.append(np.sqrt(-cv_scores.mean()))

axes[1].semilogx(gamma_values, scores_gamma, 'r-o')
axes[1].set_xlabel('γ (Gamma Parameter)')
axes[1].set_ylabel('CV RMSE')
axes[1].set_title('RBF SVR: Effect of Gamma Parameter')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'svr_hyperparameters.png'), dpi=100, bbox_inches='tight')
plt.show()

# ==============================================================================
# 2. Model Evaluation (5 points)
# ==============================================================================

print("\n2. Model Evaluation")
print("-" * 40)

# Compare test performance
print("\na) Test Performance Comparison:")
print("-" * 60)
print(f"{'Kernel':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 60)

for kernel_name, model in svr_models.items():
    test_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    mae = mean_absolute_error(y_test, test_pred)
    r2 = r2_score(y_test, test_pred)
    
    print(f"{kernel_name:<10} {rmse:<10.6f} {mae:<10.6f} {r2:<10.6f}")
    
    # Update results with test metrics
    svr_results[kernel_name]['test_rmse'] = rmse
    svr_results[kernel_name]['test_mae'] = mae
    svr_results[kernel_name]['test_r2'] = r2

# b) Analyze residuals
print("\nb) Residual Analysis:")
best_kernel = min(svr_results.keys(), key=lambda k: svr_results[k]['test_rmse'])
best_svr = svr_models[best_kernel]
test_pred = best_svr.predict(X_test)
residuals = y_test - test_pred

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Residual plot
axes[0].scatter(test_pred, residuals, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title(f'Residual Plot ({best_kernel} kernel)')

# Histogram of residuals
axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'svr_residuals.png'), dpi=100, bbox_inches='tight')
plt.show()

print(f"  Mean residual: {residuals.mean():.6f}")
print(f"  Std residual: {residuals.std():.6f}")
print(f"  Skewness: {stats.skew(residuals):.6f}")
print(f"  Kurtosis: {stats.kurtosis(residuals):.6f}")

# c) Computational requirements
print("\nc) Computational Requirements:")
import time

for kernel_name in ['linear', 'poly', 'rbf']:
    model = SVR(kernel=kernel_name)
    
    # Training time
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Prediction time
    start = time.time()
    _ = model.predict(X_test)
    pred_time = time.time() - start
    
    print(f"\n{kernel_name} kernel:")
    print(f"  Training time: {train_time:.4f} seconds")
    print(f"  Prediction time: {pred_time:.4f} seconds")
    print(f"  Complexity: O(n²·m) for training, O(n_sv·m) for prediction")
    if kernel_name == 'linear':
        print(f"  Note: Can be reduced to O(n·m) with linear SVM solvers")

# ==============================================================================
# 3. Interpretation (5 points)
# ==============================================================================

print("\n3. SVR Interpretation")
print("-" * 40)

# a) Support vectors analysis
for kernel_name, model in svr_models.items():
    if hasattr(model, 'support_'):
        n_sv = len(model.support_)
        print(f"\n{kernel_name} kernel:")
        print(f"  Support vectors: {n_sv} out of {len(X_train)} training samples")
        print(f"  Percentage: {n_sv/len(X_train)*100:.1f}%")
        
        # b) Complexity interpretation
        if n_sv < len(X_train) * 0.3:
            print(f"  → Relatively simple decision boundary")
        elif n_sv < len(X_train) * 0.7:
            print(f"  → Moderate complexity")
        else:
            print(f"  → High complexity (possible overfitting)")

# c) Challenges in interpreting SVR
print("\nc) Challenges in Interpreting SVR Models:")
print("  1. Non-linear kernels create implicit feature spaces")
print("  2. Decision function is combination of support vectors")
print("  3. No direct feature importance measures")
print("  4. Kernel parameters affect model globally")
print("  5. Black-box nature makes debugging difficult")

# ==============================================================================
# PART C: TREE-BASED REGRESSION (10 points)
# ==============================================================================

print("\n" + "=" * 80)
print("PART C: TREE-BASED REGRESSION")
print("=" * 80)

# 1. Single Trees (4 points)
print("\n1. Single Decision Trees")
print("-" * 40)

# a) Train pruned decision tree using cross-validation
param_grid_dt = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}

dt = DecisionTreeRegressor(random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search_dt = GridSearchCV(dt, param_grid_dt, cv=cv, 
                              scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)

best_dt = grid_search_dt.best_estimator_
print(f"Best parameters: {grid_search_dt.best_params_}")
print(f"Tree depth: {best_dt.get_depth()}")
print(f"Number of leaves: {best_dt.get_n_leaves()}")

# b) Visualize tree structure
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(best_dt, feature_names=X.columns, filled=True, 
          rounded=True, ax=ax, fontsize=8, max_depth=3)
plt.title('Decision Tree Structure (Top 3 Levels)')
plt.savefig(os.path.join(output_dir, 'tree_structure.png'), dpi=100, bbox_inches='tight')
plt.show()

# c) Compute feature importance
dt_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_dt.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features (Decision Tree):")
print(dt_importance.head(10).to_string(index=False))

# ==============================================================================
# 2. Ensemble Methods (3 points)
# ==============================================================================

print("\n2. Ensemble Methods")
print("-" * 40)

# a) Random Forest
print("\na) Random Forest:")
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=cv, 
                              scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

best_rf = grid_search_rf.best_estimator_
print(f"Best parameters: {grid_search_rf.best_params_}")

# b) Gradient Boosting
print("\nb) Gradient Boosting:")
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

gb = GradientBoostingRegressor(random_state=42)
grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=cv,
                              scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_gb.fit(X_train, y_train)

best_gb = grid_search_gb.best_estimator_
print(f"Best parameters: {grid_search_gb.best_params_}")

# c) Compare out-of-bag error with validation error
print("\nc) Out-of-Bag Error vs Validation Error:")

# Random Forest OOB
rf_oob = RandomForestRegressor(**best_rf.get_params(), oob_score=True)
rf_oob.fit(X_train, y_train)
oob_score = rf_oob.oob_score_
val_pred_rf = best_rf.predict(X_val)
val_score_rf = r2_score(y_val, val_pred_rf)

print(f"Random Forest:")
print(f"  OOB R² score: {oob_score:.6f}")
print(f"  Validation R² score: {val_score_rf:.6f}")
print(f"  Difference: {abs(oob_score - val_score_rf):.6f}")

# ==============================================================================
# 3. Interpretation (3 points)
# ==============================================================================

print("\n3. Tree-Based Model Interpretation")
print("-" * 40)

# a) Partial dependence plots
print("\na) Creating partial dependence plots for top features...")

# Get top 3 features from Random Forest
rf_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

top_features_rf = rf_importance.head(3)['feature'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, feature in enumerate(top_features_rf):
    # Find feature index
    feat_idx = list(X.columns).index(feature)
    
    # Manual partial dependence (simplified)
    X_temp = X_train.copy()
    feature_values = np.linspace(X_train.iloc[:, feat_idx].min(),
                                X_train.iloc[:, feat_idx].max(), 50)
    pd_values = []
    
    for val in feature_values:
        X_temp.iloc[:, feat_idx] = val
        pd_values.append(best_rf.predict(X_temp).mean())
    
    axes[idx].plot(feature_values, pd_values, linewidth=2)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Partial Dependence')
    axes[idx].set_title(f'PD Plot: {feature}')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Partial Dependence Plots - Random Forest')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'partial_dependence.png'), dpi=100, bbox_inches='tight')
plt.show()

# b) Compare feature importance across methods
print("\nb) Feature Importance Comparison:")

# Collect importance for top 10 features
top_features_union = set(dt_importance.head(10)['feature'].tolist() + 
                        rf_importance.head(10)['feature'].tolist())

importance_comparison = pd.DataFrame(index=list(top_features_union))
importance_comparison['Decision Tree'] = best_dt.feature_importances_[
    [list(X.columns).index(f) for f in importance_comparison.index]
]
importance_comparison['Random Forest'] = best_rf.feature_importances_[
    [list(X.columns).index(f) for f in importance_comparison.index]
]
importance_comparison['Gradient Boosting'] = best_gb.feature_importances_[
    [list(X.columns).index(f) for f in importance_comparison.index]
]

print("\nTop Feature Importances Across Methods:")
print(importance_comparison.sort_values('Random Forest', ascending=False).head(10))

# c) Trade-off discussion
print("\nc) Interpretability vs Performance Trade-off:")
print("  Single Tree: High interpretability, lower performance")
print("  Random Forest: Moderate interpretability (feature importance), better performance")
print("  Gradient Boosting: Lower interpretability, often best performance")
print("  SVR: Lowest interpretability, performance depends on data structure")

# ==============================================================================
# PART D: COMPREHENSIVE COMPARISON (10 points)
# ==============================================================================

print("\n" + "=" * 80)
print("PART D: COMPREHENSIVE COMPARISON")
print("=" * 80)

# 1. Performance Metrics (3 points)
print("\n1. Performance Metrics")
print("-" * 40)

# Collect all models
models = {
    'Linear Regression': lr,
    'Decision Tree': best_dt,
    'Random Forest': best_rf,
    'Gradient Boosting': best_gb,
    'SVR (Linear)': svr_models['linear'],
    'SVR (Poly)': svr_models['poly'],
    'SVR (RBF)': svr_models['rbf']
}

# Calculate metrics for all models
results = []
for name, model in models.items():
    test_pred = model.predict(X_test)
    
    results.append({
        'Model': name,
        'RMSE': np.sqrt(mean_squared_error(y_test, test_pred)),
        'MAE': mean_absolute_error(y_test, test_pred),
        'R²': r2_score(y_test, test_pred)
    })

results_df = pd.DataFrame(results).sort_values('RMSE')

print("\na) Test Set Performance Comparison:")
print("=" * 60)
print(results_df.to_string(index=False))

# b) Statistical significance testing
print("\nb) Statistical Significance Testing (Paired t-test on CV folds):")

from scipy import stats

# Get CV predictions for best models
cv_models = {
    'Random Forest': best_rf,
    'Gradient Boosting': best_gb,
    'SVR (RBF)': svr_models['rbf']
}

cv_scores = {}
for name, model in cv_models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, 
                            scoring='neg_mean_squared_error')
    cv_scores[name] = np.sqrt(-scores)

# Perform pairwise t-tests
print("\nPaired t-test results (p-values):")
print("-" * 40)
for i, model1 in enumerate(cv_models.keys()):
    for model2 in list(cv_models.keys())[i+1:]:
        t_stat, p_value = stats.ttest_rel(cv_scores[model1], cv_scores[model2])
        print(f"{model1} vs {model2}: p = {p_value:.4f}")
        if p_value < 0.05:
            print(f"  → Significant difference at α=0.05")
        else:
            print(f"  → No significant difference")

# c) Region analysis
print("\nc) Performance in Different Feature Space Regions:")

# Analyze by response variable quartiles
y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
quartiles = np.percentile(y_test_array, [25, 50, 75])

regions = {
    'Low (Q1)': y_test_array <= quartiles[0],
    'Medium (Q2-Q3)': (y_test_array > quartiles[0]) & (y_test_array <= quartiles[2]),
    'High (Q4)': y_test_array > quartiles[2]
}

print("\nRMSE by Response Variable Region:")
print("-" * 60)
print(f"{'Model':<20} {'Low':<10} {'Medium':<10} {'High':<10}")
print("-" * 60)

for name, model in [('Random Forest', best_rf), ('SVR (RBF)', svr_models['rbf'])]:
    test_pred = model.predict(X_test)
    rmse_by_region = []
    
    for region_name, mask in regions.items():
        if mask.sum() > 0:
            rmse = np.sqrt(mean_squared_error(y_test_array[mask], test_pred[mask]))
            rmse_by_region.append(rmse)
    
    print(f"{name:<20} {rmse_by_region[0]:<10.4f} {rmse_by_region[1]:<10.4f} {rmse_by_region[2]:<10.4f}")

# ==============================================================================
# 2. Practical Considerations (3 points)
# ==============================================================================

print("\n2. Practical Considerations")
print("-" * 40)

# a) Training and prediction time
print("\na) Computational Performance:")
print("-" * 60)
print(f"{'Model':<20} {'Train Time (s)':<15} {'Predict Time (ms)':<15}")
print("-" * 60)

for name, model in models.items():
    # Training time
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Prediction time (average over 100 runs)
    times = []
    for _ in range(100):
        start = time.time()
        _ = model.predict(X_test)
        times.append((time.time() - start) * 1000)  # Convert to ms
    pred_time = np.mean(times)
    
    print(f"{name:<20} {train_time:<15.4f} {pred_time:<15.4f}")

# b) Robustness analysis
print("\nb) Robustness to Outliers and Noise:")

# Add noise to test set
noise_levels = [0, 0.1, 0.2, 0.3]
robustness_results = []

for noise_level in noise_levels:
    X_test_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
    
    for name in ['Random Forest', 'SVR (RBF)']:
        model = models[name]
        pred_noisy = model.predict(X_test_noisy)
        rmse_noisy = np.sqrt(mean_squared_error(y_test, pred_noisy))
        
        robustness_results.append({
            'Model': name,
            'Noise Level': noise_level,
            'RMSE': rmse_noisy
        })

robustness_df = pd.DataFrame(robustness_results).pivot(
    index='Noise Level', columns='Model', values='RMSE'
)
print("\nRMSE with Different Noise Levels:")
print(robustness_df.to_string())

# c) Hyperparameter sensitivity
print("\nc) Hyperparameter Sensitivity Analysis:")
print("  Random Forest: Relatively robust, main sensitivity to n_estimators and max_depth")
print("  Gradient Boosting: Sensitive to learning_rate and n_estimators combination")
print("  SVR (RBF): Very sensitive to C and gamma, requires careful tuning")
print("  Decision Tree: Highly sensitive to pruning parameters")

# ==============================================================================
# 3. Recommendations (4 points)
# ==============================================================================

print("\n3. Final Recommendations")
print("-" * 40)

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_rmse = results_df.iloc[0]['RMSE']
best_r2 = results_df.iloc[0]['R²']

print(f"\na) Recommended Method: {best_model_name}")
print(f"   Test RMSE: {best_rmse:.6f}")
print(f"   Test R²: {best_r2:.6f}")

print("\nb) Justification:")
print(f"   - Best predictive performance (lowest RMSE: {best_rmse:.6f})")

if 'Forest' in best_model_name:
    print("   - Good balance between accuracy and interpretability")
    print("   - Robust to outliers and noise")
    print("   - Provides feature importance measures")
    print("   - Reasonable training time")
    print("   - No need for feature scaling")
elif 'SVR' in best_model_name:
    print("   - Excellent generalization through margin maximization")
    print("   - Sparse solution (uses support vectors only)")
    print("   - Handles high-dimensional data well")
    print("   - Robust to overfitting with proper regularization")
elif 'Gradient' in best_model_name:
    print("   - State-of-the-art performance for tabular data")
    print("   - Captures complex non-linear patterns")
    print("   - Sequential error correction")
    print("   - Good feature importance measures")

print("\nc) Next Steps for Improvement:")
print("   1. Feature engineering: Create more domain-specific features")
print("   2. Ensemble methods: Combine predictions from multiple models")
print("   3. Advanced tuning: Bayesian optimization for hyperparameters")
print("   4. Feature selection: Remove irrelevant features to reduce noise")
print("   5. Cross-validation: Use nested CV for more robust evaluation")
print("   6. Data augmentation: If possible, collect more training samples")

# ==============================================================================
# Save final results
# ==============================================================================

# Save comprehensive results
results_df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("Results saved to model_comparison_results.csv")
print("=" * 80)

# Create final comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# RMSE comparison
axes[0, 0].barh(results_df['Model'], results_df['RMSE'])
axes[0, 0].set_xlabel('RMSE')
axes[0, 0].set_title('Test Set RMSE Comparison')
axes[0, 0].invert_yaxis()

# R² comparison
axes[0, 1].barh(results_df['Model'], results_df['R²'])
axes[0, 1].set_xlabel('R²')
axes[0, 1].set_title('Test Set R² Comparison')
axes[0, 1].invert_yaxis()

# Predictions vs Actual for best model
best_model = models[best_model_name]
test_pred = best_model.predict(X_test)
axes[1, 0].scatter(y_test, test_pred, alpha=0.5)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual')
axes[1, 0].set_ylabel('Predicted')
axes[1, 0].set_title(f'Predictions vs Actual ({best_model_name})')

# Feature importance (if available)
if 'Forest' in best_model_name or 'Tree' in best_model_name or 'Gradient' in best_model_name:
    if 'Forest' in best_model_name:
        importance = best_rf.feature_importances_
    elif 'Gradient' in best_model_name:
        importance = best_gb.feature_importances_
    else:
        importance = best_dt.feature_importances_
    
    top_n = 10
    indices = np.argsort(importance)[::-1][:top_n]
    axes[1, 1].barh(range(top_n), importance[indices])
    axes[1, 1].set_yticks(range(top_n))
    axes[1, 1].set_yticklabels([X.columns[i] for i in indices])
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title(f'Top {top_n} Feature Importance ({best_model_name})')
    axes[1, 1].invert_yaxis()
else:
    axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available for\nthis model type',
                    ha='center', va='center', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)

plt.suptitle('Comprehensive Model Comparison Results', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'final_comparison.png'), dpi=100, bbox_inches='tight')
plt.show()

print(f"\nAll visualizations saved to: {output_dir}")
print(f"Generated files:")
print(f"  - exploration.png")
print(f"  - svr_hyperparameters.png")
print(f"  - svr_residuals.png")
print(f"  - tree_structure.png")
print(f"  - partial_dependence.png")
print(f"  - final_comparison.png")
print(f"  - model_comparison_results.csv")
