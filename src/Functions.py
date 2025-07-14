"""Functions"""
# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import plotly.express as px
import os
from math import ceil
from datetime import datetime
from plotly.colors import qualitative
# Core Library Settings
pd.options.display.float_format = '{:.2f}'.format
sns.set()




from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, average_precision_score,
                           f1_score, fbeta_score, roc_curve, 
                           precision_recall_curve, make_scorer, auc)
from sklearn.pipeline import Pipeline  

# Utilities
import warnings
warnings.filterwarnings('ignore')
from collections import Counter

# Data Preprocessing & Feature Engineering
from sklearn.preprocessing import (
    LabelEncoder,
    RobustScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Feature Selection Methods
# Filter Methods
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
# Wrapper Methods
from sklearn.feature_selection import RFE
# Embedded Methods
from sklearn.linear_model import LassoCV 
from sklearn.feature_selection import SelectFromModel 

# Imbalanced Learning Techniques
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline # to avoid naming conflicts
from collections import Counter
from imblearn.under_sampling import EditedNearestNeighbours


# Machine Learning Models
# Linear Models
from sklearn.linear_model import LogisticRegression
# Support Vector Machines
from sklearn.svm import SVC
# Tree-based Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
# Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier


# Model Selection & Evaluation
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

# Model Interpretability
import shap
from catboost import Pool # Specific to CatBoost for data handling

# Statistics
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.signal import find_peaks
from sklearn.base import clone



###################### EXPLORATORY DATA ANALYSIS ########################################################################3333

# define a function to analyse cardinality
def categorical_summary(df):
    # get categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    # loop through each column and print its details
    for col in categorical_columns:
        print(f'#### {col}')
        print(f"Unique values count: {df[col].nunique()}")
        print(f"Unique values: {df[col].unique()}")
        print("\n")
        
# define a function to analyse numerical column details
def numerical_summary(df):
    # get numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # loop through each column and print its details
    for col in numerical_columns:
        print(f'#### {col}')
        print(f"Unique values count: {df[col].nunique()}")
        print(f"Minimum value: {df[col].min()}")
        print(f"Maximum value: {df[col].max()}")
        print(f"Mean: {df[col].mean()}")
        print(f"Median: {df[col].median()}")
        print(f"Standard Deviation: {df[col].std()}")
        print(f"Data type: {df[col].dtype}")
        print("\n")
        
# create function to check percentage of missing values over time
def missing_perc(data):
    missing_values = round((data.isnull().sum()/data.shape[0])*100,2)
    print(missing_values)
    
def duplicate_summary(df):
    """
    Analyzes full-row duplicates in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze for duplicate rows.
    
    Returns:
    --------
    None (prints the summary)
    """
    print("=== Duplicate Row Analysis ===")
    
    total_duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {total_duplicates}")
    
    if total_duplicates > 0:
        print("\nSample duplicate rows (all columns must match):")
        print(df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist()).head())
    else:
        print("No duplicate rows found (all rows are unique).")
    
    print("\n" + "="*40 + "\n")
    
def categorical_features_dimensionality(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    # collect summary data for all categorical columns
    summary_data = []
    for col in categorical_columns:
        summary_data.append({
            'Column Name': col,
            'Unique Count': df[col].nunique(),
            'Unique Values': df[col].unique()[:5],
            'Data Type': df[col].dtype
        })
    # convert int into a dataframe
    # for better display
    summary_df = pd.DataFrame(summary_data)
    print("\nCategorical Summary:\n")
    print(summary_df.to_markdown(index=False))
    
################################## CROSS VALIDATION ##################################################################################
# functions.py

def calculate_empc(y_true, y_pred, profit_matrix):
    """
    Computes Expected Misclassification Profit/Cost (EMPC)
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        profit_matrix (np.ndarray): Business-defined profit matrix
    
    Returns:
        float: EMPC value
    """
    """Calculate Expected Misclassification Profit/Cost"""
    cm = confusion_matrix(y_true, y_pred)
    total_profit = 0
    for i in range(len(profit_matrix)):
        for j in range(len(profit_matrix)):
            total_profit += cm[i,j] * profit_matrix[i,j]
    return total_profit / len(y_true)

def cross_validate_with_stratified(model_name, model, param_grid, X_train, y_train, 
                                 numerical_columns, categorical_columns, profit_matrix):
    """Enhanced cross-validation with feature selection and F2 optimization"""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=369)
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    scaler = RobustScaler()
    
    best_model_result = {
        'best_f1_score': -1,
        'best_f2_score': -1,
        'best_classification_report': '', 
        'best_params': None, 
        'best_selected_features': {},
        'best_confusion_matrix': None,
        'best_roc_auc': -1,
        'best_pr_auc': -1,
        'best_empc': -np.inf,
        'best_model': None,
        'best_threshold': 0.5,
        'feature_names': None
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        # Split and preprocess data
        X_train_fold, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_cv = y_train[train_idx], y_train[val_idx]
        
        # Encode and scale
        for col in categorical_columns:
            label_encoders[col].fit(X_train_fold[col])
            X_train_fold[col] = label_encoders[col].transform(X_train_fold[col])
            X_val_cv[col] = X_val_cv[col].apply(
                lambda x: label_encoders[col].transform([x])[0] 
                if x in label_encoders[col].classes_ else -1
            )
        
        X_train_fold[numerical_columns] = scaler.fit_transform(X_train_fold[numerical_columns])
        X_val_cv[numerical_columns] = scaler.transform(X_val_cv[numerical_columns])

        # Feature selection
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=369, class_weight='balanced')
        rf_selector.fit(X_train_fold[numerical_columns], y_train_fold)
        selected_numerical = [
            numerical_columns[i] for i, imp in enumerate(rf_selector.feature_importances_)
            if imp >= np.median(rf_selector.feature_importances_)
        ]
        
        chi2_selector = SelectKBest(chi2, k='all')
        chi2_selector.fit(X_train_fold[categorical_columns], y_train_fold)
        selected_categorical = [
            categorical_columns[i] for i, p_val in enumerate(chi2_selector.pvalues_)
            if p_val <= 0.005
        ]
        
        selected_features = selected_numerical + selected_categorical
        X_train_fold = X_train_fold[selected_features]
        X_val_cv = X_val_cv[selected_features]
        
        best_model_result['best_selected_features'][f'Fold {fold_idx+1}'] = {
            'numerical_features': selected_numerical,
            'categorical_features': selected_categorical
        }
        
        # Model training with SMOTE-ENN
        pipeline = Pipeline([
            ('sampling', SMOTEENN(random_state=369)),
            ('classifier', clone(model))
        ])
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid={'classifier__' + k: v for k,v in param_grid.items()},
            scoring=make_scorer(fbeta_score, beta=2),
            cv=3,
            n_jobs=-1
        )
        grid_search.fit(X_train_fold, y_train_fold)
        
        # Threshold optimization
        y_val_cv_proba = grid_search.best_estimator_.predict_proba(X_val_cv)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val_cv, y_val_cv_proba)
        optimal_threshold = thresholds[np.argmax([
            fbeta_score(y_val_cv, (y_val_cv_proba >= t).astype(int), beta=2) 
            for t in thresholds
        ])]
        
        # Update best results
        y_val_cv_pred = (y_val_cv_proba >= optimal_threshold).astype(int)
        current_f1 = f1_score(y_val_cv, y_val_cv_pred)
        
        if current_f1 > best_model_result['best_f1_score']:
            best_model_result.update({
                'best_f1_score': current_f1,
                'best_f2_score': fbeta_score(y_val_cv, y_val_cv_pred, beta=2),
                'best_classification_report': classification_report(y_val_cv, y_val_cv_pred),
                'best_params': grid_search.best_params_,
                'best_confusion_matrix': confusion_matrix(y_val_cv, y_val_cv_pred),
                'best_roc_auc': roc_auc_score(y_val_cv, y_val_cv_proba),
                'best_pr_auc': average_precision_score(y_val_cv, y_val_cv_proba),
                'best_empc': calculate_empc(y_val_cv, y_val_cv_pred, profit_matrix),
                'best_model': grid_search.best_estimator_,
                'best_threshold': optimal_threshold,
                'feature_names': selected_features
            })
    best_model_result['fitted_label_encoders'] = label_encoders
    best_model_result['fitted_scaler'] = scaler
    
    return best_model_result

def plot_model_comparisons(results, X_train, y_train, numerical_columns, categorical_columns, fitted_label_encoders):
    """Plot ROC and PR curves for all models"""
    # ROC Curve
    plt.figure(figsize=(10, 8))
    for model_name, result in results.items():
        model = result['best_model']
        X_val = X_train.copy()
        
        # Use pre-fitted encoders
        for col in categorical_columns:
            X_val[col] = fitted_label_encoders[col].transform(X_val[col])
        
        # Use scaler from results if needed
        X_val[numerical_columns] = result['fitted_scaler'].transform(X_val[numerical_columns])
        X_val = X_val[result['feature_names']]
        
        fpr, tpr, _ = roc_curve(y_train, model.predict_proba(X_val)[:, 1])
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc(fpr, tpr):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves')
    plt.legend()
    plt.show()
    
    # PR Curve
    plt.figure(figsize=(10, 8))
    for model_name, result in results.items():
        model = result['best_model']
        X_val = X_train.copy()
        
        for col in categorical_columns:
            X_val[col] = fitted_label_encoders[col].transform(X_val[col])
        X_val[numerical_columns] = RobustScaler().fit_transform(X_val[numerical_columns])
        X_val = X_val[result['feature_names']]
        
        precision, recall, _ = precision_recall_curve(y_train, model.predict_proba(X_val)[:, 1])
        plt.plot(recall, precision, label=f'{model_name} (AUC = {auc(recall, precision):.2f})')
    
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.show()
    
################################### MODEL VALIDATION FUNCTIONS #####################################################################
def calculate_empc(y_true, y_pred, profit_matrix):
    """Calculate Expected Misclassification Profit/Cost"""
    cm = confusion_matrix(y_true, y_pred)
    total_profit = 0
    for i in range(len(profit_matrix)):
        for j in range(len(profit_matrix)):
            total_profit += cm[i, j] * profit_matrix[i, j]
    return total_profit / len(y_true)

def evaluate_model(model, X_train, y_train, X_val, y_val, threshold=0.5):
    """
    Simplified evaluation function for sklearn models (like RandomForest)
    that don't need special Pool objects or categorical feature handling
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Get predictions
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_val = model.predict_proba(X_val)[:, 1]
    y_pred_train = (y_proba_train >= threshold).astype(int)
    y_pred_val = (y_proba_val >= threshold).astype(int)
    
    # Business profit matrix
    profit_matrix = np.array([
        [0, -40],   # [TN, FP]
        [-300, 560]  # [FN, TP]
    ])

    # Inner evaluation function
    def print_metrics(y_true, y_pred, y_proba, name):
        print(f"\n{name} Metrics:")
        print(classification_report(y_true, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print(f"ROC AUC: {roc_auc_score(y_true, y_proba):.4f}")
        print(f"PR AUC: {average_precision_score(y_true, y_proba):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
        print(f"F2 Score: {fbeta_score(y_true, y_pred, beta=2):.4f}")
        print(f"EMPC: {calculate_empc(y_true, y_pred, profit_matrix):.4f}")
        print("=" * 80)

    # Run evaluation
    print_metrics(y_train, y_pred_train, y_proba_train, "Training")
    print_metrics(y_val, y_pred_val, y_proba_val, "Validation")
    
    # Plot curves
    plot_curves(y_train, y_proba_train, y_val, y_proba_val)
    
    return model, y_proba_val, y_val

def plot_curves(y_train, y_proba_train, y_val, y_proba_val):
    """Helper function to plot ROC and PR curves"""
    # ROC Curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_proba_val)
    plt.plot(fpr_train, tpr_train, label="Train ROC")
    plt.plot(fpr_val, tpr_val, label="Validation ROC")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    
    # PR Curve
    plt.subplot(1, 2, 2)
    prec_train, rec_train, _ = precision_recall_curve(y_train, y_proba_train)
    prec_val, rec_val, _ = precision_recall_curve(y_val, y_proba_val)
    plt.plot(rec_train, prec_train, label="Train PR")
    plt.plot(rec_val, prec_val, label="Validation PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
################################### MODEL TEST FUNCTIONS #####################################################################

def evaluate_rf_test(model, X_test, y_test, threshold=0.5):
    """
    Evaluate RandomForest model on test set using same metrics as validation
    
    Parameters:
    - model: Trained RandomForest model
    - X_test: Test features (DataFrame or array)
    - y_test: True test labels
    - threshold: Decision threshold (default 0.5)
    """
    # Get probabilities and predictions
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= threshold).astype(int)
    
    # Business profit matrix (same as validation)
    profit_matrix = np.array([
        [0, -40],   # [TN, FP]
        [-300, 560] # [FN, TP]
    ])
    
    # Calculate metrics (identical to validation evaluation)
    print("\nTest Set Metrics:")
    print(classification_report(y_test, y_pred_test))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba_test):.4f}")
    print(f"PR AUC: {average_precision_score(y_test, y_proba_test):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_test):.4f}")
    print(f"F2 Score: {fbeta_score(y_test, y_pred_test, beta=2):.4f}")
    print(f"EMPC: {calculate_empc(y_test, y_pred_test, profit_matrix):.4f}")
    print("=" * 80)
    
    # Plot curves (same style as validation)
    plot_curve(y_test, y_proba_test, "Test Set")
    
    # Return predictions for use outside
    return y_pred_test,  y_proba_test

def plot_curve(y_true, y_proba, dataset_name):
    """Consistent plotting with validation evaluation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax1.plot(fpr, tpr, label=f"{dataset_name} ROC (AUC = {roc_auc_score(y_true, y_proba):.2f})")
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax2.plot(recall, precision, label=f"{dataset_name} PR (AUC = {average_precision_score(y_true, y_proba):.2f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
