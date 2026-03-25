# -*- coding: utf-8 -*-
# Breast Cancer Prediction Model - Patricio et al. (2018) Replication
# Using Resistin, glucose, age and BMI to predict breast cancer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    raise ImportError("Please run: pip install ucimlrepo")


# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================
print("=" * 60)
print("SECTION 1: LOADING DATA...")
print("=" * 60)

dataset = fetch_ucirepo(id=451)

X_raw = dataset.data.features
y_raw = dataset.data.targets

print(f"Data shape: {X_raw.shape[0]} rows, {X_raw.shape[1]} columns")
print(f"\nColumns: {list(X_raw.columns)}")
print(f"\nTarget distribution:\n{y_raw.value_counts().to_string()}")
print("  (1 = Healthy control, 2 = Breast cancer patient)")

df = X_raw.copy()
df['Classification'] = y_raw.values.ravel()

# Remove BMI > 40 (paper criteria)
df = df[df['BMI'] <= 40].reset_index(drop=True)
print(f"\nAfter BMI <= 40 filter: {len(df)} participants")

df = df.dropna().reset_index(drop=True)
print(f"After missing value removal: {len(df)} participants")

n_controls = (df['Classification'] == 1).sum()
n_patients = (df['Classification'] == 2).sum()
print(f"\nControls (healthy): {n_controls}")
print(f"Patients (breast cancer): {n_patients}")

FEATURES = ['Glucose', 'Resistin', 'Age', 'BMI', 'HOMA',
            'Leptin', 'Insulin', 'Adiponectin', 'MCP.1']

df['label'] = (df['Classification'] == 2).astype(int)
X = df[FEATURES]
y = df['label']


# =============================================================================
# SECTION 2: DESCRIPTIVE STATISTICS (Table 1)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2: DESCRIPTIVE STATISTICS (Table 1)")
print("=" * 60)

kontrollar = df[df['label'] == 0]
hastalar   = df[df['label'] == 1]

display_vars = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA',
                'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']

print(f"\n{'Variable':<20} {'Patient (med+IQR)':<22} {'Control (med+IQR)':<22} {'p-value':<12}")
print("-" * 76)

for var in display_vars:
    hasta_vals   = hastalar[var].dropna()
    kontrol_vals = kontrollar[var].dropna()

    stat, p_val = mannwhitneyu(hasta_vals, kontrol_vals, alternative='two-sided')

    hasta_med   = hasta_vals.median()
    hasta_iqr   = hasta_vals.quantile(0.75) - hasta_vals.quantile(0.25)
    kontrol_med = kontrol_vals.median()
    kontrol_iqr = kontrol_vals.quantile(0.75) - kontrol_vals.quantile(0.25)

    sig = " ***" if p_val < 0.001 else (" **" if p_val < 0.01 else (" *" if p_val < 0.05 else ""))
    print(f"{var:<20} {hasta_med:.1f} ({hasta_iqr:.1f}){'':<8} "
          f"{kontrol_med:.1f} ({kontrol_iqr:.1f}){'':<8} "
          f"{p_val:.3f}{sig}")

print("\n* p<0.05  ** p<0.01  *** p<0.001")


# =============================================================================
# SECTION 3: UNIVARIATE ROC ANALYSIS (Table 3)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3: UNIVARIATE ROC ANALYSIS (Table 3)")
print("=" * 60)

def compute_auc_ci(y_true, scores, n_boot=1000, ci=0.95):
    auc = roc_auc_score(y_true, scores)
    if auc < 0.5:
        auc = roc_auc_score(y_true, -scores)

    boot_aucs = []
    rng = np.random.RandomState(42)
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            ba = roc_auc_score(y_true[idx], scores[idx])
            if ba < 0.5:
                ba = 1 - ba
            boot_aucs.append(ba)
        except:
            pass

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_aucs, alpha * 100)
    upper = np.percentile(boot_aucs, (1 - alpha) * 100)
    return auc, lower, upper

def youden_sensitivity_specificity(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    sens = tpr[best_idx]
    spec = 1 - fpr[best_idx]
    return sens, spec

print(f"\n{'Variable':<15} {'AUC 95% CI':<20} {'Sens%':<10} {'Spec%':<10}")
print("-" * 55)

y_arr = y.values

for var in display_vars:
    scores = df[var].values
    auc, lower, upper = compute_auc_ci(y_arr, scores)

    if lower > 0.5:
        if roc_auc_score(y_arr, scores) < 0.5:
            scores_dir = -scores
        else:
            scores_dir = scores
        sens, spec = youden_sensitivity_specificity(y_arr, scores_dir)
        sens_str = f"{sens*100:.0f}%"
        spec_str = f"{spec*100:.0f}%"
    else:
        sens_str = "-"
        spec_str = "-"

    print(f"{var:<15} [{lower:.2f}, {upper:.2f}]{'':<8} {sens_str:<10} {spec_str:<10}")


# =============================================================================
# SECTION 4: VARIABLE IMPORTANCE (Gini - Random Forest)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4: VARIABLE IMPORTANCE (Gini Coefficient)")
print("=" * 60)

rf_importance = RandomForestClassifier(n_estimators=500, random_state=42)
rf_importance.fit(X[FEATURES], y)

importances = pd.Series(rf_importance.feature_importances_, index=FEATURES)
importances_sorted = importances.sort_values(ascending=False)

print("\nVariable importance ranking (Gini):")
for rank, (feat, imp) in enumerate(importances_sorted.items(), 1):
    print(f"  {rank}. {feat:<15} {imp:.4f}")

print("\nPaper ranking:")
print("  Glucose > Resistin > Age > BMI > HOMA > Leptin > Insulin > Adiponectin > MCP-1")


# =============================================================================
# SECTION 5: MONTE CARLO CROSS-VALIDATION (Table 4)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5: MONTE CARLO CROSS-VALIDATION (500 iterations)")
print("=" * 60)
print("This may take 1-2 minutes...\n")

def monte_carlo_cv(X_data, y_data, classifier, n_iter=500,
                   train_ratio=0.698, random_state=0):
    aucs, senss, specs = [], [], []
    rng = np.random.RandomState(random_state)

    X_arr = X_data.values if hasattr(X_data, 'values') else X_data
    y_arr = y_data.values if hasattr(y_data, 'values') else y_data

    ctrl_idx = np.where(y_arr == 0)[0]
    pati_idx = np.where(y_arr == 1)[0]

    n_ctrl_train = round(len(ctrl_idx) * train_ratio)
    n_pati_train = round(len(pati_idx) * train_ratio)

    for i in range(n_iter):
        rng.shuffle(ctrl_idx)
        rng.shuffle(pati_idx)

        train_idx = np.concatenate([ctrl_idx[:n_ctrl_train],
                                    pati_idx[:n_pati_train]])
        test_idx  = np.concatenate([ctrl_idx[n_ctrl_train:],
                                    pati_idx[n_pati_train:]])

        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        clf = classifier()
        clf.fit(X_train_sc, y_train)

        if hasattr(clf, 'predict_proba'):
            scores = clf.predict_proba(X_test_sc)[:, 1]
        else:
            scores = clf.decision_function(X_test_sc)

        if len(np.unique(y_test)) < 2:
            continue

        auc = roc_auc_score(y_test, scores)
        aucs.append(auc)

        fpr, tpr, thresholds = roc_curve(y_test, scores)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        senss.append(tpr[best_idx])
        specs.append(1 - fpr[best_idx])

    def ci95(arr):
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    return {
        'auc_ci':  ci95(aucs),
        'sens_ci': ci95(senss),
        'spec_ci': ci95(specs),
        'aucs': aucs, 'senss': senss, 'specs': specs
    }

def make_lr():
    return LogisticRegression(max_iter=1000, random_state=42)

def make_rf():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def make_svm():
    return SVC(kernel='rbf', probability=True, random_state=42)

classifiers = {
    'LR':  make_lr,
    'RF':  make_rf,
    'SVM': make_svm
}

variable_sets = {
    'V1-V2': FEATURES[:2],
    'V1-V3': FEATURES[:3],
    'V1-V4': FEATURES[:4],
    'V1-V5': FEATURES[:5],
    'V1-V6': FEATURES[:6],
    'V1-V9': FEATURES[:9],
}

all_results = {}

for var_name, var_list in variable_sets.items():
    all_results[var_name] = {}
    for clf_name, clf_factory in classifiers.items():
        print(f"  {var_name} | {clf_name} running...", end=' ', flush=True)
        result = monte_carlo_cv(X[var_list], y, classifier=clf_factory, n_iter=500)
        all_results[var_name][clf_name] = result
        print(f"AUC: [{result['auc_ci'][0]:.2f}, {result['auc_ci'][1]:.2f}]")

print("\n" + "=" * 80)
print("TABLE 4: MULTIVARIATE ANALYSIS RESULTS (95% CI)")
print("=" * 80)
print(f"{'':10} {'':12} {'LR':^20} {'RF':^20} {'SVM':^20}")
print("-" * 80)

for var_name in variable_sets:
    for metric_name, metric_key in [('AUC', 'auc_ci'),
                                     ('Sensitivity', 'sens_ci'),
                                     ('Specificity', 'spec_ci')]:
        row = f"{var_name:<10} {metric_name:<12}"
        for clf_name in ['LR', 'RF', 'SVM']:
            r = all_results[var_name][clf_name][metric_key]
            row += f"[{r[0]:.2f}, {r[1]:.2f}]{'':<6}"
        print(row)
    print()

best = all_results['V1-V4']['SVM']
print("=" * 60)
print("BEST MODEL: SVM + 4 Variables (Glucose, Resistin, Age, BMI)")
print(f"  AUC 95% CI:         [{best['auc_ci'][0]:.3f}, {best['auc_ci'][1]:.3f}]")
print(f"  Sensitivity 95% CI: [{best['sens_ci'][0]:.3f}, {best['sens_ci'][1]:.3f}]")
print(f"  Specificity 95% CI: [{best['spec_ci'][0]:.3f}, {best['spec_ci'][1]:.3f}]")
print("\nPaper values:")
print("  AUC: [0.87, 0.91]  |  Sens: [82%, 88%]  |  Spec: [85%, 90%]")
print("=" * 60)

import pickle
with open('mccv_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print("\nResults saved to mccv_results.pkl")
print("Run breast_cancer_plots.py for visualizations.")
