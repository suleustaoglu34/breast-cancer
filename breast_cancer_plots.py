# =============================================================================
# GÖRSELLEŞTİRME - Patrício et al. (2018) Replikasyonu
# breast_cancer_analysis.py çalıştırıldıktan sonra bu scripti çalıştırın
# Üretilen grafikler:
#   1. Radyal grafik (Şekil 2) - hasta vs kontrol profili
#   2. LR, RF, SVM için ROC eğrileri (Şekil 3, 4, 5)
#   3. Değişken önem grafiği
#   4. AUC karşılaştırma ısı haritası
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    raise ImportError("pip install ucimlrepo")

# ---- Veriyi yeniden yükle ----
print("Veri yükleniyor...")
dataset = fetch_ucirepo(id=451)
X_raw = dataset.data.features
y_raw = dataset.data.targets

df = X_raw.copy()
df['Classification'] = y_raw.values.ravel()
df = df[df['BMI'] <= 40].dropna().reset_index(drop=True)
df['label'] = (df['Classification'] == 2).astype(int)

FEATURES = ['Glucose', 'Resistin', 'Age', 'BMI', 'HOMA',
            'Leptin', 'Insulin', 'Adiponectin', 'MCP.1']
X = df[FEATURES]
y = df['label']

kontrollar = df[df['label'] == 0]
hastalar   = df[df['label'] == 1]


# =============================================================================
# GRAFİK 1: RADYAL GRAFİK (Şekil 2 - Makale ile aynı)
# =============================================================================
print("Grafik 1: Radyal grafik çiziliyor...")

display_vars = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA',
                'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']

# Her değişken için medyan hesapla, maksimuma böl (normalize et)
hasta_medians   = hastalar[display_vars].median()
kontrol_medians = kontrollar[display_vars].median()

# Normalleştirme: her değişkenin maksimum medyanına böl
max_medians = pd.concat([hasta_medians, kontrol_medians], axis=1).max(axis=1)
hasta_norm   = hasta_medians / max_medians
kontrol_norm = kontrol_medians / max_medians

N = len(display_vars)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Kapatmak için başa ekle

hasta_vals   = hasta_norm.tolist() + hasta_norm.tolist()[:1]
kontrol_vals = kontrol_norm.tolist() + kontrol_norm.tolist()[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.plot(angles, kontrol_vals, 'o-', linewidth=2, color='#2c3e50', label='Kontrol')
ax.fill(angles, kontrol_vals, alpha=0.1, color='#2c3e50')
ax.plot(angles, hasta_vals, 's-', linewidth=2, color='#e74c3c', label='Hasta')
ax.fill(angles, hasta_vals, alpha=0.1, color='#e74c3c')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(display_vars, size=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8, color='grey')
ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
ax.set_title('Şekil 2: Hasta ve Kontrol Klinik Özellik Profilleri\n'
             '(Medyan değerler maksimuma normalize edilmiş)',
             size=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('grafik1_radyal.png', dpi=150, bbox_inches='tight')
plt.show()
print("  → 'grafik1_radyal.png' kaydedildi")


# =============================================================================
# GRAFİK 2: ROC EĞRİLERİ - LR, RF, SVM (Şekil 3, 4, 5)
# =============================================================================
print("Grafik 2: ROC eğrileri çiziliyor...")

# En iyi 4 değişkeni kullan (makale gibi)
best_features = ['Glucose', 'Resistin', 'Age', 'BMI']
X_best = df[best_features]

def get_best_worst_roc(X_data, y_data, clf_factory, n_iter=500, train_ratio=0.698):
    """500 MCCV tekrarından en iyi ve en kötü ROC eğrilerini döndür"""
    rng = np.random.RandomState(42)
    X_arr = X_data.values
    y_arr = y_data.values

    ctrl_idx = np.where(y_arr == 0)[0]
    pati_idx = np.where(y_arr == 1)[0]
    n_ctrl_train = round(len(ctrl_idx) * train_ratio)
    n_pati_train = round(len(pati_idx) * train_ratio)

    results = []
    for _ in range(n_iter):
        rng.shuffle(ctrl_idx)
        rng.shuffle(pati_idx)
        train_idx = np.concatenate([ctrl_idx[:n_ctrl_train], pati_idx[:n_pati_train]])
        test_idx  = np.concatenate([ctrl_idx[n_ctrl_train:], pati_idx[n_pati_train:]])

        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        clf = clf_factory()
        clf.fit(X_train_sc, y_train)

        if hasattr(clf, 'predict_proba'):
            scores = clf.predict_proba(X_test_sc)[:, 1]
        else:
            scores = clf.decision_function(X_test_sc)

        if len(np.unique(y_test)) < 2:
            continue

        auc = roc_auc_score(y_test, scores)
        fpr, tpr, _ = roc_curve(y_test, scores)
        results.append({'auc': auc, 'fpr': fpr, 'tpr': tpr})

    results.sort(key=lambda r: r['auc'])
    return results[-1], results[0]  # En iyi, en kötü

classifiers_plot = [
    ('LR - Lojistik Regresyon', lambda: LogisticRegression(max_iter=1000, random_state=42)),
    ('RF - Rastgele Orman',     lambda: RandomForestClassifier(n_estimators=100, random_state=42)),
    ('SVM - Destek Vektör Makinesi', lambda: SVC(kernel='rbf', probability=True, random_state=42))
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Şekil 3-4-5: En İyi ve En Kötü ROC Eğrileri (4 Değişken, 500 MCCV)',
             fontsize=13, fontweight='bold')

for ax, (clf_name, clf_factory) in zip(axes, classifiers_plot):
    print(f"  {clf_name} işleniyor...")
    best_model, worst_model = get_best_worst_roc(X_best, y, clf_factory, n_iter=500)

    ax.plot(best_model['fpr'],  best_model['tpr'],
            'o-', color='#3498db', markersize=3, linewidth=1.5,
            label=f"En iyi (AUC={best_model['auc']:.2f})")
    ax.plot(worst_model['fpr'], worst_model['tpr'],
            '^-', color='#e74c3c', markersize=3, linewidth=1.5,
            label=f"En kötü (AUC={worst_model['auc']:.2f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('1 - Spesifisite', fontsize=11)
    ax.set_ylabel('Sensitivite', fontsize=11)
    ax.set_title(clf_name, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grafik2_roc_egrileri.png', dpi=150, bbox_inches='tight')
plt.show()
print("  → 'grafik2_roc_egrileri.png' kaydedildi")


# =============================================================================
# GRAFİK 3: DEĞİŞKEN ÖNEMİ ÇUBUĞU GRAFİĞİ
# =============================================================================
print("Grafik 3: Değişken önemi grafiği çiziliyor...")

from sklearn.ensemble import RandomForestClassifier as RFC
rf = RFC(n_estimators=500, random_state=42)
rf.fit(X[FEATURES], y)

importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#e74c3c' if f in best_features else '#95a5a6' for f in importances.index]
bars = ax.barh(importances.index, importances.values, color=colors, edgecolor='white')

for bar, val in zip(bars, importances.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', ha='left', fontsize=9)

ax.set_xlabel('Gini Katsayısı (Önem Skoru)', fontsize=11)
ax.set_title('Değişken Önemi - Random Forest Gini Katsayısı\n'
             '(Kırmızı = En iyi modelde kullanılan 4 değişken)',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

red_patch = mpatches.Patch(color='#e74c3c', label='En iyi modelde kullanılan')
grey_patch = mpatches.Patch(color='#95a5a6', label='Diğer değişkenler')
ax.legend(handles=[red_patch, grey_patch], fontsize=10)

plt.tight_layout()
plt.savefig('grafik3_degisken_onem.png', dpi=150, bbox_inches='tight')
plt.show()
print("  → 'grafik3_degisken_onem.png' kaydedildi")


# =============================================================================
# GRAFİK 4: AUC KARŞILAŞTIRMA ISI HARİTASI (Tablo 4'ün görsel versiyonu)
# =============================================================================
print("Grafik 4: AUC karşılaştırma ısı haritası çiziliyor...")

# Önceden hesaplanmış sonuçları yükle (breast_cancer_analysis.py'den)
try:
    with open('mccv_results.pkl', 'rb') as f:
        all_results = pickle.load(f)

    var_names = list(all_results.keys())
    clf_names = ['LR', 'RF', 'SVM']

    # Orta nokta AUC değerleri
    auc_matrix = np.zeros((len(var_names), len(clf_names)))
    for i, vn in enumerate(var_names):
        for j, cn in enumerate(clf_names):
            lo, hi = all_results[vn][cn]['auc_ci']
            auc_matrix[i, j] = (lo + hi) / 2

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(auc_matrix, cmap='RdYlGn', vmin=0.70, vmax=0.95, aspect='auto')
    plt.colorbar(im, ax=ax, label='Ortalama AUC')

    ax.set_xticks(range(len(clf_names)))
    ax.set_yticks(range(len(var_names)))
    ax.set_xticklabels(clf_names, fontsize=12, fontweight='bold')
    ax.set_yticklabels(var_names, fontsize=11)
    ax.set_xlabel('Sınıflandırıcı', fontsize=12)
    ax.set_ylabel('Değişken Seti', fontsize=12)
    ax.set_title('AUC Karşılaştırma Isı Haritası\n(95% CI ortalaması)',
                 fontsize=12, fontweight='bold')

    for i in range(len(var_names)):
        for j in range(len(clf_names)):
            lo, hi = all_results[var_names[i]][clf_names[j]]['auc_ci']
            ax.text(j, i, f'{auc_matrix[i,j]:.2f}\n[{lo:.2f},{hi:.2f}]',
                    ha='center', va='center', fontsize=8,
                    color='black' if auc_matrix[i, j] < 0.88 else 'white')

    # En iyi hücreyi işaretle (V1-V4, SVM)
    best_row = var_names.index('V1-V4')
    best_col = clf_names.index('SVM')
    ax.add_patch(plt.Rectangle((best_col - 0.5, best_row - 0.5), 1, 1,
                                fill=False, edgecolor='blue', linewidth=3))
    ax.text(best_col, best_row - 0.6, '★ EN İYİ MODEL',
            ha='center', color='blue', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('grafik4_auc_isı_haritasi.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → 'grafik4_auc_isı_haritasi.png' kaydedildi")

except FileNotFoundError:
    print("  UYARI: mccv_results.pkl bulunamadı.")
    print("  Önce 'breast_cancer_analysis.py' çalıştırın!")

print("\n" + "=" * 60)
print("TÜM GRAFİKLER TAMAMLANDI!")
print("Üretilen dosyalar:")
print("  - grafik1_radyal.png")
print("  - grafik2_roc_egrileri.png")
print("  - grafik3_degisken_onem.png")
print("  - grafik4_auc_isı_haritasi.png")
print("=" * 60)
