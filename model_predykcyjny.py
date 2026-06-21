import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')   # wycisza ostrzeżenia RDKit

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef
)

# ─────────────────────────────────────────────────────────────
# 1. WCZYTANIE GŁÓWNEJ BAZY TOX21 + LISTY ASSAYÓW
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KROK 1: Wczytywanie danych Tox21 + listy assayów...")
print("=" * 60)

TARGET = "SR-p53"

# Główna baza
df_raw = pd.read_csv("tox21.csv.gz", compression='gzip')
print(f"  Tox21 surowe — wymiary:   {df_raw.shape}")
print(f"  Kolumny:                  {df_raw.columns.tolist()}")

df_tox = df_raw.dropna(subset=[TARGET]).reset_index(drop=True)
df_tox["sample_id"] = df_tox.index   # spójny z zbior_walidacyjny.py

class_counts = df_tox[TARGET].value_counts()
print(f"\n  Po filtracji do '{TARGET}': {len(df_tox)} wierszy")
print(f"  Nieaktywne (0): {class_counts.get(0, 0)}  "
      f"({class_counts.get(0, 0) / len(df_tox) * 100:.1f}%)")
print(f"  Aktywne    (1): {class_counts.get(1, 0)}  "
      f"({class_counts.get(1, 0) / len(df_tox) * 100:.1f}%)")
print(f"  Typ sample_id: {df_tox['sample_id'].dtype}  "
      f"  Zakres: {df_tox['sample_id'].min()} – {df_tox['sample_id'].max()}")

TARGET = "SR-p53"

# automatyczne wykrycie assayów po typie danych
SELECTED_ASSAYS = [
    col for col in df_tox.columns
    if col != TARGET
    and df_tox[col].dropna().isin([0, 1]).all()
]

print(f"\n  Auto-wykryte assay’e (binary): {len(SELECTED_ASSAYS)}")
print(SELECTED_ASSAYS)


# ─────────────────────────────────────────────────────────────
# 2. WCZYTANIE ZBIORU EKSPERYMENTALNEGO
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KROK 2: Wczytywanie zbioru eksperymentalnego...")
print("=" * 60)

try:
    df_exp = pd.read_csv("experiment_set.csv")
    print(f"  Wymiary:          {df_exp.shape}")
    print(f"  Kolumny:          {df_exp.columns.tolist()}")
    print(f"  Unikalne IDs:     {df_exp['sample_id'].nunique()}")
    print(f"  Typ sample_id:    {df_exp['sample_id'].dtype}")
    print(f"  Zakresy ID:       {df_exp['sample_id'].min()} – {df_exp['sample_id'].max()}")
    print(f"  Klasy ({TARGET}):   {df_exp[TARGET].value_counts().to_dict()}")
    if 'group' in df_exp.columns:
        print(f"  Grupy:\n{df_exp['group'].value_counts().to_string()}")
    N_SEEDS = df_exp['seed'].nunique()
    print(f"  Liczba seedów:    {N_SEEDS}")
    print(f"  Próbek per seed:  {len(df_exp) // N_SEEDS}")
    print(f"  Unikalne sample_id (łącznie): {df_exp['sample_id'].nunique()}")

    # Sprawdź czy wszystkie assaye są w zbiorze eksperymentalnym
    missing_in_exp = [a for a in SELECTED_ASSAYS if a not in df_exp.columns]
    if missing_in_exp:
        print(f"  BŁĄD: Assaye {missing_in_exp} nie ma w experiment_set.csv!")
        print("  Upewnij się że zbior_walidacyjny.py był uruchomiony po zmianach.")
        exit(1)
    else:
        print(f"  Assaye w exp CSV: OK  ({SELECTED_ASSAYS})")

except FileNotFoundError:
    print("  BŁĄD: Nie znaleziono experiment_set.csv !")
    print("  Uruchom najpierw zbior_walidacyjny.py")
    exit(1)

print("\n  [OK] Zbiór eksperymentalny wczytany")


# ─────────────────────────────────────────────────────────────
# 3. OBLICZENIE DESKRYPTORÓW RDKIT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KROK 3: Obliczanie deskryptorów chemicznych (RDKit)...")
print("=" * 60)
print("  Każda substancja: SMILES → 7 liczb opisujących strukturę")

RDKIT_FEATURE_NAMES = [
    'MolWt',            # masa cząsteczkowa (g/mol)
    'LogP',             # lipofilność
    'NumHDonors',       # donory wiązań H
    'NumHAcceptors',    # akceptory wiązań H
    'NumAromaticRings', # pierścienie aromatyczne
    'TPSA',             # polarna powierzchnia (Å²)
    'NumRotatableBonds' # elastyczność cząsteczki
]


def compute_rdkit_descriptors(smiles_str):
    mol = Chem.MolFromSmiles(str(smiles_str))
    if mol is None:
        return None
    return {
        'MolWt':             Descriptors.MolWt(mol),
        'LogP':              Descriptors.MolLogP(mol),
        'NumHDonors':        Descriptors.NumHDonors(mol),
        'NumHAcceptors':     Descriptors.NumHAcceptors(mol),
        'NumAromaticRings':  rdMolDescriptors.CalcNumAromaticRings(mol),
        'TPSA':              Descriptors.TPSA(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
    }


def build_rdkit_matrix(dataframe, id_col, smiles_col, target_col, label=""):
    rdkit_rows, valid_ids, target_vals = [], [], []
    n_failed = 0
    for _, row in dataframe.iterrows():
        desc = compute_rdkit_descriptors(row[smiles_col])
        if desc is not None:
            rdkit_rows.append(desc)
            valid_ids.append(row[id_col])
            target_vals.append(int(row[target_col]))
        else:
            n_failed += 1

    X_rdkit = pd.DataFrame(rdkit_rows, index=valid_ids)
    y       = pd.Series(target_vals, index=valid_ids, name=target_col)
    if label:
        print(f"  {label}: {len(X_rdkit)} substancji z poprawnym SMILES  "
              f"({n_failed} odrzuconych)")
    return X_rdkit, y, valid_ids


# Deskryptory dla całej bazy Tox21
X_rdkit_all, y_all, valid_tox_ids = build_rdkit_matrix(
    df_tox, 'sample_id', 'smiles', TARGET, "Tox21"
)

# Deskryptory liczymy tylko dla unikalnych substancji (bez duplikatów seedów)
df_exp_unique = df_exp.drop_duplicates(subset='sample_id').reset_index(drop=True)
X_rdkit_exp, y_exp, valid_exp_ids = build_rdkit_matrix(
    df_exp_unique, 'sample_id', 'smiles', TARGET, "Eksperymentalne (unikalne)"
)
print(f"  Unikalnych substancji eksperymentalnych: {len(X_rdkit_exp)}")

print(f"\n  Deskryptory RDKit (kolumny): {RDKIT_FEATURE_NAMES}")
print(f"  Przykład (pierwsza substancja Tox21):")
print(f"  {X_rdkit_all.iloc[0].to_dict()}")
print("\n  [OK] Deskryptory RDKit obliczone")


# ─────────────────────────────────────────────────────────────
# 4. BUDOWA POŁĄCZONEJ MACIERZY CECH (RDKIT + ASSAYE)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KROK 4: Budowanie macierzy cech RDKit + Assaye...")
print("=" * 60)
print(f"  Cechy RDKit:   {len(RDKIT_FEATURE_NAMES)}")
print(f"  Cechy assay:   {len(SELECTED_ASSAYS)}")
print(f"  Razem:         {len(RDKIT_FEATURE_NAMES) + len(SELECTED_ASSAYS)} cech")

# ── Tox21: pobierz wartości assayów dla valid_ids ─────────
df_tox_by_id = df_tox.set_index('sample_id')
X_assays_all = df_tox_by_id.loc[valid_tox_ids, SELECTED_ASSAYS].copy()

print(f"\n  Kształt X_rdkit_all:    {X_rdkit_all.shape}")
print(f"  Kształt X_assays_all:   {X_assays_all.shape}")
print(f"  Brakujące w assayach (Tox21):")
for col in SELECTED_ASSAYS:
    n_miss = X_assays_all[col].isna().sum()
    pct    = n_miss / len(X_assays_all) * 100
    print(f"    {col}: {n_miss}  ({pct:.1f}%)")

# Sprawdź wyrównanie indeksów przed konkatenacją
assert list(X_rdkit_all.index) == list(X_assays_all.index), \
    "BŁĄD: niezgodność indeksów X_rdkit_all vs X_assays_all!"

# Połącz: [MolWt, LogP, ..., TPSA] + [assay1, ..., assay6]
X_combined_all = pd.concat([X_rdkit_all, X_assays_all], axis=1)
ALL_FEATURE_NAMES = X_combined_all.columns.tolist()
print(f"\n  Łączna macierz X_combined_all: {X_combined_all.shape}")
print(f"  Nazwy wszystkich cech: {ALL_FEATURE_NAMES}")

# ── Eksperymentalne: pobierz wartości assayów ─────────────
df_exp_by_id = df_exp_unique.set_index('sample_id')
X_assays_exp = df_exp_by_id.loc[valid_exp_ids, SELECTED_ASSAYS].copy()

print(f"\n  Kształt X_rdkit_exp:    {X_rdkit_exp.shape}")
print(f"  Kształt X_assays_exp:   {X_assays_exp.shape}")
print(f"  Brakujące w assayach (eksperymentalne):")
for col in SELECTED_ASSAYS:
    n_miss = X_assays_exp[col].isna().sum()
    print(f"    {col}: {n_miss}")

assert list(X_rdkit_exp.index) == list(X_assays_exp.index), \
    "BŁĄD: niezgodność indeksów X_rdkit_exp vs X_assays_exp!"

X_combined_exp = pd.concat([X_rdkit_exp, X_assays_exp], axis=1)
print(f"\n  Łączna macierz X_combined_exp: {X_combined_exp.shape}")
print("  [OK] Macierze cech zbudowane")


# ─────────────────────────────────────────────────────────────
# 5. PODZIAŁ 80/20 — SUBSTANCJE EKSPERYMENTALNE WYKLUCZONE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KROK 5: Podział danych 80/20 (eksperymentalne wykluczone)...")
print("=" * 60)

exp_id_set   = set(valid_exp_ids)
main_mask    = ~X_combined_all.index.isin(exp_id_set)
X_main       = X_combined_all[main_mask]
y_main       = y_all[main_mask]

print(f"  IDs eksperymentalnych wykluczonych:  {len(exp_id_set)}")
print(f"  Substancji dostępnych (train+test):  {len(X_main)}")
print(f"  Aktywne    (1): {(y_main == 1).sum()}")
print(f"  Nieaktywne (0): {(y_main == 0).sum()}")
print(f"  Czy jakieś exp ID zostały w main? "
      f"{X_main.index.isin(exp_id_set).sum()}  (powinno być 0)")

# Podział stratyfikowany 80/20
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_main, y_main,
    test_size=0.20,
    random_state=42,
    stratify=y_main
)

print(f"\n  Zbiór treningowy:  {len(X_train_raw)} substancji")
print(f"    Aktywne (1):     {(y_train == 1).sum()}")
print(f"    Nieaktywne (0):  {(y_train == 0).sum()}")
print(f"\n  Zbiór testowy:     {len(X_test_raw)} substancji")
print(f"    Aktywne (1):     {(y_test == 1).sum()}")
print(f"    Nieaktywne (0):  {(y_test == 0).sum()}")

# Weryfikacja braku przecięcia z exp
overlap = (set(X_train_raw.index) | set(X_test_raw.index)) & exp_id_set
print(f"\n  Przecięcie (train ∪ test) ∩ exp (powinno być 0): {len(overlap)}")
assert len(overlap) == 0, "BŁĄD: substancje eksperymentalne trafiły do train/test!"
print("  [OK] Podział poprawny")

# ─────────────────────────────────────────────────────────────
# 6. IMPUTACJA BRAKUJĄCYCH WARTOŚCI ASSAYÓW
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KROK 6: Imputacja brakujących wartości assayów (mediana)...")
print("=" * 60)

print(f"  Brakujące w X_train_raw:  {X_train_raw.isna().sum().sum()}")
print(f"  Brakujące w X_test_raw:   {X_test_raw.isna().sum().sum()}")
print(f"  Brakujące w X_exp:        {X_combined_exp.isna().sum().sum()}")
print("  Imputator DOPASOWANY tylko na zbiorze treningowym (brak data leakage)")

# SimpleImputer wypełnia medianą — dopasowanie wyłącznie na X_train
imputer = SimpleImputer(strategy='median')

X_train = pd.DataFrame(
    imputer.fit_transform(X_train_raw),
    index   = X_train_raw.index,
    columns = ALL_FEATURE_NAMES
)
X_test = pd.DataFrame(
    imputer.transform(X_test_raw),
    index   = X_test_raw.index,
    columns = ALL_FEATURE_NAMES
)
X_exp = pd.DataFrame(
    imputer.transform(X_combined_exp),
    index   = X_combined_exp.index,
    columns = ALL_FEATURE_NAMES
)

print(f"\n  Po imputacji — brakujące w X_train: {X_train.isna().sum().sum()}")
print(f"  Po imputacji — brakujące w X_test:  {X_test.isna().sum().sum()}")
print(f"  Po imputacji — brakujące w X_exp:   {X_exp.isna().sum().sum()}")
print(f"\n  Kształt X_train: {X_train.shape}")
print(f"  Kształt X_test:  {X_test.shape}")
print(f"  Kształt X_exp:   {X_exp.shape}")
print("  [OK] Imputacja zakończona")

# ─────────────────────────────────────────────────────────────
# 7. Mutual Information (RDKit ONLY)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"KROK 7: Mutual Information — tylko RDKit (dla {TARGET})")
print("=" * 60)

from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt

X_train_rdkit = X_train[RDKIT_FEATURE_NAMES].copy()
X_test_rdkit  = X_test[RDKIT_FEATURE_NAMES].copy()
X_exp_rdkit   = X_exp[RDKIT_FEATURE_NAMES].copy()

print(f"  Liczba RDKit cech: {len(RDKIT_FEATURE_NAMES)}")

# MUTUAL INFORMATION
mi_scores = mutual_info_classif(X_train_rdkit, y_train, random_state=42)

mi_series = pd.Series(mi_scores, index=RDKIT_FEATURE_NAMES)\
    .sort_values(ascending=False)

print("\n  MI scores (RDKit features):")
print("=" * 40)
print(mi_series.to_string())

# WYKRES
fig, ax = plt.subplots(figsize=(8, 4))
mi_series.plot(kind='bar', ax=ax)
ax.set_ylabel('MI score')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(f'feature_selection_MI_RDKIT_{TARGET}.png', dpi=150)
plt.close()

print("\n  Zapisano → feature_selection_MI_RDKIT.png")
print("  [OK] MI ranking zakończony")

# NA TYM ETAPIE NIE WYBIERAMY JESZCZE K
print("\n  TOP ranking cech (do decyzji):")
print(mi_series)

# Feature selection wewnątrz foldów CV — unika selection bias
# MI liczone osobno w każdym foldzie, nie na całym zbiorze treningowym

cv_fs = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_k = []

for k in range(7, 0, -1):
    fold_auc_pr = []

    for fold_train_idx, fold_val_idx in cv_fs.split(X_train[RDKIT_FEATURE_NAMES], y_train):
        X_fold_tr = X_train[RDKIT_FEATURE_NAMES].iloc[fold_train_idx]
        X_fold_val = X_train[RDKIT_FEATURE_NAMES].iloc[fold_val_idx]
        y_fold_tr = y_train.iloc[fold_train_idx]
        y_fold_val = y_train.iloc[fold_val_idx]

        # MI liczone tylko na danych treningowych folda
        mi_fold = mutual_info_classif(X_fold_tr, y_fold_tr, random_state=42)
        mi_fold_series = pd.Series(mi_fold, index=RDKIT_FEATURE_NAMES)
        top_k_fold = mi_fold_series.nlargest(k).index.tolist()

        feat_cols_fold = top_k_fold + SELECTED_ASSAYS
        # Uzupełnij assaye z X_train (po imputacji)
        X_fold_tr_full = pd.concat([
            X_fold_tr[top_k_fold],
            X_train[SELECTED_ASSAYS].iloc[fold_train_idx]
        ], axis=1)
        X_fold_val_full = pd.concat([
            X_fold_val[top_k_fold],
            X_train[SELECTED_ASSAYS].iloc[fold_val_idx]
        ], axis=1)

        model_fold = RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        model_fold.fit(X_fold_tr_full, y_fold_tr)
        proba_fold = model_fold.predict_proba(X_fold_val_full)[:, 1]
        fold_auc_pr.append(average_precision_score(y_fold_val, proba_fold))

    mean_auc_pr = np.mean(fold_auc_pr)
    print(f"  K={k}: AUC-PR CV={mean_auc_pr:.4f} ± {np.std(fold_auc_pr):.4f}")
    results_k.append({'K': k, 'AUC_PR': mean_auc_pr,
                      'AUC_PR_std': np.std(fold_auc_pr)})

# Wybierz K z najwyższym średnim AUC-PR z CV
best_k_row = max(results_k, key=lambda x: x['AUC_PR'])
K = best_k_row['K']
# Teraz oblicz MI na pełnym zbiorze treningowym dla wybranego K
mi_scores_final = mutual_info_classif(X_train[RDKIT_FEATURE_NAMES], y_train, random_state=42)
mi_series = pd.Series(mi_scores_final, index=RDKIT_FEATURE_NAMES).sort_values(ascending=False)
selected_rdkit = mi_series.head(K).index.tolist()
print(f"\n  Wybrane K={K} (najwyższe AUC-PR CV={best_k_row['AUC_PR']:.4f})")
print(f"  Wybrane cechy RDKit: {selected_rdkit}")

# ─────────────────────────────────────────────
# 8. FINALNY ZESTAW CECH (RDKit + ASSAY)
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("KROK 8: Budowa finalnego zestawu cech (RDKit + Assay)")
print("=" * 60)

# RDKit (wybrane K cech z MI)
X_train_rdkit = X_train[selected_rdkit].copy()
X_test_rdkit  = X_test[selected_rdkit].copy()
X_exp_rdkit   = X_exp[selected_rdkit].copy()

# Assaye (już wcześniej wybrane)
X_train_assay = X_train[SELECTED_ASSAYS].copy()
X_test_assay  = X_test[SELECTED_ASSAYS].copy()
X_exp_assay   = X_exp[SELECTED_ASSAYS].copy()

# Łączenie
X_train = pd.concat([X_train_rdkit, X_train_assay], axis=1)
X_test  = pd.concat([X_test_rdkit, X_test_assay], axis=1)
X_exp   = pd.concat([X_exp_rdkit, X_exp_assay], axis=1)

print(f"  RDKit cechy: {len(selected_rdkit)}")
print(f"  Assay cechy: {len(SELECTED_ASSAYS)}")
print(f"  FINALNY wymiar X_train: {X_train.shape}")

print("\n  [OK] Zmieniono X_train / X_test / X_exp na finalne wersje")

# ─────────────────────────────────────────────────────────────
# 9. TRENING MODELU RANDOM FOREST
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KROK 9: Trening modelu Random Forest...")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators    = 200,
    class_weight    = 'balanced',
    max_depth       = None,
    min_samples_leaf= 2,
    random_state    = 42,
    n_jobs          = 1
)

rf_model.fit(X_train, y_train)

print("  Model wytrenowany.")
print(f"  Liczba drzew:     {rf_model.n_estimators}")
print(f"  Liczba cech:      {rf_model.n_features_in_}  "
      f"({len(selected_rdkit)} RDKit + {len(SELECTED_ASSAYS)} assaye)")

# Walidacja krzyżowa 5-fold na danych po SMOTE
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores   = cross_val_score(
    rf_model, X_train, y_train,
    cv=cv_splitter, scoring='roc_auc', n_jobs=1
)
print(f"\n  AUC CV (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Fold scores: {[round(s, 4) for s in cv_scores]}")
print("  [OK] Trening zakończony")

# ─────────────────────────────────────────────────────────────
# 9B. KROKOWA REDUKCJA ASSAYÓW (11 → 1)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"KROK 9B: Redukcja assayów (11 → 1) — {TARGET}")
print("=" * 60)
print("  Na każdym kroku usuwany jest assay o najniższej Permutation Importance.")
print("  Model jest trenowany od nowa i oceniany.")

reduction_results = []
current_assays = SELECTED_ASSAYS.copy()
removed_order = []

for step_n in range(len(SELECTED_ASSAYS) + 1):
    n_assays = len(current_assays)
    feat_cols = selected_rdkit + current_assays
    # ── trening modelu ─────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        min_samples_leaf=2, random_state=42, n_jobs=-1)

    model.fit(X_train[feat_cols], y_train)
    proba = model.predict_proba(X_test[feat_cols])[:, 1]

    # ── dobór progu (optymalizacja F1) ─────────────
    thresholds = np.arange(0.05, 0.60, 0.01)
    f1_scores = [
        f1_score(y_test, (proba >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_thr = thresholds[np.argmax(f1_scores)]
    y_pred = (proba >= best_thr).astype(int)

    # ── metryki ────────────────────────────────────
    auc_pr = average_precision_score(y_test, proba)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_test, y_pred, pos_label=0, zero_division=0)

    row = {
        "n_assays": n_assays,
        "AUC_PR": auc_pr,
        "Balanced_Acc": balanced_acc,
        "MCC": mcc,
        "Recall": recall,   # = czułość = recall klasy aktywnej
        "Specificity": specificity,   # = swoistość
    }

    reduction_results.append(row)

    print(
        f"Liczba assayów={n_assays:2d} | "
        f"AUC-PR={row['AUC_PR']:.3f} | "
        f"MCC={mcc:.3f} | "
        f"Balanced Accuracy={balanced_acc:.3f} | "
        f"Recall/Sensitivity={recall:.3f} | "
        f"Specificity={specificity:.3f}"
    )

    # warunek stopu
    if len(current_assays) <= 1:
        break

    # ── PI → usuwanie najmniej istotnego assayu ───
    perm = permutation_importance(
        model,
        X_test[feat_cols],
        y_test,
        n_repeats=10,
        random_state=42,
        scoring='average_precision'
    )

    perm_series = pd.Series(
        perm.importances_mean[-len(current_assays):],
        index=current_assays
    )

    worst_assay = perm_series.idxmin()
    removed_order.append(worst_assay)
    current_assays = [a for a in current_assays if a != worst_assay]


# ─────────────────────────────────────────────────────────────
# ZAPIS WYNIKÓW
# ─────────────────────────────────────────────────────────────
reduction_df = pd.DataFrame(reduction_results).sort_values("n_assays", ascending=False)

reduction_df.to_csv(f"assay_reduction_results.csv", index=False)

print("\nKolejność usuwania assayów:")
print(removed_order)

print(f"\nZapisano wyniki → assay_reduction_results.csv")


# ─────────────────────────────────────────────────────────────
# NAJLEPSZA KONFIGURACJA
# ─────────────────────────────────────────────────────────────
best = reduction_df.loc[reduction_df["AUC_PR"].idxmax()]

print(
    f"\nNAJLEPSZA KONFIGURACJA:"
    f"Liczba assayów: {int(best['n_assays'])}"
    f"AUC-PR: {best['AUC_PR']:.3f}"
    f"MCC: {best['MCC']:.3f}"
    f"Balanced Accuracy: {best['Balanced_Acc']:.3f}"
    f"Recall/Sensitivity: {best['Recall']:.3f}"
    f"Specificity: {best['Specificity']:.3f}")

ranking = reduction_df.sort_values("AUC_PR", ascending=False)

print("\nRanking konfiguracji (wg AUC-PR):")
print(ranking[["n_assays", "AUC_PR", "MCC", "Balanced_Acc", "Recall"]])


# ─────────────────────────────────────────────────────────────
# WYKRESY
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
metrics_plot = [
    ("AUC_PR", "AUC-PR", "darkorange"),
    ("MCC", "MCC", "seagreen"),
    ("Balanced_Acc", "Balanced Accuracy", "goldenrod"),
    ("Recall", "Recall/Sensitivity", "crimson"),
]

df_sorted = reduction_df.sort_values("n_assays")
x = df_sorted["n_assays"].values

for ax, (col, label, color) in zip(axes.flat, metrics_plot):
    y = df_sorted[col].values

    ax.plot(x, y, marker="o", color=color, lw=2.5)
    ax.set_title(label, fontweight="bold")
    ax.set_xlabel("Liczba assayów")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

plt.tight_layout()
plt.savefig(f"wykres_assay_reduction.png", dpi=150, bbox_inches="tight")
plt.close()

print("  Zapisano → wykres_assay_reduction.png")
print("  [OK] Zakończono redukcję assayów")

# ─────────────────────────────────────────────────────────────
# 10. OCENA MODELU NA ZBIORZE TESTOWYM
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KROK 10: Ocena modelu na zbiorze testowym (20%)...")
print("=" * 60)

# Predykcja prawdopodobieństw
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
print(f"  y_pred_proba — kształt: {y_pred_proba.shape}")
print(f"  Zakres P(aktywna): {y_pred_proba.min():.4f} – {y_pred_proba.max():.4f}")
print(f"  Średnia P(aktywna): {y_pred_proba.mean():.4f}")

# Threshold tuning na osobnym zbiorze walidacyjnym (nie testowym)
# Użycie y_test do tuningu progu = lekki leakage — test powinien być nienaruszony
X_tr_th, X_val_th, y_tr_th, y_val_th = train_test_split(
    X_train, y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)
rf_model_th = RandomForestClassifier(
    n_estimators=200, class_weight='balanced',
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf_model_th.fit(X_tr_th, y_tr_th)
proba_val_th = rf_model_th.predict_proba(X_val_th)[:, 1]

threshold_range = np.arange(0.05, 0.60, 0.01)
f1_by_threshold = [
    f1_score(y_val_th, (proba_val_th >= t).astype(int), zero_division=0)
    for t in threshold_range
]
optimal_threshold = threshold_range[np.argmax(f1_by_threshold)]
print(f"  Optymalny próg (tuning na validation, nie test): {optimal_threshold:.2f}")

# Finalny model (rf_model) predykuje na teście z tym progiem
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

print(f"\n  Przeszukano {len(threshold_range)} progów [0.05 – 0.60]")

# Ostateczne predykcje z optymalnym progiem
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

auc_roc = roc_auc_score(y_test, y_pred_proba)
avg_prec = average_precision_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1_val = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)
specificity  = recall_score(y_test, y_pred, pos_label=0, zero_division=0)

print(f"\n  {'Miara':<35} {'Wartość':>10}")
print(f"  {'-' * 48}")
print(f"  {'AUC-ROC':<35} {auc_roc:>10.4f}")
print(f"  {'Average Precision (AUC-PR)':<35} {avg_prec:>10.4f}")
print(f"  {'Accuracy':<35} {accuracy:>10.4f}")
print(f"  {'Balanced Accuracy':<35} {balanced_acc:>10.4f}")
print(f"  {'Precision':<35} {precision:>10.4f}")
print(f"  {'Recall/Sensivity':<35} {recall:>10.4f}")
print(f"  {'F1-score (opt. próg)':<35} {f1_val:>10.4f}")
print(f"  {'MCC':<35} {mcc:>10.4f}")
print(f"  {'Specificity':<35} {specificity:>10.4f}")
print(f"  {'Optimal Threshold':<35} {optimal_threshold:>10.2f}")

print("\n" + classification_report(
    y_test, y_pred, target_names=['Nieaktywna (0)', 'Aktywna (1)'], zero_division=0))

# Zapis metryk
metrics_df = pd.DataFrame({
    'Metric': ['AUC-ROC', 'AUC-PR', 'Accuracy', 'Balanced Accuracy',
               'Precision', 'Recall', 'F1-score', 'MCC', 'Specificity'],
    'Value':  [round(auc_roc,4), round(avg_prec,4), round(accuracy,4), round(balanced_acc,4),
               round(precision,4), round(recall,4), round(f1_val,4), round(mcc,4), round(specificity,4)]})

metrics_df.to_csv("metryki_modelu.csv", index=False)
print("  Metryki zapisane → metryki_modelu.csv")
print("  [OK] Ocena zakończona")

# ─────────────────────────────────────────────────────────────
# 11. WYKRESY — OCENA MODELU (7 wykresów)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KROK 11: Generowanie wykresów...")
print("=" * 60)

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11})

# ── Wykres 1: Macierz pomyłek ──────────────────────────
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp_ = cm.ravel()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred: Nieaktywna', 'Pred: Aktywna'],
            yticklabels=['True: Nieaktywna', 'True: Aktywna'],
            linewidths=0.5, linecolor='gray')
ax.set_title(f'AUC={auc_roc:.3f}  Acc={accuracy:.3f}',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Prawdziwa klasa', fontsize=11)
ax.set_xlabel('Predykowana klasa', fontsize=11)
# Etykiety opisowe komórek
for (i, j), lbl in zip([(0,0),(0,1),(1,0),(1,1)],
                        ['TN\n(dobrze nieakt.)', 'FP\n(fałsz. alarm)',
                         'FN\n(przeoczony!)', 'TP\n(dobrze akt.)']):
    ax.text(j+0.5, i+0.75, lbl, ha='center', va='center', fontsize=8, color='gray')
plt.tight_layout()
plt.savefig('wykres_1_macierz_pomylek.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Zapisano → wykres_1_macierz_pomylek.png")

# ── Wykres 2: Krzywa ROC ───────────────────────────────
fpr_vals, tpr_vals, _ = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_vals, tpr_vals, color='steelblue', lw=2.5,
        label=f'Random Forest (AUC = {auc_roc:.3f})')
ax.fill_between(fpr_vals, tpr_vals, alpha=0.1, color='steelblue')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Klasyfikator losowy (AUC = 0.5)')
ax.axhline(y=0.8, color='orange', linestyle=':', lw=1.5, label='Próg akceptacji AUC=0.8')
ax.set_xlabel('False Positive Rate (1 − swoistość)', fontsize=11)
ax.set_ylabel('True Positive Rate (czułość)', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig('wykres_2_krzywa_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Zapisano → wykres_2_krzywa_roc.png")

# ── Wykres 3: Precision-Recall ─────────────────────────
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_curve, prec_curve, color='darkorange', lw=2.5,
        label=f'Random Forest (AP = {avg_prec:.3f})')
ax.fill_between(rec_curve, prec_curve, alpha=0.1, color='darkorange')
baseline_pr = y_test.mean()
ax.axhline(y=baseline_pr, color='gray', linestyle='--', lw=1.5,
           label=f'Baseline (proporcja klas = {baseline_pr:.2f})')
ax.set_xlabel('Recall (czułość)', fontsize=11)
ax.set_ylabel('Precision (precyzja)', fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('wykres_3_precision_recall.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Zapisano → wykres_3_precision_recall.png")

# ── Wykres 4: Permutation Importance ───────────────────

perm = permutation_importance(rf_model, X_test, y_test,
    n_repeats=10, random_state=42, scoring='average_precision'
)

perm_importance = pd.Series(perm.importances_mean, index=X_test.columns
).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))

ax.barh(perm_importance.index, perm_importance.values
)

ax.set_xlabel('Permutation importance')
plt.tight_layout()

plt.savefig('wykres_4_permutation_importance.png',
    dpi=150, bbox_inches='tight')

plt.close()

print("  Zapisano → wykres_4_permutation_importance.png")

# ── Wykres 5: Rozkład prawdopodobieństw predykcji ─────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6,
        color='steelblue', label='Prawdziwie Nieaktywne (0)', density=True)
ax.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6,
        color='coral', label='Prawdziwie Aktywne (1)', density=True)
ax.axvline(x=optimal_threshold, color='black', linestyle='--', lw=1.5,
           label=f'Optymalny próg = {optimal_threshold:.2f}')
ax.set_xlabel(f'Prawdopodobieństwo P(aktywna {TARGET})', fontsize=11)
ax.set_ylabel('Gęstość', fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('wykres_5_rozklad_prawdopodobienstw.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Zapisano → wykres_5_rozklad_prawdopodobienstw.png")

# ── Wykres 6: Walidacja krzyżowa (boxplot) ─────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot(cv_scores, vert=True, patch_artist=True,
           boxprops=dict(facecolor='lightsteelblue', color='steelblue'),
           medianprops=dict(color='darkblue', linewidth=2))
ax.scatter([1] * len(cv_scores), cv_scores,
           color='steelblue', zorder=5, s=60, label='AUC per fold')
ax.axhline(y=cv_scores.mean(), color='orange', linestyle='--', lw=1.5,
           label=f'Średnia AUC = {cv_scores.mean():.3f}')
ax.set_ylabel('AUC-ROC', fontsize=11)
ax.set_title(f'Średnia AUC = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}',
             fontsize=12, fontweight='bold')
ax.set_xticks([])
ax.legend(fontsize=10)
ax.set_ylim([0.5, 1.0])
plt.tight_layout()
plt.savefig('wykres_6_cv_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Zapisano → wykres_6_cv_boxplot.png")

# ── Wykres 7: Porównanie wszystkich metryk ─────────────
metrics_names = ['ROC-AUC', 'AUC-PR', 'Accuracy', 'Balanced Accuracy',
    'Precision', 'Recall/Sensitivity', 'F1', 'MCC', 'Specificity'
]

metrics_values = [auc_roc, avg_prec, accuracy, balanced_acc,
    precision, recall, f1_val, mcc, specificity
]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(metrics_names, metrics_values)

ax.set_ylim([0, 1])

ax.set_ylabel('Wartość')
plt.xticks(rotation=20)

for bar, val in zip(bars, metrics_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
        f'{val:.3f}', ha='center', fontsize=8)

plt.tight_layout()

plt.savefig('wykres_7_metryki_modelu.png',
    dpi=150, bbox_inches='tight')

plt.close()

print("  Zapisano → wykres_7_metryki_modelu.png")

print("  [OK] Wszystkie 7 wykresów oceny modelu wygenerowane")

# ─────────────────────────────────────────────────────────────
# 12. PREDYKCJA DLA ZBIORU EKSPERYMENTALNEGO
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KROK 12: Predykcja dla zbioru eksperymentalnego...")
print("=" * 60)
print("  UWAGA: To substancje NIEWIDZIANE podczas treningu!")
print(f"  Liczba substancji: {len(X_exp)}")
print(f"  Kształt macierzy:  {X_exp.shape}")
print(f"  Optymalny próg decyzji: {optimal_threshold:.2f}")
print(f"  Łączna liczba próbek (wszystkie seedy): {len(df_exp)}")
print(f"  Próbek per seed: {len(df_exp) // N_SEEDS}")


# Predykcja dla unikalnych substancji — jeden raz
pred_exp_proba = rf_model.predict_proba(X_exp)[:, 1]
pred_exp       = (pred_exp_proba >= optimal_threshold).astype(int)

# Słownik: sample_id → (pred, proba) — do szybkiego wyszukiwania per seed
pred_dict = {
    sid: (int(pred_exp[i]), float(pred_exp_proba[i]))
    for i, sid in enumerate(X_exp.index)
}

# ── Pętla po seedach ──────────────────────────────────────
print(f"\n  Uruchamiam predykcję dla {N_SEEDS} seedów...")
seed_results_list = []
group_acc_per_seed = []

for seed_val in range(N_SEEDS):
    # Pobierz substancje dla tego seeda
    seed_df = df_exp[df_exp['seed'] == seed_val].copy()

    # Uzupełnij predykcje z dict
    seed_df['predicted_label'] = seed_df['sample_id'].map(
        lambda x: pred_dict[x][0] if x in pred_dict else np.nan
    )
    seed_df['prob_active'] = seed_df['sample_id'].map(
        lambda x: pred_dict[x][1] if x in pred_dict else np.nan
    )
    seed_df = seed_df.dropna(subset=['predicted_label'])
    seed_df['predicted_label'] = seed_df['predicted_label'].astype(int)

    true_labels = seed_df[TARGET].astype(int)
    pred_labels = seed_df['predicted_label']
    pred_probas = seed_df['prob_active']

    if true_labels.nunique() < 2:
        continue  # pomiń seed bez obu klas

    # Metryki dla tego seeda
    seed_metrics = {
        'seed':          seed_val,
        'n_total':       len(seed_df),
        'n_correct':     (true_labels == pred_labels).sum(),
        'Accuracy':      accuracy_score(true_labels, pred_labels),
        'Balanced_Acc':  balanced_accuracy_score(true_labels, pred_labels),
        'AUC_ROC':       roc_auc_score(true_labels, pred_probas),
        'AUC_PR':        average_precision_score(true_labels, pred_probas),
        'F1':            f1_score(true_labels, pred_labels, zero_division=0),
        'Precision':     precision_score(true_labels, pred_labels, zero_division=0),
        'Recall':        recall_score(true_labels, pred_labels, zero_division=0),
        'Specificity':   recall_score(true_labels, pred_labels, pos_label=0, zero_division=0),
        'MCC':           matthews_corrcoef(true_labels, pred_labels),
    }
    seed_results_list.append(seed_metrics)

    # Zapis każdej próbki osobno dla każdego seeda
    for _, row in seed_df.iterrows():
        group_acc_per_seed.append({
            'seed': seed_val,
            'sample_id': row['sample_id'],
            'group': row['group'],
            'true_label': int(row[TARGET]),
            'predicted_label': int(row['predicted_label']),
            'prob_active': round(float(row['prob_active']), 2)
        })

seed_results_df = pd.DataFrame(seed_results_list)
group_seed_df   = pd.DataFrame(group_acc_per_seed)

metric_cols = [
    'Accuracy','Balanced_Acc','AUC_ROC','AUC_PR',
    'F1','Precision','Recall','Specificity','MCC'
]

seed_results_df[metric_cols] = (
    seed_results_df[metric_cols].round(2)
)

# ── Statystyki zbiorcze po seedach ───────────────────────
print(f"\n  Wyniki dla {len(seed_results_df)} seedów:")
print(f"\n  {'Miara':<25} {'Średnia':>9} {'SD':>8} {'Min':>8} {'Max':>8}")
print(f"  {'─'*61}")
for col in ['Accuracy','Balanced_Acc','AUC_ROC','AUC_PR','F1','Precision','Recall','Specificity','MCC']:
    mean_v = seed_results_df[col].mean()
    std_v  = seed_results_df[col].std()
    min_v  = seed_results_df[col].min()
    max_v  = seed_results_df[col].max()
    print(f"  {col:<25} {mean_v:>9.4f} {std_v:>8.4f} {min_v:>8.4f} {max_v:>8.4f}")

# ── Wyniki per group (średnia po seedach) ─────────────────
print(f"\n  WYNIKI PER GRUPA (średnia po {N_SEEDS} seedach):")
print("  " + "─" * 65)
print(f"  {'Grupa':<30} {'Śr. Accuracy':>14} {'SD':>8} {'N total':>10}")
print("  " + "─" * 65)

group_seed_df['correct'] = (
    group_seed_df['true_label'] ==
    group_seed_df['predicted_label']
).astype(int)

group_summary_mean = (
    group_seed_df
    .groupby('group')['correct']
    .agg(mean_acc='mean', std_acc='std')
    .reset_index()
)

# Łączna liczba próbek we wszystkich seedach
group_total_n = (
    df_exp.groupby('group')
    .size()
    .reset_index(name='n_total')
)

group_summary_mean = group_summary_mean.merge(
    group_total_n,
    on='group',
    how='left'
)

# Zaokrąglenie
group_summary_mean['mean_acc'] = group_summary_mean['mean_acc'].round(2)
group_summary_mean['std_acc']  = group_summary_mean['std_acc'].round(2)

for _, row in group_summary_mean.iterrows():
    print(f"  {row['group']:<30} {row['mean_acc']:>14.3f} {row['std_acc']:>8.3f} {row['n_total']:>10}")
print("  " + "─" * 65)

# ── Zapis wyników ─────────────────────────────────────────
seed_results_df.to_csv(f"wyniki_eksperymentalne_per_seed_{TARGET}.csv", index=False)
group_seed_df.to_csv(f"wyniki_per_group_per_seed_{TARGET}.csv", index=False)
group_summary_mean.to_csv(f"wyniki_per_group_srednia_{TARGET}.csv", index=False)
print(f"\n  Zapisano → wyniki_eksperymentalne_per_seed_{TARGET}.csv")
print(f"  Zapisano → wyniki_per_group_per_seed_{TARGET}.csv")
print(f"  Zapisano → wyniki_per_group_srednia_{TARGET}.csv")

# ── Wykresy po seedach ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Lewy: boxplot kluczowych metryk po seedach
metrics_box = ['AUC_PR', 'MCC', 'Balanced_Acc', 'Recall', 'Specificity']
axes[0].boxplot([seed_results_df[m] for m in metrics_box],
                labels=metrics_box, patch_artist=True,
                boxprops=dict(facecolor='lightsteelblue', color='steelblue'),
                medianprops=dict(color='darkblue', lw=2))
axes[0].set_ylabel('Wartość')
axes[0].set_ylim([0, 1.05])
plt.setp(axes[0].get_xticklabels(), rotation=30, ha='right', fontsize=8)

# Środkowy: accuracy per group (średnia ± SD)
colors_g = ['coral' if 'active' in g else 'steelblue'
            for g in group_summary_mean['group']]
bars = axes[1].bar(group_summary_mean['group'],
                   group_summary_mean['mean_acc'],
                   yerr=group_summary_mean['std_acc'],
                   color=colors_g, edgecolor='white',
                   capsize=4)
axes[1].axhline(y=0.5, color='gray', linestyle='--', lw=1.5, label='Baseline=0.5')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim([0, 1.15])
axes[1].legend(fontsize=8)
plt.setp(axes[1].get_xticklabels(), rotation=30, ha='right', fontsize=7)
for bar, val in zip(bars, group_summary_mean['mean_acc']):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 f'{val:.2f}', ha='center', fontsize=8)

# Prawy: AUC-PR per seed (linia)
axes[2].plot(seed_results_df['seed'], seed_results_df['AUC_PR'],
             marker='o', color='darkorange', lw=2)
axes[2].axhline(y=seed_results_df['AUC_PR'].mean(), color='gray',
                linestyle='--', lw=1.5,
                label=f"Średnia={seed_results_df['AUC_PR'].mean():.3f}")
axes[2].set_xlabel('Seed')
axes[2].set_ylabel('AUC-PR')
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'wykres_eksperymentalne_multi_seed_{TARGET}.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Zapisano → wykres_eksperymentalne_multi_seed_{TARGET}.png")
print("  [OK] Predykcja eksperymentalna zakończona")

# ─────────────────────────────────────────────────────────────
# PODSUMOWANIE KOŃCOWE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PODSUMOWANIE KOŃCOWE")
print("=" * 60)
print(f"  Endpoint: {TARGET}")
print(f"  Cechy modelu ({X_train.shape[1]} łącznie):")
print(f"    RDKit ({len(selected_rdkit)}):  {selected_rdkit}")
print(f"    Assaye ({len(SELECTED_ASSAYS)}): {SELECTED_ASSAYS}")
print(f"\n  Substancji treningowych:     {len(X_train)}")
print(f"  Substancji testowych:        {len(X_test)}")
print(f"  Substancji eksperyment.:     {len(X_exp)}")
# print(f"\n  Accuracy eksperymentalna:    {exp_acc:.4f}  "
#     f"({n_correct}/{n_total} substancji)")
print("\n" + "=" * 60)
print("KONIEC")
print("=" * 60)
