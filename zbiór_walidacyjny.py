import pandas as pd
import numpy as np
import matplotlib
from sklearn.feature_selection import mutual_info_classif

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1. WCZYTANIE DANYCH
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("KROK 1: Wczytywanie danych Tox21...")
print("=" * 60)

TARGET = "SR-p53"

df_raw = pd.read_csv("tox21.csv.gz")
print(f"  Surowe wymiary: {df_raw.shape}")
print(f"  Kolumny: {df_raw.columns.tolist()}")

# Filtracja do wierszy z etykietą SR-p53, reset indeksów
df = df_raw.dropna(subset=[TARGET]).reset_index(drop=True)
df["sample_id"] = df.index   # unikalny indeks = pozycja po reset_index

print(f"\n  Po filtracji do '{TARGET}': {len(df)} wierszy")
print(f"  Aktywne  (1):     {(df[TARGET] == 1).sum()}")
print(f"  Nieaktywne (0):   {(df[TARGET] == 0).sum()}")
print(f"  Typ sample_id:    {df['sample_id'].dtype}")
print(f"  Zakres sample_id: {df['sample_id'].min()} – {df['sample_id'].max()}")
print(f"  Pierwsze 3 sample_id: {df['sample_id'].head(3).tolist()}")
print("  [OK] Dane wczytane")

# ─────────────────────────────────────────────────────────────
# 2. WYBÓR 6 ASSAYÓW (3 dodatnio + 3 ujemnie skorelowane)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"KROK 2: Wybór 6 assayów korelujących z {TARGET}...f")
print("=" * 60)

# Znajdź kolumny binarne (0/1), z pominięciem TARGET
binary_assay_columns = []
for col in df.columns:
    unique_vals = set(df[col].dropna().unique())
    if unique_vals.issubset({0.0, 1.0, 0, 1}) and col != TARGET:
        binary_assay_columns.append(col)

print(f"  Znalezione binarne assaye (poza {TARGET}): {len(binary_assay_columns)}")
print(f"  Lista: {binary_assay_columns}")

# Mutual Information dla assayów binarnych
mi_scores = mutual_info_classif(
    df[binary_assay_columns].fillna(0),
    df[TARGET],
    discrete_features=True,
    random_state=42
)

mi_series = pd.Series(
    mi_scores,
    index=binary_assay_columns
).sort_values()

print(f"\n  Mutual Information assayów z {TARGET}:")
print(mi_series.to_string())

top_assays = mi_series.sort_values(ascending=False).head(3).index.tolist()
bottom_assays = mi_series.sort_values().head(3).index.tolist()
selected_assays     = top_assays + bottom_assays   # 6 łącznie

print(f"  Wybrane 6 assayów:          {selected_assays}")

# Wykres MI
bar_colors = ['coral' if v > 0 else 'steelblue' for v in mi_series.values]
fig, ax = plt.subplots(figsize=(10, 5))
mi_series.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='white')
ax.axhline(0, color='black', lw=1)
ax.set_ylabel("Mutual Information")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("mi_assayow.png", dpi=150)
plt.close()
print("  Zapisano → mi_assayow.png")
print("  [OK] Assaye wybrane")

# ─────────────────────────────────────────────────────────────
# 3. PROFILOWANIE SUBSTANCJI (grupy aktywnych i nieaktywnych)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KROK 3: Profilowanie substancji na grupy...")
print("=" * 60)

df_active   = df[df[TARGET] == 1].copy()
df_inactive = df[df[TARGET] == 0].copy()


def compute_overlap_ratio(row, assays):
    values = [row[a] for a in assays]
    active = sum(v == 1 for v in values if pd.notna(v))
    total  = len(assays)
    return active / total


def assign_active_group(row):
    """Przypisuje grupę dla substancji AKTYWNEJ w SR-p53."""
    ratio = compute_overlap_ratio(row, selected_assays)
    if ratio == 0:
        return "active_no_overlap"
    elif ratio < 0.66:
        return "active_partial_overlap"
    else:
        return "active_high_overlap"


def assign_inactive_group(row):
    """Przypisuje grupę dla substancji NIEAKTYWNEJ w SR-p53."""
    ratio = compute_overlap_ratio(row, selected_assays)
    if ratio == 0:
        return "inactive_no_overlap"
    elif ratio < 0.66:
        return "inactive_partial_overlap"
    else:
        return "inactive_high_overlap"


df_active["group"]   = df_active.apply(assign_active_group, axis=1)
df_inactive["group"] = df_inactive.apply(assign_inactive_group, axis=1)

print(f"  Rozkład grup - AKTYWNE ({len(df_active)} substancji):")
print(df_active["group"].value_counts().to_string())
print(f"\n  Rozkład grup - NIEAKTYWNE ({len(df_inactive)} substancji):")
print(df_inactive["group"].value_counts().to_string())
print("  [OK] Profilowanie zakończone")

# ─────────────────────────────────────────────────────────────
# 4. PRÓBKOWANIE ZBIORU EKSPERYMENTALNEGO
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KROK 4: Próbkowanie - 5 substancji z każdej grupy...")
print("=" * 60)

N_PER_GROUP = 5
experiment_set = pd.DataFrame()

# ─────────────────────────────
# ACTIVE
# ─────────────────────────────
for seed in range(20):

    print("\n" + "=" * 60)
    print(f"SEED {seed} - ACTIVE")
    print("=" * 60)

    seed_sample = pd.DataFrame()

    for group_name in sorted(df_active["group"].unique()):

        subset = df_active[df_active["group"] == group_name]

        sampled = subset.sample(
            n=min(N_PER_GROUP, len(subset)),
            random_state=seed
        )

        seed_sample = pd.concat([seed_sample, sampled])

        print(f"[active / {group_name}]: {len(sampled)} / {len(subset)}")

    # kontrola unikalności w seedzie
    assert seed_sample["sample_id"].is_unique, "Duplikaty w seedzie ACTIVE!"

    # dodaj numer seed
    seed_sample["seed"] = seed

    # shuffle kolejności
    seed_sample = seed_sample.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    # dodaj do głównego zbioru
    experiment_set = pd.concat(
        [experiment_set, seed_sample],
        ignore_index=True
    )


# ─────────────────────────────
# INACTIVE
# ─────────────────────────────
for seed in range(20):

    print("\n" + "=" * 60)
    print(f"SEED {seed} - INACTIVE")
    print("=" * 60)

    seed_sample = pd.DataFrame()

    for group_name in sorted(df_inactive["group"].unique()):

        subset = df_inactive[df_inactive["group"] == group_name]

        sampled = subset.sample(
            n=min(N_PER_GROUP, len(subset)),
            random_state=seed
        )

        seed_sample = pd.concat([seed_sample, sampled])

        print(f"[inactive / {group_name}]: {len(sampled)} / {len(subset)}")

    # kontrola unikalności w seedzie
    assert seed_sample["sample_id"].is_unique, "Duplikaty w seedzie INACTIVE!"

    # dodaj numer seed
    seed_sample["seed"] = seed

    # shuffle kolejności
    seed_sample = seed_sample.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    # dodaj do głównego zbioru
    experiment_set = pd.concat(
        [experiment_set, seed_sample],
        ignore_index=True
    )


print(f"\n  Łączny rozmiar zbioru eksperymentalnego: {len(experiment_set)}")
print(f"  Unikalne sample_id: {experiment_set['sample_id'].nunique()}")
print(f"  Rozkład klas: {experiment_set[TARGET].value_counts().to_dict()}")
print(f"  Rozkład grup:\n{experiment_set['group'].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────
# 5. ZAPIS PLIKU WYJŚCIOWEGO
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KROK 5: Zapis zbioru eksperymentalnego...")
print("=" * 60)

remaining_assays = [
    col for col in binary_assay_columns
    if col not in selected_assays
]

# Kolumny: identyfikatory + SMILES + TARGET + group + 6 assayów
output_columns = ["seed", "sample_id", "smiles", TARGET, "group"] + selected_assays + remaining_assays
experiment_set_out = experiment_set[output_columns].copy()

print(f"  Kolumny w pliku CSV: {experiment_set_out.columns.tolist()}")
print(f"\n  Brakujące wartości w assayach (NaN):")
for col in selected_assays:
    n_miss = experiment_set_out[col].isna().sum()
    print(f"    {col}: {n_miss}")

print(f"\n  Podgląd - ID, klasa, grupa, assaye):")
print(experiment_set_out[["sample_id", TARGET, "group"] + selected_assays].head(len(experiment_set_out)).to_string())

experiment_set_out.to_csv("experiment_set.csv", index=False)
print(f"\n  Zapisano → experiment_set.csv  ({len(experiment_set_out)} wierszy)")

print("\n" + "=" * 60)
print("PODSUMOWANIE")
print("=" * 60)
print(f"  experiment_set.csv:  {len(experiment_set_out)} próbek")
print(f"\n  Kolumny CSV:         {experiment_set_out.columns.tolist()}")
print(f"  Aktywne   (1):       {(experiment_set_out[TARGET] == 1).sum()}")
print(f"  Nieaktywne (0):      {(experiment_set_out[TARGET] == 0).sum()}")
print(f"  Wybrane assaye:      {selected_assays}")
print("=" * 60)
print("KONIEC")
print("=" * 60)