import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ══════════════════════════════════════════════════════════
# 1. DANE WEJŚCIOWE
# ══════════════════════════════════════════════════════════
PLIK_CSV       = "alures_nts_projects.csv"
KRAJ           = "Poland"
LATA           = range(2021, 2026)            # 2026 niekompletny

# Koszty in vivo za procedurę - PIW-PIB Puławy (cennik lipiec 2025)
INVIVO_MIN, INVIVO_MAX = 330, 700   # PLN brutto / procedura

SCENARIUSZE = [("10% — konserwatywny", 0.10), ("30% — bazowy", 0.30), ("50% — optymistyczny", 0.50)]

# Słowa kluczowe powiązane z endpointem SR-p53 i zakresem modelu
# (sprawdzane w kolumnie `purposes`, case-insensitive)
KEYWORDS_TOX = [
    # basic research
    "oncology",
    "cardiovascular blood and lymphatic system",
    "nervous system",
    "respiratory system",
    "gastrointestinal system including liver",
    "musculoskeletal system",
    "immune system",
    "urogenital/reproductive system",
    "sensory organs (skin, eyes and ears)",
    "endocrine system/metabolism",
    "developmental biology",
    "multisystemic",
    "other",

    
    # translational and applied research
    "human cancer",
    "human infectious disorders",
    "human cardiovascular disorders",
    "human nervous and mental disorders",
    "human respiratory disorders",
    "human gastrointestinal disorders",
    "human musculoskeletal disorders",
    "human immune disorders",
    "human endocrine/metabolism disorders",
    "human urogenital/reproductive disorders",
    "human sensory organ disorders",
    "other human disorders",
    "non-regulatory toxicology and ecotoxicology",

    # regulatory use and routine production
    "regulatory use and routine production",
]

# ══════════════════════════════════════════════════════════
# 2. WCZYTANIE I FILTRACJA
# ══════════════════════════════════════════════════════════
df = pd.read_csv(PLIK_CSV, low_memory=False)
df = df[(df["country"] == KRAJ) & (df["publication_year"].isin(LATA))].copy()
df["publication_year"] = df["publication_year"].astype(int)


def is_tox(purposes):
    if pd.isna(purposes): return False
    t = purposes.lower()
    return any(k.lower() in t for k in KEYWORDS_TOX)

df = df[df["purposes"].apply(is_tox)].copy()
print(f"Projekty toksykologiczne PL po filtracji: {len(df)}")

# ══════════════════════════════════════════════════════════
# 3. LICZBA ZWIERZĄT z expected_harms
# ══════════════════════════════════════════════════════════

def extract_severity_animals(txt):
    if pd.isna(txt):
        return 0

    matches = re.findall(
        r'(?:Mild|Moderate|Severe|Non[- ]?recovery)[^\d]*(\d+)',
        str(txt),
        flags=re.IGNORECASE
    )
    return sum(map(int, matches)) if matches else 0


def extract_reused(txt):
    if pd.isna(txt):
        return 0

    matches = re.findall(
        r'reused[^\d]*(\d+)', str(txt), flags=re.IGNORECASE
    )
    return sum(map(int, matches)) if matches else 0


def extract_returned(txt):
    if pd.isna(txt):
        return 0

    matches = re.findall(
        r'returned[^\d]*(\d+)', str(txt), flags=re.IGNORECASE
    )
    return sum(map(int, matches)) if matches else 0

def extract_rehomed(txt):
    if pd.isna(txt):
        return 0

    matches = re.findall(
        r'rehomed[^\d]*(\d+)',
        str(txt),
        flags=re.IGNORECASE
    )
    return sum(map(int, matches)) if matches else 0


# === GŁÓWNE KOLUMNY ===
df["animals_severity"] = df["expected_harms"].apply(extract_severity_animals)
df["reused_n"] = df["animals_fate"].apply(extract_reused)
df["returned_n"] = df["animals_fate"].apply(extract_returned)
df["rehomed_n"] = df["animals_fate"].apply(extract_rehomed)


# ══════════════════════════════════════════════════════════
# 4. AGREGACJA ROCZNA + TREND
# ══════════════════════════════════════════════════════════
agg = df.groupby("publication_year").agg(
    n_projects = ("nts_id", "count"),
    n_procedures = ("animals_severity", "sum"),
    reused=("reused_n", "sum"),
    returned=("returned_n", "sum"),
    rehomed=("rehomed_n", "sum")
).reset_index()

agg["n_animals_max"] = agg["n_procedures"]

agg["n_animals_min"] = (
    agg["n_animals_max"] - agg["reused"] - agg["returned"]
)

agg["n_animals_min"] = agg["n_animals_min"].clip(lower=0)


print("\nTrend roczny (projekty toksykologiczne PL):")
print(agg.to_string(index=False))

years_arr = agg["publication_year"].values.astype(float)
proj_arr = agg["n_projects"].values.astype(float)
anim_max = agg["n_animals_max"].values.astype(float)
anim_min = agg["n_animals_min"].values.astype(float)

# ══════════════════════════════════════════════════════════
# 6. ROK BAZOWY — symulacja kosztów
# ══════════════════════════════════════════════════════════
rok_baz   = int(agg["publication_year"].max())

n_proj = int(
    agg.loc[
        agg["publication_year"] == rok_baz,
        "n_procedures"
    ].iloc[0]
)

n_zw_max = int(
    agg.loc[
        agg["publication_year"] == rok_baz,
        "n_animals_max"
    ].iloc[0]
)

n_zw_min = int(
    agg.loc[
        agg["publication_year"] == rok_baz,
        "n_animals_min"
    ].iloc[0]
)
print(
    f"\nRok bazowy: {rok_baz}"
    f" | procedury: {n_proj}"
    f" | animals MAX: {n_zw_max}"
    f" | animals MIN: {n_zw_min}"
)
print(f"Koszt 1 procedury in vivo: {INVIVO_MIN}–{INVIVO_MAX} PLN")

wyniki = []
for nazwa, pct in SCENARIUSZE:
    nz = int(n_proj * pct)
    nzw_max = int(n_zw_max * pct)
    nzw_min = int(n_zw_min * pct)
    oszcz_min = nzw_min*INVIVO_MIN
    oszcz_max = nzw_max*INVIVO_MAX
    wyniki.append(dict(scenariusz=nazwa, pct=pct, n_projects=nz,
                       n_animals_min=nzw_min, n_animals_max=nzw_max,
                       oszcz_min=oszcz_min, oszcz_max=oszcz_max))
    print(f"\n{nazwa} ({int(pct*100)}%): "
          f" Liczba zastąpionych procedur: {nz}"
          f" | Liczba zaoszczedzonych zwierząt: {nzw_min:,.0f}–{nzw_max:,.0f}"
          f" | Potencjalne oszczędności: {oszcz_min:,.0f}–{oszcz_max:,.0f} PLN")

# ══════════════════════════════════════════════════════════
# WYKRES 1 - liczba projektów
# ══════════════════════════════════════════════════════════
C_IV, C_AI, C_G = '#D85A30', '#2E75B6', '#1D9E75'

fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(
    years_arr,
    proj_arr,
    color=C_AI,
    alpha=0.8,
    width=0.6,
    label='Dane (ALURES)',
    zorder=3
)

ax.set_title('Projekty toksykologiczne PL', fontweight='bold')
ax.set_ylabel('Liczba projektów')
ax.legend(fontsize=8)
ax.set_xticks(range(2021, 2026))
ax.grid(axis='y', alpha=0.3)

for r, v in zip(years_arr, proj_arr):
    ax.text(
        r,
        v + 0.4,
        str(int(v)),
        ha='center',
        fontsize=8,
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig('projekty_PL.png', dpi=300, bbox_inches='tight')
plt.close()

print("Zapisano → projekty_PL.png")


# ══════════════════════════════════════════════════════════
# WYKRES 2 - liczba zwierząt
# ══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(
    years_arr,
    anim_max,
    color=C_IV,
    alpha=0.8,
    width=0.6,
    label='Dane (ALURES)',
    zorder=3
)

ax.set_title('Zwierzęta w projektach PL', fontweight='bold')
ax.set_ylabel('Liczba zwierząt')
ax.legend(fontsize=8)
ax.set_xticks(range(2021, 2026))
ax.grid(axis='y', alpha=0.3)

ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
)

for r, v in zip(years_arr, anim_max):
    ax.text(
        r,
        v + 5,
        str(int(v)),
        ha='center',
        fontsize=8,
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig('zwierzeta_PL.png', dpi=300, bbox_inches='tight')
plt.close()

print("Zapisano → zwierzeta_PL.png")

# ══════════════════════════════════════════════════════════
# 9. RAPORT WALIDACJI DANYCH
# ══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("RAPORT WALIDACJI DANYCH")
print("="*60)

# --- rozmiar zbioru ---
n_unique = df["nts_id"].nunique()
n_rows = len(df)

print("\n[Wielkość zbioru danych]")
print(f"Liczba unikalnych projektów (nts_id): {n_unique}")
print(f"Liczba wszystkich rekordów: {n_rows}")

# --- struktura celów badań ---
print("\n[Najczęstsze kategorie badań (purposes)]")
print(df["purposes"].value_counts().head(20).to_string())

# --- kontrola filtrów tematycznych ---
human_count = df["purposes"].str.contains("human", case=False, na=False).sum()
endo_count  = df["purposes"].str.contains("endocrine", case=False, na=False).sum()

print("\n[Kontrola działania filtrów tematycznych]")
print(f"Liczba rekordów zawierających 'human': {human_count}")
print(f"Liczba rekordów zawierających 'endocrine': {endo_count}")

# --- kontrola poprawności ekstrakcji severity ---
print("\n[50 rekordów o największej liczbie zwierząt (kontrola ekstrakcji)]")
print(
    df[
        ["nts_id", "publication_year", "country",
         "animals_severity", "purposes", "expected_harms"]
    ]
    .sort_values("animals_severity", ascending=False)
    .head(50)
    .to_string(index=False)
)

print("\n" + "="*60 + "\n")