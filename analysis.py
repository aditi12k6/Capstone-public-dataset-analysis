"""Which tumour-cell nuclear features differ between malignant and benign
breast masses? Differential analysis with multiple-testing control.

Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), public, UCI / scikit-learn.
Run:  python analysis.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_breast_cancer

ALPHA = 0.05
np.random.seed(0)


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(ranked, 1)
    return q


data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign
mal, ben = X[y == 0], X[y == 1]
print(f"Samples: {len(X)} (malignant={len(mal)}, benign={len(ben)}), features: {X.shape[1]}")

rows = []
for f in X.columns:
    a, b = mal[f], ben[f]
    t, p_t = stats.ttest_ind(a, b, equal_var=False)
    u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (a.mean() - b.mean()) / pooled  # Cohen's d (malignant - benign)
    rows.append(dict(feature=f, mean_malignant=a.mean(), mean_benign=b.mean(),
                     cohens_d=d, p_welch=p_t, p_mannwhitney=p_u))
res = pd.DataFrame(rows)

# Multiple-testing control: Benjamini-Hochberg FDR and Bonferroni
for col in ["p_welch", "p_mannwhitney"]:
    res["q_" + col[2:]] = bh_fdr(res[col])
res["p_bonferroni_mwu"] = np.minimum(res["p_mannwhitney"] * len(res), 1)
res["significant_fdr"] = res["q_mannwhitney"] < ALPHA
res = res.sort_values("p_mannwhitney").reset_index(drop=True)
res.to_csv("results_table.csv", index=False)

n_raw = (res["p_mannwhitney"] < ALPHA).sum()
n_fdr = res["significant_fdr"].sum()
print(f"Features with raw p<{ALPHA}: {n_raw}/{len(res)}")
print(f"Features with BH q<{ALPHA}: {n_fdr}/{len(res)}")
print(res[["feature", "cohens_d", "p_mannwhitney", "q_mannwhitney"]].head(10).to_string(index=False))

# Figure 1: volcano-style plot
fig, ax = plt.subplots(figsize=(7, 5))
colors = np.where(res["significant_fdr"], "crimson", "grey")
ax.scatter(res["cohens_d"], -np.log10(res["q_mannwhitney"].clip(lower=1e-300)), c=colors)
ax.axhline(-np.log10(ALPHA), ls="--", c="k", lw=0.8)
for _, r in res.head(5).iterrows():
    ax.annotate(r["feature"], (r["cohens_d"], -np.log10(max(r["q_mannwhitney"], 1e-300))), fontsize=7)
ax.set_xlabel("Effect size (Cohen's d, malignant - benign)")
ax.set_ylabel("-log10 BH q-value")
ax.set_title("Nuclear features: malignant vs benign")
fig.tight_layout(); fig.savefig("volcano.png", dpi=150); plt.close(fig)

# Confounder / redundancy check: features are highly correlated, so the 30 tests
# are NOT independent (BH remains valid under positive dependence, but the
# number of truly distinct signals is much smaller than 30).
corr = X.corr()
eig = np.linalg.eigvalsh(corr.values)[::-1]
n_pc90 = int(np.searchsorted(np.cumsum(eig) / eig.sum(), 0.90) + 1)
print(f"Principal components needed for 90% variance: {n_pc90} of {X.shape[1]}")
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(30)); ax.set_yticks(range(30))
ax.set_xticklabels(X.columns, rotation=90, fontsize=5); ax.set_yticklabels(X.columns, fontsize=5)
fig.colorbar(im); ax.set_title("Feature correlation"); fig.tight_layout()
fig.savefig("correlation.png", dpi=150); plt.close(fig)

# Sanity check: permuting labels should give ~0 discoveries (empirical false-positive control)
perm_hits = []
for _ in range(200):
    yp = np.random.permutation(y.values)
    p = [stats.mannwhitneyu(X.loc[yp == 0, f], X.loc[yp == 1, f]).pvalue for f in X.columns]
    perm_hits.append((bh_fdr(p) < ALPHA).sum())
print(f"Label-permutation control: mean discoveries at q<{ALPHA} = {np.mean(perm_hits):.2f} "
      f"(max {max(perm_hits)})")
