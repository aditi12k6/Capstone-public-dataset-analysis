# Which nuclear features distinguish malignant from benign breast masses?

Task 04 capstone: analysis of a real public dataset with multiple-testing control.

## Biological question
Fine-needle-aspirate images of breast masses are digitised and nuclei are measured.
**Which nuclear morphology features (size, shape, texture) differ between malignant
and benign masses, and how large are those differences?**

## Data
Wisconsin Diagnostic Breast Cancer (WDBC) dataset, public (UCI Machine Learning
Repository; bundled with scikit-learn as `load_breast_cancer`).
569 masses (212 malignant, 357 benign), 30 numeric features (10 nuclear measurements,
each summarised as mean, standard error and "worst").

## How to run
```bash
pip install -r requirements.txt
python analysis.py
```
Outputs: `results_table.csv`, `volcano.png`, `correlation.png`, and a console summary.

## Methods
1. For each of the 30 features, compare malignant vs benign with a two-sided
   Mann-Whitney U test (no normality assumption; Welch t-test is also reported).
2. Effect size: Cohen's d (malignant - benign).
3. **Multiple-testing control:** Benjamini-Hochberg FDR (q < 0.05) over all 30 tests;
   Bonferroni p-values are also in the table for a stricter view.
4. Checks: feature correlation / effective dimensionality (PCA eigenvalues) and a
   label-permutation control (200 shuffles) to confirm the pipeline does not
   produce spurious discoveries.

## Results
- 27 of 30 features differ significantly at BH q < 0.05 (same count as uncorrected
  p < 0.05, so correction did not remove any hit).
- Strongest signals are size and shape-irregularity features: `worst perimeter`
  (d = 2.37), `worst radius` (2.32), `worst area` (1.97), `worst concave points`
  (2.60), `mean concave points` (2.33). Malignant nuclei are larger and have
  more concave, irregular outlines.
- Not significant: `smoothness error`, `mean fractal dimension`, `texture error`
  (|d| < 0.15, q > 0.2).
- Label-permutation control: mean 0.14 false discoveries per shuffle.

## Interpretation (cautious)
Nuclear enlargement and contour irregularity are consistent with the known
cytological hallmarks of malignancy (pleomorphism), which supports the biological
plausibility of the result. The very large effect sizes mean these features are
useful descriptors of the groups, but this analysis is **associative and
descriptive**, not a validated diagnostic model.

## Limitations and confounders
- **Non-independent tests.** Many features are near-duplicates (radius, perimeter
  and area are geometrically linked; mean/worst versions are correlated). Only ~7
  principal components explain 90% of variance, so there are far fewer independent
  signals than 30. BH is valid under positive dependence, but the count "27
  significant" should not be read as 27 independent findings. The permutation control
  occasionally gave several hits in a single shuffle (max 9), which shows how
  correlated features move together.
- **Possible confounding not measurable here.** The dataset has no patient age,
  menopausal status, hormone status, imaging batch, or collection site, all of
  which could relate to both diagnosis and nuclear measurements.
- **Selection / sampling bias.** Masses come from one clinical source and were
  already biopsied for a reason; results may not hold in screening populations.
- **Label and measurement issues.** Diagnosis is the reference standard, but
  image segmentation choices affect feature values.
- **Large n inflates significance.** With 569 samples tiny differences become
  "significant"; effect sizes matter more than p-values.
- **No external validation.** Everything is in-sample; a replication cohort would
  be needed for any clinical claim.
- Possible multiple observations per patient are not documented in the dataset.

## Files
| File | Contents |
|---|---|
| `analysis.py` | full pipeline |
| `results_table.csv` | per-feature means, d, p-values, q-values |
| `volcano.png` | effect size vs significance |
| `correlation.png` | feature correlation heatmap |
