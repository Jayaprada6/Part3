# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

Builds on Part 2's exact preprocessing (same `random_state=42`, reproduced
at the top of `part3_analysis.py` to regenerate identical
`X_train_scaled` / `X_test_scaled` / `y_clf_train` / `y_clf_test`).

---

## Task 1 — Decision Tree baseline (unconstrained)

| Metric | Value |
|---|---|
| Training accuracy | **0.8856** |
| Test accuracy | **0.7248** |
| Train-test gap | **0.1608** |

**This tree clearly overfits** — training accuracy is 16 points higher
than test accuracy, meaning it's memorized quirks of the training rows
rather than learning generalizable patterns.

**Why decision trees are high-variance models:** at each split, a
decision tree greedily picks whatever feature and threshold best divides
the data *at that node, in that moment* — it never revisits or reconsiders
earlier splits once made. With no depth limit, it keeps splitting until
nodes are pure (or tiny), carving out increasingly specific rules that fit
individual training rows' idiosyncrasies rather than the true underlying
pattern. Because that greedy process is so sensitive to the exact rows it
happens to see, a slightly different training sample can produce a
substantially different tree — that sensitivity to the specific training
data is what "high variance" means.

---

## Task 2 — Controlled Decision Tree

`max_depth=5, min_samples_split=20`:

| Metric | Value |
|---|---|
| Training accuracy | 0.7704 |
| Test accuracy | 0.7693 |
| Train-test gap | **0.0011** |

**The gap collapsed from 0.1608 to 0.0011** — essentially eliminating the
overfitting seen in Task 1, at the cost of a lower training accuracy
(0.886 → 0.770).

**What `max_depth` and `min_samples_split` do:**
- `max_depth=5` caps how many levels of splits the tree can make,
  directly limiting how specific/complex its decision rules can get. This
  trades away some ability to fit fine-grained training patterns (adding
  bias) in exchange for much better generalization (less variance) — the
  classic bias-variance trade-off.
- `min_samples_split=20` prevents a node from splitting further unless it
  has at least 20 samples, which stops the tree from carving out rules
  based on tiny, noisy subsets (e.g. a split that "perfectly" separates 3
  training rows is almost certainly fitting noise, not signal).

Together, these two constraints are why the controlled tree's train and
test accuracy land almost on top of each other, while the unconstrained
tree's training accuracy vastly outpaces its test accuracy.

---

## Task 3 — Gini vs Entropy comparison

Both trained with `max_depth=5`:

| Criterion | Test accuracy |
|---|---|
| Gini | 0.7692 |
| Entropy | 0.7693 |

Virtually identical — a 0.0001 difference, well within noise.

**Gini impurity:** `1 - Σ pᵢ²`
**Entropy:** `-Σ pᵢ log₂(pᵢ)`

(where `pᵢ` is the proportion of samples belonging to class `i` in a
given node)

**What Gini = 0 means:** a node is perfectly pure — every single sample
that reached that node belongs to the same class, so there's no
remaining uncertainty to resolve at that point in the tree. Both
Gini and Entropy measure the same underlying idea (node impurity/disorder)
on slightly different mathematical scales, which is exactly why swapping
one for the other barely moved the result here — they tend to select
very similar splits in practice.

---

## Task 4 — Random Forest

`n_estimators=100, max_depth=10, random_state=42`:

| Metric | Value |
|---|---|
| Training accuracy | **0.7809** |
| Test accuracy | **0.7754** |
| Test ROC-AUC | **0.8497** |

**Top 5 features by importance:**
| Feature | Importance |
|---|---|
| `Request Category_Login Access` | 0.4940 |
| `Issue Type_IT Request` | 0.1340 |
| `priority_level` | 0.1255 |
| `Request Category_System` | 0.1057 |
| `Request Category_Software` | 0.0635 |

*(ranking mostly matches the coefficient importance found in Part 2's
Linear Regression, which is a reassuring cross-check between two very
different model types agreeing on what matters most)*

**How Random Forest computes feature importance:** for each feature,
across every split in every tree in the forest where that feature was
used, the algorithm measures how much that split reduced Gini impurity
(weighted by how many samples passed through that split), then averages
that reduction across all trees. A feature that consistently produces
big, clean splits across many trees gets a high importance score.

**Why this differs from a linear regression coefficient:** a regression
coefficient measures a specific, signed, *linear* relationship between a
(scaled) feature and the target — "how much does the prediction change,
in which direction, per unit increase." Feature importance from a forest
is unsigned and non-linear — it only measures "how useful was this
feature for reducing impurity," with no notion of direction, and it can
capture non-linear or interaction effects that a linear coefficient
cannot, but in exchange tells you nothing about the direction of the
relationship.

**Bagging, in one paragraph:** bootstrap aggregating means each of the
100 trees in the forest is trained on its own random sample drawn *with
replacement* from the training data (a "bootstrap sample") — so each tree
sees a slightly different, overlapping subset of rows, some rows repeated
and others left out entirely. On top of that, at every single split, only
a random subset of √(number of features) ≈ 3 of the 10 features are even
considered as candidates, forcing the trees to diversify further rather
than all converging on the same dominant feature every time. Averaging
the predictions of many such deliberately-decorrelated trees cancels out
each individual tree's idiosyncratic overfitting — where one tree
overfits to noise in one direction, another tree, trained on a different
bootstrap sample, is unlikely to overfit to the same noise in the same
way, so the errors partially cancel in the average. This is exactly why
the Random Forest's train-test gap (0.7809 vs 0.7754 — about half a
point) is so much tighter than the single unconstrained tree's 16-point
gap in Task 1.

---

## Task 4a — Gradient Boosting

`n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42`:

| Metric | Value |
|---|---|
| Training accuracy | 0.7750 |
| Test accuracy | 0.7756 |
| Test ROC-AUC | **0.8496** |

Nearly identical performance to Random Forest, included in the Task 5
cross-validated comparison below.

---

## Task 4b — Feature ablation study

**5 lowest-importance features** (from the Random Forest in Task 4):
`agent_age`, `severity_level`, `ticket_month`, `ticket_dayofweek`,
`agent_assigned`.

| Model | Test ROC-AUC |
|---|---|
| Full model (10 features) | 0.8497 |
| Reduced model (5 features removed) | 0.8433 |
| **Change** | **-0.0064** |

**Were the removed features genuinely uninformative, or were they
contributing?** They were **mildly contributing**, not pure noise — AUC
dropped by 0.0064 (about two-thirds of a percentage point) when they were
removed, a small but real degradation rather than an improvement or a
flat line. If these features were truly uninformative, removing them
would have left AUC unchanged or even improved it slightly (by reducing
overfitting to noise); instead performance consistently ticked down.

**Production trade-off:** a 0.0064 AUC drop is small enough that, in many
real deployments, it would be a perfectly reasonable trade for a
**simpler, 5-feature model** — fewer features mean less data to collect
and validate at inference time, a smaller/faster model to serve, and less
ongoing maintenance burden (fewer upstream data pipelines that can break).
Whether that trade is "acceptable" ultimately depends on how the model is
used: for a low-stakes internal triage nudge, a 0.6-point AUC cost for a
simpler, cheaper pipeline is an easy yes; for a decision with real
financial or SLA-penalty consequences riding on precise ranking, even a
small, consistent degradation might not be worth it. There's no universal
threshold — it's a judgment call that should be made explicitly with
whoever owns the business cost of missed/late tickets, not assumed away.

---

## Task 5 — Cross-validated comparison

5-fold `StratifiedKFold(shuffle=True, random_state=42)`, `scoring='roc_auc'`,
evaluated on `X_train_scaled` / `y_clf_train`:

| Model | Mean CV AUC | Std CV AUC |
|---|---|---|
| Logistic Regression | 0.8314 | 0.0033 |
| Decision Tree (max_depth=5) | 0.8387 | 0.0031 |
| Random Forest | 0.8461 | 0.0028 |
| **Gradient Boosting** | **0.8472** | **0.0024** |

Gradient Boosting comes out on top by mean CV AUC, with Random Forest a
close second — both ensembles clearly outperform the single Decision
Tree and Logistic Regression, and both show slightly *lower* variance
(std) across folds than the simpler models, another sign of the
stability that ensembling buys.

**Why cross-validation is a more reliable estimate of generalization
than a single train-test split:** a single 80/20 split gives you exactly
one sample of "how well does this model do on held-out data" — and
that one number is itself subject to random luck in which particular
rows happened to land in the test set. A run of bad (or good) luck in
that one split could make a model look worse (or better) than it really
is. 5-fold cross-validation instead evaluates the model on five different
train/test partitions and reports both the average performance *and* its
spread (std) across those partitions — giving a far more stable estimate
of true generalization performance, and the std itself is valuable
information a single split can't provide at all (it tells you how much
the model's performance actually varies with the specific data it sees).

---

## Task 6 — Hyperparameter tuning with GridSearchCV

**Pipeline:** `make_pipeline(SimpleImputer(strategy='median'),
StandardScaler(), RandomForestClassifier(random_state=42))`, fit on
**raw, unimputed `X_train`** (5,498 nulls in `priority_level`, 18,770 in
`agent_age`) — the pipeline's own `SimpleImputer` and `StandardScaler`
steps handle both, fit fresh on each individual training fold inside
`GridSearchCV`, so there's no leakage of imputation statistics or scaling
statistics from validation folds into training folds.

**Parameter grid:**
```python
param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5],
}
```
3 × 3 × 2 = **18 total configurations × 5 folds = 90 total model fits.**

**Best params:** `{'max_depth': 10, 'min_samples_leaf': 1, 'n_estimators': 200}`
**Best CV score (mean AUC):** **0.8462**
**Best pipeline test-set AUC:** **0.8498**

**Grid Search vs Randomized Search trade-off:** Grid Search exhaustively
tries every combination in the grid, which guarantees finding the best
combination *within that grid* — but its cost grows multiplicatively
with every parameter and every value added (adding one more value to any
one parameter here would mean 30 configs × 5 folds = 150 fits instead of
90), making it impractical for larger grids or more parameters.
Randomized Search instead samples a fixed number of random combinations
from the parameter space (including continuous distributions, not just
discrete lists), which scales independently of how many parameters or
values you specify — you decide the compute budget directly (e.g. "try
40 random combinations") rather than it being dictated by the grid's
size. The trade-off is that Randomized Search isn't guaranteed to find
the single best combination — it might miss a good spot the grid would
have hit — but in practice it tends to find a *near*-optimal combination
far more compute-efficiently, especially when only a couple of the
hyperparameters actually matter much (a very common situation, and
arguably true here too, since `min_samples_leaf` barely moved anything
in the CV results one would find by inspecting the full grid results).

---

## Task 7 — Manual learning curve

Best pipeline from Task 6, refit on five progressively larger subsets of
`X_train`:

| Training fraction | N rows | Training AUC | Test AUC |
|---|---|---|---|
| 0.2 | 15,599 | 0.8864 | 0.8468 |
| 0.4 | 31,199 | 0.8684 | 0.8482 |
| 0.6 | 46,798 | 0.8626 | 0.8486 |
| 0.8 | 62,398 | 0.8602 | 0.8491 |
| 1.0 | 77,998 | 0.8581 | 0.8498 |

![Learning curve](plots/learning_curve.png)

**(i) Does training AUC decrease as the training set grows?** **Yes** —
from 0.886 at 20% down to 0.858 at 100%. This is the expected signature
of a high-variance/high-capacity model: with very little data, the model
can fit that small dataset extremely well (even too well, in a sense),
but as more data is added, it can no longer fit every row perfectly and
training performance naturally comes back down toward the model's true
underlying skill level.

**(ii) Does test AUC increase with more training data?** **Yes**, though
modestly and with clearly diminishing returns — from 0.8468 at 20% up to
0.8498 at 100%, a gain of only 0.003 across the entire second half of the
curve (60%→100%). More data is still helping, just less and less at each
step.

**(iii) Conclusion — data-limited or capacity-limited?** The model looks
**closer to capacity-limited than data-limited** at this point. Test AUC
is still technically rising at 100% of the training data, so more data
would likely help *a little* more — but the curve has flattened
substantially (the gain from 80%→100% is tiny compared to 20%→40%), and
the persistent ~1-2 point gap between training and test AUC that never
fully closes suggests the Random Forest's current hyperparameters (or the
feature set itself) are closer to the ceiling of what they can extract
from this data than the data collection is limiting them. Doubling the
dataset size from here would probably buy a small additional AUC gain,
but meaningfully better performance would more likely come from better
features (e.g. an actual free-text ticket description, which this
dataset doesn't have) than from simply collecting more rows in the
current feature space.

---

## Task 8 — Serialize the best model

```python
import joblib
joblib.dump(best_pipeline, "best_model.pkl")
```
`best_model.pkl` — **14.4 MB**, committed to the repository (under the
100 MB limit, no regeneration script needed).

**Reload-and-predict block** (included and confirmed running without
errors in `part3_analysis.py` / `Part3_Modeling.ipynb`):
```python
loaded_model = joblib.load("best_model.pkl")

hand_crafted_rows = pd.DataFrame([
    {  # Likely fast: Login Access, high priority
        "priority_level": 3, "severity_level": 2, "agent_age": 35,
        "ticket_month": 6, "ticket_dayofweek": 2, "agent_assigned": 1,
        "Request Category_Login Access": 1, "Request Category_Software": 0,
        "Request Category_System": 0, "Issue Type_IT Request": 0,
    },
    {  # Likely slow: Hardware, low priority, unassigned
        "priority_level": 0, "severity_level": 3, "agent_age": 35,
        "ticket_month": 11, "ticket_dayofweek": 4, "agent_assigned": 0,
        "Request Category_Login Access": 0, "Request Category_Software": 0,
        "Request Category_System": 0, "Issue Type_IT Request": 1,
    },
])
preds = loaded_model.predict(hand_crafted_rows)
probas = loaded_model.predict_proba(hand_crafted_rows)[:, 1]
```

**Output:**
- Row 1 (Login Access, high priority): predicted **No breach**, P(breach) = 0.004
- Row 2 (Hardware, low priority, unassigned): predicted **SLA BREACH**, P(breach) = 0.575

Both predictions align with the patterns found throughout Parts 1–3 —
Login Access tickets resolve fast almost regardless of other factors,
while unassigned, low-priority Hardware tickets are exactly the profile
most likely to breach.

---

## Task 9 — Summary comparison table

| Model | 5-fold CV Mean AUC | 5-fold CV Std AUC | Test-set AUC |
|---|---|---|---|
| Logistic Regression | 0.8314 | 0.0033 | 0.8343 |
| Decision Tree (max_depth=5) | 0.8387 | 0.0031 | 0.8410 |
| Random Forest | 0.8461 | 0.0028 | 0.8497 |
| Gradient Boosting | 0.8472 | 0.0024 | 0.8496 |
| **Tuned RF Pipeline (GridSearchCV best)** | **0.8462** | — | **0.8498** |

**Recommendation: the tuned Random Forest pipeline from GridSearchCV
(Task 6).** It achieves the highest test-set AUC (0.8498) essentially
tied with the untuned Gradient Boosting model (0.8496) and the untuned
Random Forest (0.8497) — all three ensembles are within noise of each
other — but the tuned pipeline is the only one of the three packaged as
a single, self-contained, serialized `Pipeline` object that handles its
own imputation and scaling internally, meaning it can be handed to
another engineer or deployed as-is without them needing to separately
reimplement or remember the exact preprocessing steps. Given that raw
predictive performance is essentially a three-way tie, that operational
simplicity and reproducibility is the deciding factor for a real
production recommendation, not a marginal AUC difference in the third
decimal place.

---

## Files in this repository

- `cleaned_data.csv` — input, produced by Part 1
- `best_model.pkl` — the serialized, tuned Random Forest pipeline (14.4 MB)
- `plots/learning_curve.png` — training vs test AUC across five training-set fractions
- `README.md` — this file

## How to reproduce

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
python3 llm_structured.ipynb
```
Note: `GridSearchCV` in Task 6 runs 90 model fits and can take several
minutes on a single-core machine.
