# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

Builds on Part 2's exact preprocessing (same `random_state=42`, reproduced at the top of `part3_analysis.py` so `X_train_scaled` / `X_test_scaled` / `y_clf_train` / `y_clf_test` come out identical to what Part 2 used).

---

## Task 1 — Decision Tree baseline (unconstrained)

| Metric | Value |
|---|---|
| Training accuracy | 0.8856 |
| Test accuracy | 0.7248 |
| Train-test gap | 0.1608 |

This one overfits pretty badly — training accuracy sits 16 points above test accuracy, which basically means the tree memorized quirks specific to the training rows instead of picking up on patterns that actually generalize.

Decision trees get called "high variance" for a reason worth spelling out: at every split, the tree greedily grabs whatever feature and threshold divides the data best *right at that moment*, and it never goes back to reconsider a split once it's made. With no depth limit, it just keeps going until nodes are pure or nearly empty, which means it ends up carving out increasingly specific rules that fit individual training rows rather than the underlying pattern those rows came from. Because that greedy process depends so heavily on the exact rows it happens to see, a slightly different training sample can produce a noticeably different tree — and that sensitivity to the specific data is what "high variance" actually refers to.

---

## Task 2 — Controlled Decision Tree

`max_depth=5, min_samples_split=20`:

| Metric | Value |
|---|---|
| Training accuracy | 0.7704 |
| Test accuracy | 0.7693 |
| Train-test gap | 0.0011 |

The gap went from 0.1608 down to 0.0011 — pretty much wiped out the overfitting from Task 1, though it cost some training accuracy along the way (0.886 down to 0.770).

`max_depth=5` caps how many levels deep the tree can go, which directly limits how specific its rules can get. That trades away some ability to fit fine details in the training data (adds a bit of bias) in exchange for generalizing a lot better (cuts down variance) — the standard bias-variance trade-off. `min_samples_split=20` stops a node from splitting further unless there are at least 20 samples sitting in it, which keeps the tree from building rules around tiny, noisy subsets — a split that manages to "perfectly" separate 3 rows is almost certainly just fitting noise, not anything real.

Between the two constraints, that's why the controlled tree's train and test numbers land so close together, compared to the unconstrained tree where training accuracy way outpaces test accuracy.

---

## Task 3 — Gini vs Entropy comparison

Both trained with `max_depth=5`:

| Criterion | Test accuracy |
|---|---|
| Gini | 0.7692 |
| Entropy | 0.7693 |

Basically identical — 0.0001 apart, which is nothing.

Gini impurity: `1 - Σ pᵢ²`
Entropy: `-Σ pᵢ log₂(pᵢ)`

(`pᵢ` here is the proportion of samples in a given node that belong to class `i`.)

A Gini score of 0 means a node is perfectly pure — every sample that landed there belongs to the same class, so there's nothing left to resolve. Gini and Entropy are really measuring the same thing (how mixed-up a node is) just on slightly different scales, which is why swapping one for the other barely changed anything here — in practice they tend to pick very similar splits.

---

## Task 4 — Random Forest

`n_estimators=100, max_depth=10, random_state=42`:

| Metric | Value |
|---|---|
| Training accuracy | 0.7809 |
| Test accuracy | 0.7754 |
| Test ROC-AUC | 0.8497 |

**Top 5 features by importance:**
| Feature | Importance |
|---|---|
| `Request Category_Login Access` | 0.4940 |
| `Issue Type_IT Request` | 0.1340 |
| `Priority_level` | 0.1255 |
| `Request Category_System` | 0.1057 |
| `Request Category_Software` | 0.0635 |

This ranking mostly lines up with the coefficient importance I found back in Part 2's Linear Regression, which was a nice sanity check — two very different kinds of models landing on the same idea of what matters.

How Random Forest actually gets these importance numbers: for every feature, look at every split across every tree where that feature got used, measure how much that split reduced Gini impurity (weighted by how many samples went through it), then average that across all the trees. A feature that keeps producing big, clean splits over and over gets ranked higher.

This is a different animal from a regression coefficient. A coefficient gives you a signed, linear relationship — literally "how much does the prediction move, and in which direction, per unit increase." Feature importance from a forest doesn't have a sign or a direction at all — it just tells you how useful a feature was for cutting down impurity, and it can pick up on non-linear or interaction effects a linear coefficient would completely miss, but you lose any sense of which way the relationship points.

As for bagging — the idea is that each of the 100 trees gets trained on its own bootstrap sample, meaning a random draw *with replacement* from the training data, so every tree ends up seeing a different, overlapping slice of the rows, some repeated, some skipped entirely. On top of that, at every split only a random subset of about √10 ≈ 3 features even get considered as candidates, which forces the trees to diversify instead of all leaning on the same dominant feature every time. When you average predictions across a bunch of trees that have been deliberately decorrelated like this, each tree's individual overfitting mostly cancels out — one tree might overfit to noise in one direction, but another tree trained on a different bootstrap sample is unlikely to overfit to that exact same noise the same way, so it washes out in the average. That's basically why the Random Forest's train-test gap (0.7809 vs 0.7754, about half a point) is so much tighter than the 16-point gap the single unconstrained tree had in Task 1.

---

## Task 4a — Gradient Boosting

`n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42`:

| Metric | Value |
|---|---|
| Training accuracy | 0.7750 |
| Test accuracy | 0.7756 |
| Test ROC-AUC | 0.8496 |

Performs almost identically to the Random Forest. It's included in the cross-validated comparison in Task 5 below.

---

## Task 4b — Feature ablation study

The 5 lowest-importance features from the Task 4 Random Forest: `Agent_age`, `Severity_level`, `ticket_month`, `ticket_dayofweek`, `agent_assigned`.

| Model | Test ROC-AUC |
|---|---|
| Full model (10 features) | 0.8497 |
| Reduced model (5 features removed) | 0.8433 |
| Change | -0.0064 |

Were these features actually useless, or were they pulling some weight? Looks like the latter — AUC dropped by 0.0064 when I took them out, which is small but real, not just noise. If they'd truly been dead weight, removing them should have left AUC flat or even nudged it up slightly (less overfitting to noise). Instead it consistently went down a bit.

Whether that's a trade worth making depends entirely on context. A 0.0064 AUC hit is small enough that in a lot of real deployments, I'd take the simpler 5-feature model without hesitation — fewer features means less data to collect and validate at inference time, a lighter model to serve, and fewer upstream pipelines that can break down the road. But for something with real financial consequences or SLA penalties riding on precise ranking, even a small, consistent drop might not be worth it. There's no clean universal rule here — it's a call that should get made together with whoever actually owns the cost of missed or late tickets, not just decided in isolation by whoever built the model.

---

## Task 5 — Cross-validated comparison

5-fold `StratifiedKFold(shuffle=True, random_state=42)`, scored on `roc_auc`, run on `X_train_scaled` / `y_clf_train`:

| Model | Mean CV AUC | Std CV AUC |
|---|---|---|
| Logistic Regression | 0.8314 | 0.0033 |
| Decision Tree (max_depth=5) | 0.8387 | 0.0031 |
| Random Forest | 0.8461 | 0.0028 |
| Gradient Boosting | 0.8472 | 0.0024 |

Gradient Boosting edges out the rest on mean AUC, with Random Forest close behind — both ensembles clearly beat the single Decision Tree and Logistic Regression, and both also show a bit less spread across folds, which is another sign of the stability you get from ensembling.

Why cross-validation beats a single train-test split for estimating how well a model generalizes: a single 80/20 split only gives you one shot at "how well does this do on unseen data," and that one number depends partly on which rows happened to land in the test set — a bit of bad luck there could make a perfectly good model look worse than it is, or vice versa. Running 5-fold CV instead evaluates the model across five different splits and gives you both the average and the spread (std) across them, which is a much steadier estimate of real generalization — and the std by itself is useful information a single split just can't give you, since it tells you how much performance actually swings depending on which data the model happens to see.

---

## Task 6 — Hyperparameter tuning with GridSearchCV

**Pipeline:** `make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RandomForestClassifier(random_state=42))`, fit on the raw, unimputed `X_train` (5,498 nulls in `Priority_level`, 18,770 in `Agent_age`). The pipeline's own `SimpleImputer` and `StandardScaler` handle both steps internally, fit fresh on each training fold inside `GridSearchCV` — so there's no leakage of imputation or scaling statistics from validation folds into training folds.

**Parameter grid:**
```python
param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5],
}
```
That's 3 × 3 × 2 = 18 configurations, times 5 folds = 90 total model fits.

**Best params:** `{'max_depth': 10, 'min_samples_leaf': 1, 'n_estimators': 200}`
**Best CV score (mean AUC):** 0.8462
**Best pipeline test-set AUC:** 0.8498

On Grid Search vs Randomized Search: Grid Search tries every single combination in the grid, so it's guaranteed to find the best one *within that grid* — but the cost multiplies fast with every extra parameter or value you add (just adding one more value to any single parameter here would jump this from 90 fits to 150). That gets impractical quickly once the grid grows. Randomized Search instead samples a fixed number of random combinations from the space, including continuous ranges rather than just fixed lists, so the compute cost is something you control directly — "try 40 combinations" — rather than something dictated by how big the grid happens to be. The catch is it's not guaranteed to land on the actual best combination; it might miss a spot the grid search would have hit. But in practice it tends to get close to optimal for a lot less compute, especially when only a couple of the hyperparameters really matter — which looks like the case here too, since `min_samples_leaf` barely moved the needle across the grid results.

---

## Task 7 — Manual learning curve

Took the best pipeline from Task 6 and refit it on five progressively larger chunks of `X_train`:

| Training fraction | N rows | Training AUC | Test AUC |
|---|---|---|---|
| 0.2 | 15,599 | 0.8864 | 0.8468 |
| 0.4 | 31,199 | 0.8684 | 0.8482 |
| 0.6 | 46,798 | 0.8626 | 0.8486 |
| 0.8 | 62,398 | 0.8602 | 0.8491 |
| 1.0 | 77,998 | 0.8581 | 0.8498 |


**Does training AUC drop as the training set grows?** Yes — from 0.886 at 20% down to 0.858 at 100%. That's what you'd expect from a high-variance, high-capacity model: with very little data it can fit that small set almost too well, but as more data comes in it can't perfectly fit every row anymore, so training performance settles back down toward whatever the model's real underlying skill level actually is.

**Does test AUC go up with more data?** Yes, but only modestly, and it's clearly leveling off — 0.8468 at 20% up to 0.8498 at 100%, and most of that climb happened early. From 60% to 100% the gain is only about 0.003. More data is still helping a little, just less and less each time.

**So — data-limited, or capacity-limited?** Looks closer to capacity-limited at this point. Test AUC is technically still inching up at 100% of the data, so more rows would probably help a bit more, but the curve has flattened out enough, and there's a persistent 1-2 point gap between training and test AUC that never fully closes, that I'd guess the current hyperparameters (or maybe just the feature set itself) are closer to their ceiling than the data collection is the bottleneck. Doubling the dataset from here would likely buy a small extra bump, but I'd expect better features — something like the actual free-text ticket description, which isn't in this dataset — to move the needle more than just gathering more rows in the same feature space.

---

## Task 8 — Serialize the best model

```python
import joblib
joblib.dump(best_pipeline, "best_model.pkl")
```
`best_model.pkl` comes out to 14.4 MB, which is under the 100 MB limit, so it's committed directly rather than needing a regeneration script.

**Reload-and-predict block** (this runs cleanly in both `part3_analysis.py` and `Part3_Modeling.ipynb`):
```python
loaded_model = joblib.load("best_model.pkl")

hand_crafted_rows = pd.DataFrame([
    {  # Likely fast: Login Access, high priority
        "Priority_level": 3, "Severity_level": 2, "Agent_age": 35,
        "ticket_month": 6, "ticket_dayofweek": 2, "agent_assigned": 1,
        "Request Category_Login Access": 1, "Request Category_Software": 0,
        "Request Category_System": 0, "Issue Type_IT Request": 0,
    },
    {  # Likely slow: Hardware, low priority, unassigned
        "Priority_level": 0, "Severity_level": 3, "Agent_age": 35,
        "ticket_month": 11, "ticket_dayofweek": 4, "agent_assigned": 0,
        "Request Category_Login Access": 0, "Request Category_Software": 0,
        "Request Category_System": 0, "Issue Type_IT Request": 1,
    },
])
preds = loaded_model.predict(hand_crafted_rows)
probas = loaded_model.predict_proba(hand_crafted_rows)[:, 1]
```

**What came out:**
- Row 1 (Login Access, high priority): predicted No breach, P(breach) = 0.004
- Row 2 (Hardware, low priority, unassigned): predicted SLA BREACH, P(breach) = 0.575

Both predictions line up with everything found across Parts 1 through 3 — Login Access tickets resolve fast almost no matter what else is going on, and an unassigned, low-priority Hardware ticket is close to the textbook profile for something likely to breach.

---

## Task 9 — Summary comparison table

| Model | 5-fold CV Mean AUC | 5-fold CV Std AUC | Test-set AUC |
|-------|--------------------|-------------------|--------------|
| Logistic Regression | 0.8314 | 0.0033 | 0.8343 |
| Decision Tree (max_depth=5) | 0.8387 | 0.0031 | 0.8410 |
| Random Forest | 0.8461 | 0.0028 | 0.8497 |
| Gradient Boosting | 0.8472 | 0.0024 | 0.8496 |
| Tuned RF Pipeline (GridSearchCV best) | 0.8462 | — | 0.8498 |

**My recommendation is the tuned Random Forest pipeline from GridSearchCV (Task 6).** Its test-set AUC (0.8498) is basically tied with the untuned Gradient Boosting model (0.8496) and the untuned Random Forest (0.8497) — realistically all three are within noise of each other, so I'm not picking it because it "won." I'm picking it because it's the only one of the three packaged as a single, self-contained Pipeline object that handles its own imputation and scaling internally, which means someone else could pick it up and deploy it as-is without needing to remember or reimplement the exact preprocessing steps separately. Given that raw performance is essentially a three-way tie, that kind of operational simplicity is what actually tips the decision for a production recommendation — not a difference in the third decimal place of AUC.

---

## Files in this repository

- `cleaned_data.csv` — input, produced by Part 1
- `best_model.pkl` — the serialized, tuned Random Forest pipeline (14.4 MB)
- `README.md` — this file

## How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/Jayaprada6/Part3.git
cd Part3
```

### 2. Install the required packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

### 4. Open the notebook

Open `llm_structured.ipynb`.

### 5. Run all cells

Run top to bottom to reproduce the full implementation and all outputs.

Note: `GridSearchCV` in Task 6 runs 90 model fits, so it can take several minutes on a single-core machine.
