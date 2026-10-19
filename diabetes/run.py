
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    confusion_matrix
)
from sklearn.inspection import permutation_importance


# ==========================================
# 1. Project paths
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / 'results'

FIG = OUT / 'figures'
TAB = OUT / 'tables'

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Load clinical dataset
# ==========================================

df = pd.read_excel(
    ROOT / 'data' / 'diabetes_dataset.xlsx'
)

df.columns = [
    x.strip().replace(' ', '_')
    for x in df.columns
]

X = df.drop(columns='class')

y = df['class'].map({
    'Positive': 1,
    'Negative': 0
})

assert not y.isna().any(), 'Unexpected class labels'


# ==========================================
# 3. Class distribution
# ==========================================

count = y.value_counts().reindex([1, 0])
names = ['Positive', 'Negative']

fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(names, count.values)

ax.set_title('Diabetes Class Distribution')
ax.set_ylabel('Number of records')

for i, v in enumerate(count.values):
    ax.text(i, v + 4, str(v), ha='center')

fig.tight_layout()

fig.savefig(
    FIG / 'class_distribution.png',
    dpi=220
)

plt.close(fig)


# ==========================================
# 4. Age distribution
# ==========================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(
    [
        df.loc[y == 0, 'Age'],
        df.loc[y == 1, 'Age']
    ],
    bins=15,
    label=['Negative', 'Positive']
)

ax.legend()

ax.set(
    xlabel='Age',
    ylabel='Records',
    title='Age by diabetes class'
)

fig.tight_layout()

fig.savefig(
    FIG / 'age_distribution_by_class.png',
    dpi=220
)

plt.close(fig)


# ==========================================
# 5. Symptom prevalence analysis
# ==========================================

symptoms = [
    c for c in X.columns
    if c not in ('Age', 'Gender')
]

rows = []

for c in symptoms:
    r = X[c].eq('Yes')

    rows.append({
        'symptom': c,
        'positive': r[y == 1].mean(),
        'negative': r[y == 0].mean()
    })

prev = pd.DataFrame(rows)

prev['difference'] = (
    prev['positive'] - prev['negative']
).abs()

prev.sort_values(
    'difference',
    ascending=False
).to_csv(
    TAB / 'symptom_rates.csv',
    index=False
)

show = (
    prev.nlargest(12, 'difference')
    .sort_values('difference')
)

j = np.arange(len(show))

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    j - 0.2,
    show['negative'],
    height=0.38,
    label='Negative'
)

ax.barh(
    j + 0.2,
    show['positive'],
    height=0.38,
    label='Positive'
)

ax.set_yticks(
    j,
    show['symptom'].str.replace('_', ' ')
)

ax.set(
    xlabel='Share of patients reporting symptom',
    title='Symptom prevalence by class'
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIG / 'symptom_prevalence_comparison.png',
    dpi=220
)

plt.close(fig)


# ==========================================
# 6. Correlation heatmap
# ==========================================

enc = X.copy()

for c in X.columns:
    if c == 'Age':
        continue

    enc[c] = X[c].map({
        'Yes': 1,
        'No': 0,
        'Male': 1,
        'Female': 0
    })

enc['Class_Positive'] = y

corr = enc.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(11, 9))

im = ax.imshow(
    corr.values,
    vmin=-1,
    vmax=1,
    cmap='coolwarm'
)

ax.set_xticks(np.arange(len(corr)))

ax.set_xticklabels(
    corr.columns,
    rotation=90,
    fontsize=7
)

ax.set_yticks(np.arange(len(corr)))

ax.set_yticklabels(
    corr.index,
    fontsize=7
)

fig.colorbar(im, ax=ax)

ax.set_title('Encoded feature correlations')

fig.tight_layout()

fig.savefig(
    FIG / 'correlation_heatmap.png',
    dpi=220
)

plt.close(fig)


# ==========================================
# 7. Feature preprocessing
# ==========================================

cat = [
    c for c in X.columns
    if c != 'Age'
]

prep = ColumnTransformer([
    (
        'num',
        Pipeline([
            (
                'imputer',
                SimpleImputer(strategy='median')
            ),
            (
                'scale',
                StandardScaler()
            )
        ]),
        ['Age']
    ),
    (
        'cat',
        Pipeline([
            (
                'imputer',
                SimpleImputer(strategy='most_frequent')
            ),
            (
                'ohe',
                OneHotEncoder(handle_unknown='ignore')
            )
        ]),
        cat
    )
])


# ==========================================
# 8. Train-test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 9. Define machine learning models
# ==========================================

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
}


# ==========================================
# 10. Training and evaluation
# ==========================================

results = []

fig, ax = plt.subplots(figsize=(7, 6))

for name, model in models.items():

    pipe = Pipeline([
        ('preprocess', prep),
        ('model', model)
    ])

    cv = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        ),
        scoring=['roc_auc', 'f1', 'recall']
    )

    pipe.fit(X_train, y_train)

    yp = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)

    results.append({
        'model': name,
        'CV ROC-AUC': cv['test_roc_auc'].mean(),
        'Test Accuracy': accuracy_score(y_test, yp),
        'Test Precision': precision_score(y_test, yp),
        'Test Recall': recall_score(y_test, yp),
        'Test F1': f1_score(y_test, yp),
        'Test ROC-AUC': auc
    })

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba)

    ax.plot(
        fpr,
        tpr,
        label=f'{name} (AUC={auc:.3f})'
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, yp)

    f, a = plt.subplots(figsize=(4.5, 4))

    a.imshow(cm, cmap='Blues')

    a.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=['Pred Neg', 'Pred Pos'],
        yticklabels=['True Neg', 'True Pos'],
        title=f'{name} Confusion Matrix'
    )

    for i in range(2):
        for j in range(2):
            a.text(
                j,
                i,
                str(cm[i, j]),
                ha='center',
                va='center',
                fontsize=14
            )

    f.tight_layout()

    filename = (
        'confusion_matrix_'
        + name.lower().replace(' ', '_')
        + '.png'
    )

    f.savefig(
        FIG / filename,
        dpi=220
    )

    plt.close(f)

    # Random Forest feature importance
    if name == 'Random Forest':

        pi = permutation_importance(
            pipe,
            X_test,
            y_test,
            scoring='roc_auc',
            random_state=42,
            n_repeats=20
        )

        imp = pd.DataFrame({
            'feature': X.columns,
            'importance': pi.importances_mean
        }).sort_values(
            'importance',
            ascending=False
        )

        imp.to_csv(
            TAB / 'random_forest_importance.csv',
            index=False
        )

        top = imp.head(12).iloc[::-1]

        f, a = plt.subplots(figsize=(9, 6))

        a.barh(
            top['feature'],
            top['importance']
        )

        a.set(
            xlabel='Mean decrease in ROC-AUC',
            title='Random Forest Permutation Importance'
        )

        f.tight_layout()

        f.savefig(
            FIG / 'random_forest_permutation_importance.png',
            dpi=220
        )

        plt.close(f)

    # Logistic Regression coefficients
    else:

        features = (
            pipe.named_steps['preprocess']
            .get_feature_names_out()
        )

        coef = pd.DataFrame({
            'feature': features,
            'coefficient': (
                pipe.named_steps['model'].coef_[0]
            )
        })

        coef['abs'] = coef['coefficient'].abs()

        coef = (
            coef.nlargest(12, 'abs')
            .sort_values('coefficient')
        )

        coef.to_csv(
            TAB / 'logistic_coefficients.csv',
            index=False
        )

        f, a = plt.subplots(figsize=(9, 6))

        a.barh(
            coef['feature'],
            coef['coefficient']
        )

        a.set(
            title='Logistic Regression Coefficients',
            xlabel='Coefficient'
        )

        f.tight_layout()

        f.savefig(
            FIG / 'logistic_top_coefficients.png',
            dpi=220
        )

        plt.close(f)


# ==========================================
# 11. Save ROC curves
# ==========================================

ax.plot(
    [0, 1],
    [0, 1],
    'k--'
)

ax.set(
    xlabel='False Positive Rate',
    ylabel='True Positive Rate',
    title='Test ROC Curves'
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIG / 'roc_curves.png',
    dpi=220
)

plt.close(fig)


# ==========================================
# 12. Save model results
# ==========================================

results_df = pd.DataFrame(results)

print(
    results_df.round(4).to_string(index=False)
)

results_df.to_csv(
    TAB / 'model_metrics.csv',
    index=False
)

print(f'Completed. Open {FIG}')
