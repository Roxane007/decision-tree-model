"""Clinic no-show decision tree (v1).

Reproduces the stratified 70/30 split, the classification tree, test-set fit
against the hold-every-slot baseline, the overbook/hold call for appointments
D and E, and the dollar value of each policy on the test appointments.

Run from the repository root:

    .venv/bin/python analysis.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "clinic_appointments.csv"
OUTPUT_DIR = ROOT / "output"

# Positive class is a missed appointment. Precision and recall are on no-shows.
SHOW = "checked-in"
NO_SHOW = "no-show"
LABELS = [SHOW, NO_SHOW]

# Offered to the tree. Appointment hour is left out: on the training rows the
# no-show rate does not move with the hour, so it is not a useful split.
FEATURES = [
    "lead_time_days",
    "prior_no_shows",
    "age",
    "distance_miles",
    "sms_reminder",
    "new_patient",
]

# Depth 3 is enough to separate the two groups worth overbooking (three or
# more past misses, and one or two past misses with no text reminder) without
# fitting a rule to a handful of training rows. Each leaf must hold at least
# 10 training appointments.
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 10
RANDOM_STATE = 42
TEST_SIZE = 0.30

# Value of one test appointment under the cost table. Positive class = no-show.
#   hold / showed            $0
#   overbook / showed        -$60  (overtime and a refunded copay)
#   hold / missed            -$120 (empty slot)
#   overbook / missed        $0    (the second patient fills the slot)
COST_TRUE_NEGATIVE = 0
COST_FALSE_POSITIVE = -60
COST_FALSE_NEGATIVE = -120
COST_TRUE_POSITIVE = 0

# The file does not state annual volume. 400 rows is about one month at
# 20 appointments a day. 20 a day, 5 days a week, 50 weeks a year = 5,000.
ASSUMED_ANNUAL_APPOINTMENTS = 5000

# Fitted in main() before slot_decision is called.
MODEL: DecisionTreeClassifier | None = None


def load_appointments(path: Path = DATA_PATH) -> pd.DataFrame:
    appointments = pd.read_csv(path)
    expected = FEATURES + ["appointment_hour", "no_show"]
    missing = [column for column in expected if column not in appointments.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if appointments[expected].isna().any().any():
        raise ValueError("Appointments file has missing values.")
    return appointments


def fit_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Confusion counts treat a no-show as the positive class."""
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    total_value = (
        COST_TRUE_NEGATIVE * int(true_negative)
        + COST_FALSE_POSITIVE * int(false_positive)
        + COST_FALSE_NEGATIVE * int(false_negative)
        + COST_TRUE_POSITIVE * int(true_positive)
    )
    count = int(len(y_true))
    return {
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "tp": int(true_positive),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_no_show": precision_score(
            y_true, y_pred, pos_label=NO_SHOW, zero_division=0
        ),
        "recall_no_show": recall_score(
            y_true, y_pred, pos_label=NO_SHOW, zero_division=0
        ),
        "total_value": total_value,
        "value_per_appointment": total_value / count,
        "n": count,
    }


def slot_decision(appointment: dict) -> str:
    """Overbook the slot if v1 predicts a no-show. Hold it if v1 predicts a show."""
    if MODEL is None:
        raise RuntimeError("Fit v1 before scoring an appointment.")
    row = pd.DataFrame([{feature: appointment[feature] for feature in FEATURES}])
    predicted = MODEL.predict(row)[0]
    if predicted == NO_SHOW:
        return "overbook"
    return "hold"


def leaf_report(appointment: dict) -> dict:
    """Path from the root to the leaf, and the training counts in that leaf."""
    if MODEL is None:
        raise RuntimeError("Fit v1 before describing a leaf.")
    row = pd.DataFrame([{feature: appointment[feature] for feature in FEATURES}])
    tree = MODEL.tree_
    node_id = int(MODEL.apply(row)[0])
    steps = []
    for node in MODEL.decision_path(row).indices:
        feature_index = int(tree.feature[node])
        if feature_index < 0:
            continue
        feature = FEATURES[feature_index]
        threshold = float(tree.threshold[node])
        value = appointment[feature]
        comparison = "<=" if value <= threshold else ">"
        steps.append(f"{feature} {value:g} {comparison} {threshold:.2f}")
    class_index = {name: index for index, name in enumerate(MODEL.classes_)}
    # sklearn stores class proportions in tree.value, not raw counts.
    training_n = int(tree.n_node_samples[node_id])
    proportions = tree.value[node_id][0]
    showed = int(round(proportions[class_index[SHOW]] * training_n))
    missed = training_n - showed
    probability = MODEL.predict_proba(row)[0]
    return {
        "leaf_id": node_id,
        "steps": steps,
        "training_showed": showed,
        "training_missed": missed,
        "training_n": training_n,
        "no_show_rate": missed / training_n,
        "predicted": MODEL.predict(row)[0],
        "decision": slot_decision(appointment),
        "probabilities": {
            name: float(probability[index]) for index, name in enumerate(MODEL.classes_)
        },
    }


def training_rates(train: pd.DataFrame) -> None:
    """Why these features: no-show rate on the training appointments only."""
    print("Training no-show rate by the features that can change a booking")
    outcome = (train["no_show"] == NO_SHOW).astype(int)
    for feature in ["prior_no_shows", "sms_reminder", "new_patient"]:
        grouped = (
            pd.DataFrame({feature: train[feature], "no_show": outcome})
            .groupby(feature)["no_show"]
            .agg(["mean", "count"])
        )
        print(f"\n{feature}")
        print(grouped.to_string(float_format=lambda value: f"{value:.3f}"))
    by_hour = (
        pd.DataFrame({"appointment_hour": train["appointment_hour"], "no_show": outcome})
        .groupby("appointment_hour")["no_show"]
        .agg(["mean", "count"])
    )
    print("\nappointment_hour (not used: the rate stays near the base rate all day)")
    print(by_hour.to_string(float_format=lambda value: f"{value:.3f}"))


def print_metrics(name: str, metrics: dict) -> None:
    print(
        f"{name}: TN {metrics['tn']}  FP {metrics['fp']}  "
        f"FN {metrics['fn']}  TP {metrics['tp']}"
    )
    print(
        f"  accuracy {metrics['accuracy']:.3f}   "
        f"precision on no-shows {metrics['precision_no_show']:.3f}   "
        f"recall on no-shows {metrics['recall_no_show']:.3f}"
    )
    print(
        f"  total value ${metrics['total_value']:,.0f}   "
        f"per appointment ${metrics['value_per_appointment']:,.2f}   "
        f"n {metrics['n']}"
    )


def save_tree_figure(model: DecisionTreeClassifier) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(16, 8))
    plot_tree(
        model,
        feature_names=FEATURES,
        class_names=list(model.classes_),
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=9,
        ax=axis,
    )
    axis.set_title("v1 classification tree, fit on the training appointments")
    figure.tight_layout()
    path = OUTPUT_DIR / "decision_tree.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    global MODEL

    appointments = load_appointments()
    no_show_count = int((appointments["no_show"] == NO_SHOW).sum())
    print("Data")
    print(
        f"  {len(appointments)} rows, {appointments.shape[1]} columns, "
        f"{no_show_count} no-shows ({no_show_count / len(appointments):.1%})"
    )

    train, test = train_test_split(
        appointments,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=appointments["no_show"],
    )
    print("\nSplit (stratified, random_state=42)")
    print(
        f"  train {len(train)} ({(train['no_show'] == NO_SHOW).mean():.1%} no-show, "
        f"n={(train['no_show'] == NO_SHOW).sum()})"
    )
    print(
        f"  test  {len(test)} ({(test['no_show'] == NO_SHOW).mean():.1%} no-show, "
        f"n={(test['no_show'] == NO_SHOW).sum()})"
    )
    print()
    training_rates(train)

    model = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )
    model.fit(train[FEATURES], train["no_show"])
    MODEL = model

    print("\nTree")
    print(
        f"  classification tree, max_depth={MAX_DEPTH}, "
        f"min_samples_leaf={MIN_SAMPLES_LEAF}, criterion=gini"
    )
    print(f"  features offered: {', '.join(FEATURES)}")
    print(f"  grown depth {model.get_depth()}, leaves {model.get_n_leaves()}")
    print("\n" + export_text(model, feature_names=FEATURES, decimals=2, show_weights=True))
    print("Split importance (share of impurity reduction)")
    for feature, importance in sorted(
        zip(FEATURES, model.feature_importances_), key=lambda item: -item[1]
    ):
        print(f"  {feature:20} {importance:.3f}")

    figure_path = save_tree_figure(model)
    print(f"\nTree figure: {figure_path.relative_to(ROOT)}")

    baseline_test = pd.Series([SHOW] * len(test), index=test.index)
    v1_test = model.predict(test[FEATURES])
    v1_train = model.predict(train[FEATURES])
    baseline_metrics = fit_metrics(test["no_show"], baseline_test)
    v1_test_metrics = fit_metrics(test["no_show"], v1_test)
    train_accuracy = accuracy_score(train["no_show"], v1_train)

    print("\nFit on the test appointments")
    print("  Positive class is no-show.")
    print("  TN showed and held, FP showed and overbooked,")
    print("  FN missed and held, TP missed and overbooked.")
    print_metrics("baseline (predict show for every test appointment)", baseline_metrics)
    print_metrics("v1", v1_test_metrics)
    print(
        f"v1 training accuracy {train_accuracy:.3f}   "
        f"v1 test accuracy {v1_test_metrics['accuracy']:.3f}"
    )

    upcoming = {
        "D": {
            "lead_time_days": 21,
            "prior_no_shows": 1,
            "age": 27,
            "distance_miles": 9.5,
            "sms_reminder": 0,
            "new_patient": 1,
            "appointment_hour": 15,
        },
        "E": {
            "lead_time_days": 3,
            "prior_no_shows": 0,
            "age": 61,
            "distance_miles": 2.0,
            "sms_reminder": 1,
            "new_patient": 0,
            "appointment_hour": 9,
        },
    }
    print("\nDecisions")
    for name, appointment in upcoming.items():
        report = leaf_report(appointment)
        print(f"  {name}: decision {report['decision']}  (predicted {report['predicted']})")
        print(f"     leaf {report['leaf_id']}")
        for step in report["steps"]:
            print(f"     {step}")
        print(
            f"     training leaf: {report['training_showed']:.0f} checked-in, "
            f"{report['training_missed']:.0f} no-show "
            f"({report['no_show_rate']:.1%} no-show, n={report['training_n']})"
        )

    gain_per_appointment = (
        v1_test_metrics["value_per_appointment"] - baseline_metrics["value_per_appointment"]
    )
    annual_gain = gain_per_appointment * ASSUMED_ANNUAL_APPOINTMENTS
    print("\nValue on the test appointments")
    print(
        f"  baseline ${baseline_metrics['total_value']:,.0f} total, "
        f"${baseline_metrics['value_per_appointment']:,.2f} per appointment"
    )
    print(
        f"  v1       ${v1_test_metrics['total_value']:,.0f} total, "
        f"${v1_test_metrics['value_per_appointment']:,.2f} per appointment"
    )
    print(f"  v1 minus baseline: ${gain_per_appointment:,.2f} per appointment")
    print(
        f"  assumed volume: {ASSUMED_ANNUAL_APPOINTMENTS:,} appointments/year "
        f"(20/day × 5 days × 50 weeks)"
    )
    print(f"  implied annual gain versus holding every slot: ${annual_gain:,.0f}")


if __name__ == "__main__":
    main()
