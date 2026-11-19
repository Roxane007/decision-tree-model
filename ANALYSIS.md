# Clinic no-shows: should this slot be overbooked?

The clinic loses an empty slot when it holds a booking and the patient does not come (−$120). It loses overtime and a refunded copay when it overbooks and the patient does come (−$60). Filling an overbooked slot that the original patient misses, or holding a slot the patient keeps, is worth $0. The current policy holds every slot, which is the same as predicting that every patient will show.

v1 is one classification tree. On appointments it never saw, overbooking the slots it flags as no-shows loses $10.50 per appointment instead of $27.00. That is $16.50 better per appointment than holding every slot. At an assumed 5,000 appointments a year, that gap is about $82,500 a year. The file does not state the clinic's volume; 5,000 is 20 appointments a day, 5 days a week, 50 weeks a year, and 400 rows is about one month at that pace.

`analysis.py` reproduces every figure below. The tree drawing is `output/decision_tree.png`. Slides are not part of this write-up.

## Data and split

`data/clinic_appointments.csv` has 400 appointments and 8 columns, with no missing values. 91 patients did not show (22.8%).

The outcome is `no_show`: `no-show` or `checked-in`. The tree was offered lead time, prior no-shows, age, distance, whether a text reminder was sent, and whether the patient is new. Appointment hour was left out. On the training rows the no-show rate by hour stays between about 17% and 31%, around the 23% base rate, so the hour does not separate shows from no-shows.

Thirty percent of the rows were held out as the test set, stratified on `no_show`, with `random_state=42`. The tree was fit only on the other 70%.

| Part | Appointments | No-shows | No-show rate |
|---|---:|---:|---:|
| Training | 280 | 64 | 22.9% |
| Test | 120 | 27 | 22.5% |

On the training rows, prior misses and a missing text are the rates that move:

| Prior no-shows | Training no-show rate | Appointments |
|---:|---:|---:|
| 0 | 7.2% | 138 |
| 1 | 27.7% | 94 |
| 2 | 41.9% | 31 |
| 3 | 87.5% | 16 |
| 4 | 100% | 1 |

No text reminder: 41.5% no-show (118 appointments). Text sent: 9.3% (162). New patients miss more often than returning patients (34.9% vs 17.8%), but the tree never needs that split once prior misses and the reminder are known.

## The tree

v1 is a scikit-learn classification tree, Gini splits, `max_depth=3`, `min_samples_leaf=10`. Depth 3 is the depth that separates the two groups worth overbooking: patients with 3 or more past misses, and patients with 1 or 2 past misses who were not texted. A leaf must contain at least 10 training appointments, so the tree cannot cut a rule out of a handful of rows.

It grew to depth 3 and 5 leaves. Counts below are training appointments, in the order [checked-in, no-show]. In the drawing, the left branch is the `<=` side.

1. **Root: `prior_no_shows <= 2.5`.** Three or more past misses (17 appointments: 2 showed, 15 missed, 88% no-show) are predicted no-shows. Everyone else goes on.
2. **Next, for patients with 0–2 past misses: `sms_reminder <= 0.5`.** No text versus a text.
3. **No text: `prior_no_shows <= 0.5`.** Zero past misses and no text still mostly show (56 appointments: 46 showed, 10 missed), so the tree holds the slot. One or two past misses and no text are predicted no-shows (52 appointments: 23 showed, 29 missed, 56% no-show).
4. **Text sent: `age <= 18.5`.** Both sides are still predicted shows. Patients 18 or younger: 9 showed and 5 missed (14 appointments). Older patients who were texted: 136 showed and 5 missed (141 appointments, 3.5% no-show). Age does not change the booking decision.

`prior_no_shows` accounts for 62.7% of the impurity reduction, `sms_reminder` 30.2%, and `age` 7.1%. Lead time, distance, and new-patient status were available and were not split on.

The operational rule is therefore: **overbook if the patient has missed 3 or more past appointments, or has missed at least one and was not sent a reminder. Hold the slot otherwise.** A reminder is the clinic's own action, so the tree is also saying that texting the patients who have missed before is what keeps those slots from being overbooked.

## Fit on the test appointments

The baseline predicts `checked-in` for all 120 test appointments. Positive class is `no-show`.

|  | TN (showed, held) | FP (showed, overbooked) | FN (missed, held) | TP (missed, overbooked) | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 93 | 0 | 27 | 0 | 0.775 (93/120) | 0.000 | 0.000 (0/27) |
| v1 | 86 | 7 | 7 | 20 | 0.883 (106/120) | 0.741 (20/27) | 0.741 (20/27) |

Baseline precision is 0 because the policy never predicts a no-show, so there are no positive predictions to be right about. Recall is 0 because all 27 missed appointments were held.

v1 training accuracy is 0.839 (235/280). Test accuracy is 0.883 (106/120). Training is not higher than test, so the tree did not memorize the training rows. The test set is only 120 appointments, so that four-point gap is ordinary sample noise.

Of the 27 test no-shows, v1 overbooks 20 and still holds 7. Of the 93 patients who showed, it overbooks 7. When it does predict a no-show, it is right 20 times out of 27.

## Decisions for D and E

`slot_decision` in `analysis.py` takes one appointment and returns `overbook` if v1 predicts `no-show`, otherwise `hold`.

**D — overbook.** Leaf 4. Path: `prior_no_shows` 1 ≤ 2.50, `sms_reminder` 0 ≤ 0.50, `prior_no_shows` 1 > 0.50. That leaf is the 52 training appointments with one or two past misses and no text: 23 showed, 29 missed (55.8% no-show). Lead time 21, age 27, distance 9.5, new patient, and the 3 p.m. hour do not enter the path.

**E — hold.** Leaf 7. Path: `prior_no_shows` 0 ≤ 2.50, `sms_reminder` 1 > 0.50, `age` 61 > 18.50. That leaf is the 141 training appointments who were texted and are older than 18: 136 showed, 5 missed (3.5% no-show).

## What the test set is worth

Each test appointment is scored from the confusion matrix and the cost table.

| Policy | Arithmetic | Total on 120 test appointments | Per appointment |
|---|---|---:|---:|
| Baseline | 27 empty slots × −$120 | −$3,240 | −$27.00 |
| v1 | 7 wrong overbooks × −$60, plus 7 empty slots × −$120 | −$1,260 | −$10.50 |

v1 is worth $16.50 more per appointment than holding every slot ($1,980 across these 120 appointments). Scaled to the assumed 5,000 appointments a year, the same per-appointment gap is about $82,500 a year, provided new bookings look like this test set. Both policies still lose money relative to a perfect forecast. v1 loses less because it avoids most of the empty slots and only occasionally pays the overtime-and-refund cost.
