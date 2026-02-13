# Lab 3: Contextual Bandit-Based News Article Recommendation System

**Course:** Reinforcement Learning Fundamentals  
**Student:** Varun S G  
**Roll Number:** U20230015  
**Branch:** `varun_U20230015`

---

## 1. Overview

This project implements a **News Recommendation System** using a **Contextual Multi-Armed Bandit (CMAB)** framework. The system classifies users into one of three contexts (User1, User2, User3) using a supervised learning model and then uses bandit algorithms to learn the optimal news category (Entertainment, Education, Tech, Crime) for each user type, maximising expected reward.

---

## 2. Approach

### 2.1 Data Pre-processing

- **News articles** (`news_articles.csv`): Dropped rows with missing headlines, filled missing `short_description` and `authors` with empty strings, then filtered to the four bandit-relevant categories (ENTERTAINMENT, EDUCATION, TECH, CRIME).
- **User datasets** (`train_users.csv`, `test_users.csv`): Filled missing `age` values with the training-set median, encoded boolean `subscriber` as integer, label-encoded `region_code`, dropped non-numeric columns (`user_id`, `browser_version`), and applied `StandardScaler` to all remaining numeric features.

### 2.2 User Classification (Context Detector)

- Split `train_users.csv` into 80% train / 20% validation (stratified).
- Trained a **Random Forest Classifier** (300 estimators, max depth 20) on the scaled features.
- Achieved **~90% accuracy** on the validation set.
- Retrained the final model on the full training set for deployment in the recommendation pipeline.

### 2.3 Contextual Bandit Algorithms

Three strategies were implemented, each treating user category as the **context** and news category as the **arm** (4 arms per context, 12 total arms):

1. **Epsilon-Greedy** — Tested with epsilon values of 0.01, 0.1, and 0.3.
2. **Upper Confidence Bound (UCB)** — Tested with exploration parameter C values of 0.5, 1.0, and 2.0.
3. **SoftMax (Boltzmann)** — Fixed temperature tau = 1.0.

Each algorithm was run for **T = 10,000 steps** per context, using the `rlcmab-sampler` package (initialised with roll number i = 15) to obtain rewards.

### 2.4 Recommendation Engine

An end-to-end pipeline that:
1. **Classifies** a new user into a context using the trained classifier.
2. **Selects** the best news category using the learned bandit policy (epsilon-greedy, epsilon = 0.1).
3. **Recommends** a randomly sampled article from the selected category.

### 2.5 Evaluation

- Classification accuracy reported via `sklearn.metrics.classification_report`.
- Average Reward vs. Time plots for each context and strategy.
- Hyperparameter comparison plots for epsilon-greedy (3 epsilon values) and UCB (3 C values).
- Summary bar chart and table comparing all strategies.

---

## 3. Key Results

| Strategy | User1 Avg Reward | User2 Avg Reward | User3 Avg Reward | Best Arm (all contexts) |
|---|---|---|---|---|
| Eps-Greedy (epsilon=0.01) | 10.97 | 4.40 | 5.30 | Crime |
| Eps-Greedy (epsilon=0.1) | 10.24 | 3.75 | 4.91 | Crime |
| Eps-Greedy (epsilon=0.3) | 8.17 | 2.23 | 3.77 | Crime |
| UCB (C=0.5) | 11.34 | 4.45 | 5.49 | Crime |
| UCB (C=1.0) | 11.34 | 4.46 | 5.47 | Crime |
| UCB (C=2.0) | 11.34 | 4.45 | 5.48 | Crime |
| SoftMax (tau=1.0) | 11.35 | 4.37 | 5.24 | Crime |

- All strategies converge to **Crime** as the optimal arm for every context.
- **UCB** and **SoftMax** achieve the highest average rewards, close to the theoretical optimum, thanks to more efficient exploration.
- **Epsilon-Greedy** is more sensitive to the choice of epsilon; lower epsilon gives higher reward but risks under-exploration.

---

## 4. Observations & Insights

### Strategy Comparison
- **Epsilon-Greedy**: Simple and effective. Lower epsilon values (0.01) converge faster to the best arm but risk getting stuck on a sub-optimal arm early. Higher epsilon (0.3) explores more but sacrifices reward.
- **UCB**: Balances exploration and exploitation automatically via the confidence bound. Converges reliably regardless of C, since exploration is driven by uncertainty rather than randomness.
- **SoftMax (tau=1)**: Weights exploration by estimated arm quality — arms with higher Q-values are pulled more often. Avoids the hard randomness of epsilon-greedy while still exploring sub-optimal arms.

### Hyperparameter Sensitivity
- **epsilon (Eps-Greedy)**: Very sensitive. Too low leads to under-exploration; too high leads to over-exploration. epsilon = 0.1 is a good practical default.
- **C (UCB)**: Moderate sensitivity. C = 1.0 is a reasonable default. Larger C values lead to more uniform exploration.
- **tau (SoftMax)**: tau = 1.0 is a balanced default. Lower tau behaves more greedily; higher tau approaches random selection.

### Key Takeaways
1. All three strategies successfully identify the best arm per context given T = 10,000 steps.
2. UCB and SoftMax generally provide smoother convergence compared to epsilon-greedy.
3. The contextual structure (separate bandit per user type) allows the system to learn different optimal categories for different user segments.
4. The recommendation engine successfully chains classification, bandit policy lookup, and article sampling into an end-to-end pipeline.

---