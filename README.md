# ML Teaching Notebooks

A collection of end-to-end, step-by-step machine learning notebooks for students, originally developed under the supervision of Dr. Rao Muhammad Adeel Nawab. Each project follows the same teaching structure (Import Libraries -> Load Data -> Preprocess -> Encode -> Train -> Test -> Deploy -> Collect Feedback) so students can compare approaches across problem types.

## Projects

### 1. Titanic Passenger Survival Prediction (Binary Classification)

Predicts whether a passenger survived the Titanic disaster from PClass, Gender, Sibling count, and Embarked port, using a Support Vector Classifier.

All files live in one folder, [`1 - Titanic Survival Prediction`](./1%20-%20Titanic%20Survival%20Prediction), since they share the same dataset, trained model, and `requirements.txt`. Two notebooks are included inside it:

| Notebook | Description |
|---|---|
| `Titanic_Passenger_Survival_Recommended.ipynb` | The original notebook with fixes: proper evaluation metrics (precision/recall/F1/confusion matrix) and a filled-in feedback section. Minimal changes, same modeling approach. |
| `Titanic_Passenger_Survival_FullEnhancement.ipynb` | Everything in the Recommended notebook, plus a data visualization step (survival rate by Gender/PClass) and a model comparison step (SVC vs. Logistic Regression vs. Decision Tree). |

Both achieve **0.80 accuracy** on the held-out test set and run top-to-bottom with zero errors.

### 3. Car Price Prediction with K-Fold Cross Validation (Regression)

Compares **5 regression algorithms** (Linear Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Gradient Boosting) for predicting a car's price (MSRP), using **K-fold cross validation** and **Root Mean Squared Error (RMSE)**. Each notebook ends with per-fold and summary comparison tables, charts, and the best model trained on all the data.

All files live in [`3 - Car Price Prediction K-Fold Cross Validation`](./3%20-%20Car%20Price%20Prediction%20K-Fold%20Cross%20Validation). The notebooks are designed for **Google Colab**:

| Notebook | Data | Validation | Open |
|---|---|---|---|
| `car_price_kfold_100_instances_3fold.ipynb` | 100 instances | 3-fold | [Open in Colab](https://colab.research.google.com/github/haqnawaz99/ml-teaching-notebooks/blob/main/3%20-%20Car%20Price%20Prediction%20K-Fold%20Cross%20Validation/car_price_kfold_100_instances_3fold.ipynb) |
| `car_price_kfold_full_dataset_10fold.ipynb` | Full dataset (~11,900 cars) | 10-fold | [Open in Colab](https://colab.research.google.com/github/haqnawaz99/ml-teaching-notebooks/blob/main/3%20-%20Car%20Price%20Prediction%20K-Fold%20Cross%20Validation/car_price_kfold_full_dataset_10fold.ipynb) |

To run: open a notebook in Colab, click **Runtime → Run all**, and upload [`car_price.csv`](./3%20-%20Car%20Price%20Prediction%20K-Fold%20Cross%20Validation/car_price.csv) when Step 2.1 asks for it. The trained `.pkl` model is created when the notebook runs, so it is not stored in this repo.

| Algorithm | Mean RMSE – 100 cars, 3-fold | Mean RMSE – full data, 10-fold |
|---|---|---|
| Decision Tree | **$22,317** | $15,595 |
| Gradient Boosting | $25,122 | $18,675 |
| Random Forest | $30,100 | **$15,362** |
| K-Nearest Neighbors | $34,584 | $23,680 |
| Linear Regression | $40,466 | $33,534 |

With more data, every algorithm's error drops, and the best algorithm changes from Decision Tree to Random Forest.

### 5. English Emotion Analysis – Binary Classification with Imbalanced Data

Detects whether an English comment expresses **Joy** or **Not Joy**. The data is **imbalanced on purpose** (815 Joy vs 1,585 Not Joy, about 1 : 2), so the notebook teaches how to work with biased data:

- a **baseline** that always predicts *Not Joy*: **66% accuracy, but it finds 0 Joy comments** (the accuracy trap)
- a **stratified** split, shown side by side with a non-stratified one
- **Precision, Recall and F1-Score of the Joy class** instead of accuracy alone
- **two variations** of the algorithms: without class weights and with `class_weight='balanced'`

| Algorithm | Recall (Joy): no weights → balanced | F1 (Joy): no weights → balanced |
|---|---|---|
| Logistic Regression | 0.35 → **0.77** | 0.51 → **0.77** |
| Linear SVM | 0.69 → 0.75 | 0.75 → 0.77 |
| Decision Tree | 0.78 → 0.82 | 0.81 → **0.81 (best)** |
| Random Forest | 0.52 → 0.37 | 0.65 → 0.51 |

Class weights helped three algorithms a lot, but made Random Forest worse, which shows why both variations must be tested.

| Notebook | Open |
|---|---|
| `English_Emotion_Analysis_BinaryClass.ipynb` | [Open in Colab](https://colab.research.google.com/github/haqnawaz99/ml-teaching-notebooks/blob/main/5%20-%20English%20Emotion%20Analysis%20-%20Binary%20Classification/English_Emotion_Analysis_BinaryClass.ipynb) |

To run: open in Colab, click **Runtime → Run all**, and upload [`emotion_binary_sample_data.csv`](./5%20-%20English%20Emotion%20Analysis%20-%20Binary%20Classification/emotion_binary_sample_data.csv) when Step 2 asks for it. The `.html` file in the folder shows the complete notebook with all outputs (no need to run anything).

### 6. English Emotion Analysis – Multi-class Classification

Predicts one of **six emotions** (sadness, joy, love, anger, fear, surprise) from an English comment and compares **six algorithms** on a **balanced** sample (400 comments per emotion, 2,400 in total), with an 80 / 20 stratified split and TF-IDF features (2,000 words). Each algorithm gets its own confusion matrix; the notebook ends with a comparison table, a bar chart and the best model predicting new comments.

| Rank | Algorithm | Accuracy | F1 (macro) |
|---|---|---|---|
| 1 | Linear SVM | **78.5%** | **0.781** |
| 2 | Logistic Regression | 78.1% | 0.777 |
| 3 | Random Forest | 77.7% | 0.775 |
| 4 | Decision Tree | 74.4% | 0.740 |
| 5 | Multinomial Naive Bayes | 70.6% | 0.698 |
| 6 | K-Nearest Neighbors | 55.8% | 0.553 |

| Notebook | Open |
|---|---|
| `English_Emotion_Analysis_MultiClass.ipynb` | [Open in Colab](https://colab.research.google.com/github/haqnawaz99/ml-teaching-notebooks/blob/main/6%20-%20English%20Emotion%20Analysis%20-%20Multi-class%20Classification/English_Emotion_Analysis_MultiClass.ipynb) |

To run: open in Colab, click **Runtime → Run all**, and upload [`emotion_sample_data.csv`](./6%20-%20English%20Emotion%20Analysis%20-%20Multi-class%20Classification/emotion_sample_data.csv) when Step 2 asks for it.

**Dataset (projects 5 and 6):** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) – Saravia et al. (2018), *CARER: Contextualized Affect Representations for Emotion Recognition*, EMNLP. For educational and research use. The trained models (`.pkl`) are created when the notebooks run, so they are not stored in this repo.

## Try It Online (No Setup Required)

A live demo of the Titanic survival predictor is deployed on Streamlit Community Cloud — students can try it directly in a browser, no Python or Jupyter install needed:

**[Titanic Survival Predictor — Live App](https://ml-teaching-notebooks-jrrdwercxskw5kwc6rufvp.streamlit.app/)**

The app code lives in [`streamlit-app/`](./streamlit-app) and uses the trained model from the Recommended notebook.

## Getting Started (each project folder)

Every project folder is self-contained with its own `requirements.txt`. From inside a project folder:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# Windows (cmd):
venv\Scripts\activate.bat
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

Full setup instructions (including troubleshooting) are also included as the first section inside each notebook. The car price notebooks (project 3) and the emotion notebooks (projects 5 and 6) run on Google Colab instead, so they need no local setup.

## Roadmap

More projects will be added here as they are finished:
- English Sentiment Analysis (Binary Classification)
- Car Price Prediction (Regression), single train/test split version
