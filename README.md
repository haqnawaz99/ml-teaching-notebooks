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

Full setup instructions (including troubleshooting) are also included as the first section inside each notebook.

## Roadmap

More projects will be added here as they are finished:
- English Sentiment Analysis (Binary Classification)
- Car Price Prediction (Regression)
