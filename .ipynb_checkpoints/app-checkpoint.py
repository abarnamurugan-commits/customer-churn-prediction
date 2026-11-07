import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Churn Prediction Dataset.csv")
OUTPUTS_PATH = os.path.join(BASE_DIR, "notebooks", "outputs")

# ─── Load Data (cached) ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

df = load_data()

# ─── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.title("📡 Churn Predictor")
st.sidebar.markdown("**Telecom Industry · Mini Project**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    ["🏠 Home", "📊 EDA", "🤖 Models", "📈 Results", "💡 Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Team Members**")
st.sidebar.markdown("👩‍💻 Abarna M")
st.sidebar.markdown("👩‍💻 Hemalatha R")
st.sidebar.markdown("---")
st.sidebar.markdown("*Customer Churn Prediction*")
st.sidebar.markdown("*Telecom Industry — 2025*")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("📡 Customer Churn Prediction in Telecom Industry")
    st.markdown("### A Machine Learning Mini Project")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", "7,043")
    col2.metric("Features", "21")
    col3.metric("Best Accuracy", "79.53%")

    st.markdown("---")
    st.markdown("## 🎯 Project Objective")
    st.write("""
    Customer churn — when subscribers leave a service provider — is a critical challenge in the
    telecom industry. This project builds and compares **8 machine learning models** to predict
    whether a customer will churn, helping telecom companies take proactive retention steps.
    """)

    st.markdown("## 📁 Dataset Overview")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        - **Source:** Kaggle — Telco Customer Churn  
        - **Rows:** 7,043 customers  
        - **Columns:** 21 features  
        - **Target:** `Churn` (Yes / No)
        """)
    with col_b:
        st.markdown("""
        - **Demographics:** Gender, SeniorCitizen, Partner, Dependents  
        - **Services:** PhoneService, InternetService, StreamingTV…  
        - **Account:** Tenure, Contract, MonthlyCharges, TotalCharges  
        """)

    st.markdown("## 👥 Team")
    t1, t2 = st.columns(2)
    t1.info("👩‍💻 **Abarna M** — ML Models, Dashboard, EDA")
    t2.info("👩‍💻 **Hemalatha R** — Data Preprocessing, Presentation")

    st.markdown("## 🔍 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    # --- Churn Distribution ---
    st.subheader("1. Churn Distribution")
    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4))
    churn_counts = df['Churn'].value_counts()
    ax1[0].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%',
               colors=['#4CAF50', '#F44336'], startangle=90)
    ax1[0].set_title("Churn Proportion")
    sns.countplot(x='Churn', data=df, palette=['#4CAF50', '#F44336'], ax=ax1[1])
    ax1[1].set_title("Churn Count")
    st.pyplot(fig1)
    st.info("📌 ~26.5% of customers churned — the dataset is moderately imbalanced.")

    st.markdown("---")

    # --- Tenure vs Monthly Charges ---
    st.subheader("2. Tenure & Monthly Charges by Churn")
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(x='Churn', y='tenure', data=df, palette=['#4CAF50', '#F44336'], ax=ax2[0])
    ax2[0].set_title("Tenure by Churn")
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette=['#4CAF50', '#F44336'], ax=ax2[1])
    ax2[1].set_title("Monthly Charges by Churn")
    st.pyplot(fig2)
    st.info("📌 Churned customers tend to have shorter tenure and higher monthly charges.")

    st.markdown("---")

    # --- Contract Type ---
    st.subheader("3. Contract Type vs Churn")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.countplot(x='Contract', hue='Churn', data=df, palette=['#4CAF50', '#F44336'], ax=ax3)
    ax3.set_title("Contract Type vs Churn")
    st.pyplot(fig3)
    st.info("📌 Month-to-month contract customers churn far more than long-term contract holders.")

    st.markdown("---")

    # --- Correlation Heatmap ---
    st.subheader("4. Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
    ax4.set_title("Correlation Heatmap")
    st.pyplot(fig4)
    st.info("📌 Tenure and TotalCharges are highly correlated. MonthlyCharges correlates with churn.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Models":
    st.title("🤖 Machine Learning Models")
    st.markdown("---")
    st.write("8 classifiers were trained and evaluated on the Telco Churn dataset.")

    models_info = {
        "Logistic Regression": {
            "icon": "📐", "acc": "78.54%",
            "desc": "A linear model that estimates churn probability using a sigmoid function. Serves as a strong interpretable baseline.",
            "params": "max_iter=1000, solver=lbfgs"
        },
        "Support Vector Machine": {
            "icon": "📏", "acc": "79.10%",
            "desc": "Finds the optimal hyperplane that separates churners from non-churners with maximum margin.",
            "params": "kernel=rbf, C=1.0"
        },
        "Decision Tree": {
            "icon": "🌿", "acc": "72.64%",
            "desc": "A tree-structured model that splits data based on feature thresholds. Highly interpretable but prone to overfitting.",
            "params": "max_depth=5, criterion=gini"
        },
        "Random Forest": {
            "icon": "🌲", "acc": "79.18%",
            "desc": "An ensemble of decision trees using bagging. Reduces variance and improves accuracy over a single tree.",
            "params": "n_estimators=100, max_depth=None"
        },
        "K-Nearest Neighbors": {
            "icon": "🔵", "acc": "73.99%",
            "desc": "Classifies customers based on the majority label among their K nearest neighbors in feature space.",
            "params": "n_neighbors=5, metric=minkowski"
        },
        "Gradient Boosting": {
            "icon": "🚀", "acc": "79.53% ⭐",
            "desc": "Sequentially builds trees where each corrects errors of the previous. Best overall performer in this project.",
            "params": "n_estimators=100, learning_rate=0.1"
        },
        "XGBoost": {
            "icon": "⚡", "acc": "76.40%",
            "desc": "An optimized gradient boosting library with regularization. Fast and efficient for structured/tabular data.",
            "params": "n_estimators=100, use_label_encoder=False"
        },
        "Artificial Neural Network": {
            "icon": "🧠", "acc": "73.99%",
            "desc": "A multi-layer perceptron with hidden layers that learns non-linear patterns in the data.",
            "params": "hidden_layers=(100,50), activation=relu"
        },
    }

    for name, info in models_info.items():
        with st.expander(f"{info['icon']} {name} — Accuracy: {info['acc']}"):
            st.write(f"**Description:** {info['desc']}")
            st.code(f"Parameters: {info['params']}", language="python")

    st.markdown("---")
    st.markdown("### ⚙️ Common Setup")
    st.code("""
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
    """, language="python")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Results":
    st.title("📈 Results & Evaluation")
    st.markdown("---")

    # Model comparison table
    st.subheader("Model Comparison Table")
    results_data = {
        "Model": ["Logistic Regression", "SVM", "Decision Tree", "Random Forest",
                  "KNN", "Gradient Boosting", "XGBoost", "ANN"],
        "Accuracy (%)": [78.54, 79.10, 72.64, 79.18, 73.99, 79.53, 76.40, 73.99],
    }
    results_df = pd.DataFrame(results_data).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
    results_df.index += 1
    st.dataframe(results_df, use_container_width=True)

    # Saved image
    table_img_path = os.path.join(OUTPUTS_PATH, "model_comparison_table.png")
    if os.path.exists(table_img_path):
        st.image(table_img_path, caption="Model Comparison Table", use_container_width=True)

    st.markdown("---")

    # ROC Curve
    st.subheader("ROC Curve — All Models")
    roc_path = os.path.join(OUTPUTS_PATH, "roc_curve_all_models.png")
    if os.path.exists(roc_path):
        st.image(roc_path, caption="ROC Curve (All Models)", use_container_width=True)
    else:
        st.warning("ROC curve image not found. Make sure roc_curve_all_models.png is in notebooks/outputs/")

    st.markdown("---")

    # Bar Chart
    st.subheader("Accuracy Comparison — Bar Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#FF6B6B' if m == 'Gradient Boosting' else '#4C9BE8' for m in results_df['Model']]
    bars = ax.barh(results_df['Model'], results_df['Accuracy (%)'], color=colors)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison")
    ax.set_xlim(68, 82)
    for bar, val in zip(bars, results_df['Accuracy (%)']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val}%', va='center', fontsize=9)
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Insights":
    st.title("💡 Key Insights & Recommendations")
    st.markdown("---")

    st.markdown("## 🔑 Key Findings")
    st.success("✅ **Best Model:** Gradient Boosting with **79.53% accuracy**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **📊 Data Insights:**
        - ~26.5% of customers churned
        - Month-to-month contracts have highest churn
        - Customers with fiber optic internet churn more
        - Short-tenure customers are highest risk
        - Higher monthly charges → higher churn rate
        """)
    with col2:
        st.markdown("""
        **🤖 Model Insights:**
        - Gradient Boosting outperformed all models
        - Ensemble methods (RF, GB, XGB) consistently strong
        - Linear models (LR, SVM) performed comparably
        - Tree-based single models underperformed
        - ANN did not outperform classical ML here
        """)

    st.markdown("---")
    st.markdown("## 💼 Business Recommendations")

    recs = [
        ("🎯", "Target Month-to-Month Customers", "Offer incentives to convert them to annual or two-year contracts."),
        ("💰", "Review Pricing Strategy", "High monthly charges correlate with churn — consider loyalty discounts."),
        ("🕐", "Engage New Customers Early", "First 12 months are critical — proactive onboarding reduces churn risk."),
        ("🌐", "Improve Fiber Optic Service", "Fiber optic users churn more — investigate service quality issues."),
        ("🤖", "Deploy Churn Prediction Model", "Use Gradient Boosting model to score customers monthly and flag at-risk ones."),
    ]

    for icon, title, desc in recs:
        with st.expander(f"{icon} {title}"):
            st.write(desc)

    st.markdown("---")
    st.markdown("## 🏆 Model Performance Summary")
    st.markdown("""
    | Rank | Model | Accuracy |
    |------|-------|----------|
    | 🥇 1 | Gradient Boosting | 79.53% |
    | 🥈 2 | Random Forest | 79.18% |
    | 🥉 3 | SVM | 79.10% |
    | 4 | Logistic Regression | 78.54% |
    | 5 | XGBoost | 76.40% |
    | 6 | KNN | 73.99% |
    | 7 | ANN | 73.99% |
    | 8 | Decision Tree | 72.64% |
    """)

    st.markdown("---")
    st.info("📌 This project demonstrates that ensemble methods, especially Gradient Boosting, are highly effective for churn prediction in the telecom domain.")