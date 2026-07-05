Customer Churn Prediction in Telecom Industry

A machine learning project that predicts customer churn for a telecom company using the IBM Telco Customer Churn dataset. The project covers the complete ML pipeline — data cleaning, exploratory data analysis, model training, evaluation, and deployment via an interactive Streamlit dashboard.
🔗 Live Demo
https://customer-churn-prediction-wugmeeus2jeveh4zqdnf45.streamlit.app/

📌 Problem Statement

Customer churn is a major challenge in the telecom industry, where retaining existing customers is more cost-effective than acquiring new ones. This project builds a predictive system that identifies customers likely to churn, enabling proactive retention strategies.


🎯 Objectives
* Analyze customer data to uncover patterns behind churn.
* Train and compare multiple machine learning models.
* Identify the best-performing model based on key metrics.
* Deploy an interactive dashboard for real-time insights

📊 Dataset
* Source: IBM Telco Customer Churn Dataset
* Description: Contains customer demographics, account information, and service usage details, with a target label indicating churn status

🛠 Tech Stack
* Language: Python
* Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
* Dashboard: Streamlit
* Deployment: Streamlit Cloud
* Version Control: Git & GitHub

🔍 Methodology
1. Data Cleaning & Preprocessing - handled missing values, encoded categorical variables, scaled numerical features
2. Exploratory Data Analysis (EDA) - churn distribution, box plots, correlation heatmap
3. Model Training - trained 8 models including Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting, and XGBoost
4. Model Evaluation - compared models using Accuracy, Precision, Recall, F1-score, AUCROC
5. Deployment - built a 5-page Streamlit dashboard (Home, EDA, Models, Results, Insights) 

🏆 Results
Gradient Boosting - Best AUC and F1-score
XGBoost - Strong performance, used for implementation walkthrough


Confusion matrices, ROC curves, and a full model comparison table are available in notebooks/outputs/ .

📁 Project Structure
customer-churn-prediction/ 
│
├── data/                         #Dataset files
├── notebooks/                # Jupyter notebooks for EDA & modeling
│       └── outputs/          # Confusion matrices, ROC curves, comparison table 
├── app.py                      #Streamlit dashboard 
├── requirements.txt       #Project dependencies 
└── README.md

⚙ How to Run Locally
git clone https://github.com/abarnamurugancommits/customer-churn-prediction.git cd customer-churn-prediction 
pip install -r requirements.txt 
streamlit run app.py

📈 Dashboard Features
* Home — project overview 
* EDA — interactive visualizations of churn patterns 
* Models — model architecture and training details 
* Results — performance metrics and comparisons 
* Insights — key business takeaways and recommendations

🚀 Future Scope
* Hyperparameter tuning for further performance gains 
* Real-time prediction via API integration
* Incorporating additional customer behavior data

👤 Author
Abarna.M
Hemalatha.R
B.Tech, Artificial Intelligence and Data Science

📝 License
This project is for academic purposes.
