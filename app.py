import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    load_data,
    get_summary,
    preprocess_data,
    train_and_save_model,
    load_model,
    evaluate_model
)
from sklearn.model_selection import train_test_split

# Set dark theme for Streamlit
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .stApp {
            background-color: #0e1117;
        }
    </style>
""", unsafe_allow_html=True)

# Load the Titanic dataset
df = load_data()

# Sidebar navigation
st.sidebar.title("\U0001F4C1 Navigation")
main_selection = st.sidebar.radio("Go to", ["Dataset Overview", "Dataset Summary", "Prediction", "EDA", "Model Evaluation", "Conclusion"])

# If Dataset Summary is selected, show dropdown options inside
if main_selection == "Dataset Summary":
    st.title("\U0001F4CA Dataset Summary")
    sub_selection = st.selectbox("Choose detail to view", [
        "Data Types",
        "Missing Values",
        "Unique Value Counts",
        "Summary Statistics"
    ])

# === Preprocess Data & Prepare Model (Behind the Scenes) ===
X, y, scaler = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
try:
    model = load_model()
except FileNotFoundError:
    model = train_and_save_model(X_train, y_train)

# === Main Page Based on Navigation Selection ===
if main_selection == "Dataset Overview":
    st.title("\U0001F6A2 Titanic Dataset Project")
    st.header("Dataset Overview")
    st.dataframe(df.head(8))

elif main_selection == "Dataset Summary":
    summary = get_summary(df)
    if sub_selection == "Data Types":
        st.subheader("Column Data Types")
        dt = summary['data_types'].reset_index()
        dt.columns = ["Column", "DataType"]
        st.table(dt)
    elif sub_selection == "Missing Values":
        st.subheader("Missing Values")
        missing_df = summary['missing_values'].reset_index()
        missing_df.columns = ["Column", "Missing Value"]
        st.table(missing_df)
    elif sub_selection == "Unique Value Counts":
        st.subheader("Unique Value Counts")
        unique_df = summary['unique_counts'].reset_index()
        unique_df.columns = ["Column", "Unique Value"]
        st.table(unique_df)
    elif sub_selection == "Summary Statistics":
        st.subheader("Summary Statistics")
        st.table(summary['summary_stats'])

elif main_selection == "Prediction":
    st.title("\U0001F9E0 Titanic Survival Prediction")
    st.subheader("Enter passenger details:")
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 1, 80, 25)
    sibsp = st.number_input("Siblings / Spouses Aboard", 0, 8, 0)
    parch = st.number_input("Parents / Children Aboard", 0, 6, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 32.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
    sex_encoded = 0 if sex == "male" else 1
    embarked_Q = 1 if embarked == "Q" else 0
    embarked_S = 1 if embarked == "S" else 0
    input_dict = {
        'Pclass': pclass,
        'Sex': sex_encoded,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked_Q': embarked_Q,
        'Embarked_S': embarked_S
    }
    input_df = pd.DataFrame([input_dict])
    num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    prediction = model.predict(input_df)[0]
    result = "\u2705 Survived" if prediction == 1 else "\u274C Did Not Survive"
    if st.button("Predict"):
        st.success(f"Prediction: {result}")

elif main_selection == "EDA":
    st.title("\U0001F4CA Exploratory Data Analysis")
    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid", palette="Set2")
    eda_option = st.selectbox("Select an EDA visualization", [
        "Survival Count", "Age Distribution", "Fare Distribution",
        "Survival by Sex", "Survival by Pclass", "Boxplot: Age vs Pclass",
        "Correlation Heatmap", "Missing Value Heatmap", "Survival by Embarked"
    ])
    fig, ax = plt.subplots()
    if eda_option == "Survival Count":
        df['Survived'].value_counts().plot(kind='bar', ax=ax, color=['#FF7F0E', '#1F77B4'])
        ax.set_xticklabels(['Did Not Survive', 'Survived'])
    elif eda_option == "Age Distribution":
        df['Age'].hist(bins=30, edgecolor='white', color='#2ca02c', ax=ax)
    elif eda_option == "Fare Distribution":
        df['Fare'].hist(bins=40, edgecolor='white', color='#d62728', ax=ax)
    elif eda_option == "Survival by Sex":
        pd.crosstab(df['Sex'], df['Survived']).plot(kind='bar', ax=ax)
    elif eda_option == "Survival by Pclass":
        pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar', ax=ax)
    elif eda_option == "Boxplot: Age vs Pclass":
        sns.boxplot(data=df, x='Pclass', y='Age', ax=ax)
    elif eda_option == "Correlation Heatmap":
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='Spectral', fmt=".2f", ax=ax)
    elif eda_option == "Missing Value Heatmap":
        sns.heatmap(df.isnull(), cbar=False, cmap="mako", ax=ax)
    elif eda_option == "Survival by Embarked":
        pd.crosstab(df['Embarked'], df['Survived']).plot(kind='bar', ax=ax)
    st.pyplot(fig)
    plt.close(fig)

elif main_selection == "Model Evaluation":
    st.title("\U0001F4C8 Model Evaluation on Test Data")
    acc, cm, report = evaluate_model(model, X_test, y_test)
    st.subheader("\u2705 Accuracy")
    st.metric("Test Accuracy", f"{acc:.2%}")
    st.subheader("\U0001F9EE Confusion Matrix")
    cm_df = pd.DataFrame(cm, index=["Actual: 0", "Actual: 1"], columns=["Predicted: 0", "Predicted: 1"])
    st.table(cm_df)
    st.subheader("\U0001F4CB Classification Report")
    st.text(report)

elif main_selection == "Conclusion":
    st.title("\U0001F4DD Project Conclusion")
    st.markdown("""
        ### Key Takeaways
        - Titanic survival rates are affected by **gender**, **class**, and **fare**.
        - Logistic Regression reached **79.89% accuracy**.
        - Key features: **Sex**, **Pclass**, and **Fare**.

        ### Potential Improvements
        - Try advanced models like **Random Forest** or **XGBoost**.
        - Engineer features like **Title** or **Family Size**.
        - Improve handling of missing values and apply cross-validation.

        ### Tools Used
        - Python, Pandas, Scikit-learn, Matplotlib, Streamlit

        ---
        **This concludes the Titanic survival prediction project.**
    """)
