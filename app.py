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

# Set page layout
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

# Load the Titanic dataset
df = load_data()

# Sidebar navigation
st.sidebar.title("📁 Navigation")
main_selection = st.sidebar.radio("Go to", ["Dataset Overview", "Dataset Summary","Prediction","EDA","Model Evaluation","Conclusion"])

# If Dataset Summary is selected, show dropdown options inside
if main_selection == "Dataset Summary":

        st.title("📊 Dataset Summary")

        sub_selection = st.selectbox("Choose detail to view", [
            "Data Types",
            "Missing Values",
            "Unique Value Counts",
            "Summary Statistics"
        ])


# === Preprocess Data & Prepare Model (Behind the Scenes) ===
X, y, scaler = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

try:
    model = load_model()
except FileNotFoundError:
    model = train_and_save_model(X_train, y_train)

# === Main Page Based on Navigation Selection ===

if main_selection == "Dataset Overview":
    st.title("🚢 Titanic Dataset Project")
    st.header("Dataset Overview")
    st.dataframe(df.head())

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
        unique_df.columns= ["Column", "Unique Value"]
        st.table(unique_df)

    elif sub_selection == "Summary Statistics":
        st.subheader("Summary Statistics")
        sum_stat = summary['summary_stats']
        st.table(sum_stat)
elif main_selection == "Prediction":
    st.title("🧠 Titanic Survival Prediction")
    st.subheader("Enter passenger details:")

    pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 1, 80, 25)
    sibsp = st.number_input("Siblings / Spouses Aboard", 0, 8, 0)
    parch = st.number_input("Parents / Children Aboard", 0, 6, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 32.0)
    embarked = st.selectbox("Port of Embarkation (refers to where the passengers boarded from) [S==Southampton , C==Cherbourg , Q==Queenstown]", ["S", "C", "Q"])

    # Encode input
    sex_encoded = 0 if sex == "male" else 1
    embarked_C = 1 if embarked == "C" else 0
    embarked_Q = 1 if embarked == "Q" else 0

    # Input data in correct order
    input_data = [[pclass, sex_encoded, age, sibsp, parch, fare, embarked_C, embarked_Q]]

    # Create a DataFrame
    input_df = pd.DataFrame(input_data, columns=[
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q"
    ])

    # Scale only numerical columns
    num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predict using trained model
    prediction = model.predict(input_df)[0]
    result = "✅ Survived" if prediction == 1 else "❌ Did Not Survive"

    if st.button("Predict"):
        st.success(f"Prediction: {result}")

elif main_selection == "EDA":
    st.title("📊 Exploratory Data Analysis")

    eda_option = st.selectbox("Select an EDA visualization", [
        "Survival Count",
        "Age Distribution",
        "Fare Distribution",
        "Survival by Sex",
        "Survival by Pclass",
        "Boxplot: Age vs Pclass",
        "Correlation Heatmap",
        "Missing Value Heatmap",
        "Survival by Embarked",
        "Pairplot (selected features)"
    ])

    if eda_option == "Survival Count":
        st.subheader("Survival Count")
        fig, ax = plt.subplots()
        df['Survived'].value_counts().plot(kind='bar', ax=ax)
        ax.set_xticklabels(['Did Not Survive', 'Survived'], rotation=0)
        st.pyplot(fig)

    elif eda_option == "Age Distribution":
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        df['Age'].hist(bins=30, edgecolor='black', ax=ax)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif eda_option == "Fare Distribution":
        st.subheader("Fare Distribution")
        fig, ax = plt.subplots()
        df['Fare'].hist(bins=40, edgecolor='black', ax=ax)
        ax.set_xlabel("Fare")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif eda_option == "Survival by Sex":
        st.subheader("Survival by Sex")
        fig, ax = plt.subplots()
        pd.crosstab(df['Sex'], df['Survived']).plot(kind='bar', ax=ax)
        st.pyplot(fig)

    elif eda_option == "Survival by Pclass":
        st.subheader("Survival by Passenger Class")
        fig, ax = plt.subplots()
        pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar', ax=ax)
        st.pyplot(fig)

    elif eda_option == "Boxplot: Age vs Pclass":
        st.subheader("Age Distribution by Pclass")
        fig, ax = plt.subplots()
        df.boxplot(column='Age', by='Pclass', ax=ax)
        st.pyplot(fig)

    elif eda_option == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        df_corr = df.copy()
        df_corr['Sex'] = df_corr['Sex'].map({'male': 0, 'female': 1})
        df_corr['Embarked'] = df_corr['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        df_corr.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        corr = df_corr.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)


    elif eda_option == "Missing Value Heatmap":
        st.subheader("Missing Value Heatmap")
        import seaborn as sns
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

    elif eda_option == "Survival by Embarked":
        st.subheader("Survival by Port of Embarkation")
        fig, ax = plt.subplots()
        pd.crosstab(df['Embarked'], df['Survived']).plot(kind='bar', ax=ax)
        st.pyplot(fig)

    elif eda_option == "Pairplot (selected features)":
        st.subheader("Pairplot: Selected Features")
        import seaborn as sns
        fig = sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']].dropna(), hue='Survived')
        st.pyplot(fig)

elif main_selection == "Model Evaluation":
    st.title("📈 Model Evaluation on Test Data")

    # Evaluate the model
    acc, cm, report = evaluate_model(model, X_test, y_test)

    st.subheader("✅ Accuracy")
    st.metric("Test Accuracy", f"{acc:.2%}")

    st.subheader("🧮 Confusion Matrix")
    cm_df = pd.DataFrame(cm, index=["Actual: 0", "Actual: 1"], columns=["Predicted: 0", "Predicted: 1"])
    st.table(cm_df)

    st.subheader("📋 Classification Report")
    st.text(report)

elif main_selection == "Conclusion":
    st.title("📝 Project Conclusion")

    st.markdown("""
    ### Key Takeaways

    - The Titanic dataset shows strong survival patterns based on gender, class, and age.
    - Logistic Regression achieved **72% accuracy** — a good starting point.
    - Feature importance shows **Sex**, **Pclass**, and **Fare** are key drivers.
    - Simple preprocessing and scaling were sufficient for baseline performance.

    ### What Can Be Improved

    - Try advanced models like **Random Forest**, **XGBoost**, or **SVM**
    - Engineer new features (e.g., family size, titles from names)
    - Use cross-validation for better generalization
    - Handle missing data (Cabin) using imputation techniques

    ### Tools Used

    - Python, Pandas, Scikit-learn, Matplotlib, Streamlit

    ---
    **This concludes the Titanic survival prediction project.**
    """)


