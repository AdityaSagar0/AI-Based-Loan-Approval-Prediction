import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load("models/loan_model.pkl")

# Load dataset
data = pd.read_csv("data/loan_data.csv")

# Page configuration
st.set_page_config(
    page_title="Loan Approval AI",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        color: #12355b;
        font-size: 38px;
        font-weight: bold;
    }

    .sub-title {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .section-title {
        color: #12355b;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("🏦 Loan AI System")
st.sidebar.write("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🔮 Loan Prediction",
        "📊 Dataset Analysis",
        "🤖 Model Information",
        "ℹ️ About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("AI-Based Loan Approval Prediction System")

# Home page
if page == "🏠 Home":
    st.markdown(
        '<div class="main-title">🏦 AI-Based Loan Approval Prediction System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Machine Learning Based Loan Approval Prediction Platform</div>',
        unsafe_allow_html=True
    )

    st.image(
        "https://images.unsplash.com/photo-1556742049-0cfed4f6a45d",
        caption="Smart and AI-powered financial decision support",
        use_container_width=True
    )

    st.subheader("📌 Project Overview")

    st.write(
        """
        This project uses Machine Learning to predict whether a loan
        application is likely to be approved or rejected based on
        applicant information such as income, education, credit history,
        loan amount, and property area.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(data))

    with col2:
        st.metric("Input Features", len(data.columns) - 2)

    with col3:
        st.metric("ML Algorithm", "Random Forest")

# Loan prediction page
elif page == "🔮 Loan Prediction":
    st.markdown(
        '<div class="main-title">🔮 Loan Approval Prediction</div>',
        unsafe_allow_html=True
    )

    st.write("Enter applicant details below.")

    st.subheader("👤 Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])

    with col2:
        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000,
            step=100
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0,
            value=0,
            step=100
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            value=100,
            step=10
        )

        loan_term = st.selectbox(
            "Loan Term",
            [120, 180, 240, 300, 360, 480],
            index=4
        )

        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0],
            format_func=lambda x: "Good" if x == 1.0 else "Bad"
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    st.divider()

    if st.button("🔍 Predict Loan Approval", use_container_width=True):

        input_data = pd.DataFrame({
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area]
        })

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        predicted_class_index = list(model.classes_).index(prediction)
        confidence = probabilities[predicted_class_index] * 100

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success("✅ Loan is likely to be Approved!")
        else:
            st.error("❌ Loan is likely to be Rejected!")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(int(confidence))

        st.info(
            "This result is an AI-based prediction for academic purposes "
            "and is not an official bank decision."
        )

# Dataset analysis page
elif page == "📊 Dataset Analysis":
    st.markdown(
        '<div class="main-title">📊 Dataset Analysis</div>',
        unsafe_allow_html=True
    )

    st.write("Overview of the dataset used to train the model.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", data.shape[0])

    with col2:
        st.metric("Columns", data.shape[1])

    with col3:
        st.metric("Missing Values", int(data.isnull().sum().sum()))

    st.subheader("📋 Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)

    st.subheader("📈 Loan Status Distribution")

    if "Loan_Status" in data.columns:
        status_count = data["Loan_Status"].value_counts()
        st.bar_chart(status_count)

    st.subheader("📌 Dataset Statistics")
    st.dataframe(data.describe(include="all"), use_container_width=True)

# Model information page
elif page == "🤖 Model Information":
    st.markdown(
        '<div class="main-title">🤖 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Performance metrics of the Random Forest classification model."
    )

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    # Prepare dataset
    performance_data = data.copy()

    if "Loan_ID" in performance_data.columns:
        performance_data = performance_data.drop("Loan_ID", axis=1)

    X = performance_data.drop("Loan_Status", axis=1)
    y = performance_data["Loan_Status"].map({"Y": 1, "N": 0})

    # Use the same split configuration as training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")

    with col2:
        st.metric("Precision", f"{precision * 100:.2f}%")

    with col3:
        st.metric("Recall", f"{recall * 100:.2f}%")

    with col4:
        st.metric("F1 Score", f"{f1 * 100:.2f}%")

    #
    st.subheader("📈 Performance Comparison")

    metric_chart = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [accuracy, precision, recall, f1]
    })

    st.bar_chart(
        metric_chart.set_index("Metric"),
        y="Score"
    )

    st.divider()

    st.subheader("📉 Confusion Matrix Heatmap")

    matrix = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Loan Prediction Confusion Matrix")

    st.pyplot(fig)

    st.subheader("📌 Metric Explanation")

    st.write(
        """
        **Accuracy:** Percentage of total predictions that are correct.

        **Precision:** Percentage of predicted approvals that are actually approvals.

        **Recall:** Percentage of actual approvals correctly identified by the model.

        **F1 Score:** Combined measure of precision and recall.
        """
    )

    st.subheader("⚙️ Technologies Used")

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Joblib",
            "Streamlit"
        ],
        "Purpose": [
            "Programming Language",
            "Data Processing",
            "Numerical Computation",
            "Machine Learning",
            "Model Saving",
            "Web Interface"
        ]
    })

    st.table(technologies)

    st.subheader("🔄 Project Workflow")

    st.write(
        """
        1. Collect the loan dataset
        2. Clean and preprocess the data
        3. Handle missing values
        4. Encode categorical features
        5. Train the Random Forest model
        6. Evaluate the model
        7. Save the trained model
        8. Predict loan approval through Streamlit
        """
    )

# About project page
elif page == "ℹ️ About Project":
    st.markdown(
        '<div class="main-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )

    st.subheader("Project Title")
    st.write("AI-Based Loan Approval Prediction System")

    st.subheader("Objective")
    st.write(
        """
        The objective of this project is to develop a machine-learning
        application that predicts loan approval status using applicant
        financial and personal information.
        """
    )

    st.subheader("Key Features")

    st.write(
        """
        - User-friendly web interface
        - Machine Learning based prediction
        - Approval confidence percentage
        - Dataset analysis
        - Model and technology information
        - Interactive dashboard
        """
    )

    st.subheader("Disclaimer")

    st.warning(
        "This application is developed for educational and demonstration "
        "purposes. It does not replace the official loan evaluation process "
        "of banks or financial institutions."
    )
    st.divider()

st.markdown(
    """
    <div style="text-align: center; color: gray;">
        <p><b>AI-Based Loan Approval Prediction System</b></p>
        <p>Developed using Python, Machine Learning and Streamlit</p>
        <p>© 2026 | B.Tech AI&DS Major Project</p>
    </div>
    """,
    unsafe_allow_html=True
)