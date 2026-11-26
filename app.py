

import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. LOAD TRAINED MACHINE LEARNING MODEL
# ==========================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    data = joblib.load("student_performance_model.pkl")
    return data["model"], list(data["feature_names"])

try:
    model, feature_names = load_model()
except FileNotFoundError:
    st.error("Model file not found. Keep student_performance_model.pkl in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


# ==========================================
# 2. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #ffffff;
}

h1, h2, h3 {
    color: #ffffff !important;
}

p, label {
    color: #d1d5db;
}

div[data-testid="stMetric"] {
    background: #17243a;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #263b59;
}

div.stButton > button {
    background: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}

div[data-testid="stForm"] {
    border: 1px solid #263b59;
    border-radius: 14px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. HEADER
# ==========================================

st.title("🎓 Student Performance Prediction")

st.markdown("""
Predict student final academic marks using
**Machine Learning and Student Academic Data.**
""")

st.divider()

st.info(
    "Enter the student's information below to estimate their final marks."
)


# ==========================================
# 4. INPUT DATA
# ==========================================

input_data = {feature: 0 for feature in feature_names}


def set_numeric(feature, value):
    if feature in input_data:
        input_data[feature] = value


def set_one_hot(prefix, selected):
    """Encode a categorical selection using the model's feature names."""

    for feature in feature_names:
        if feature.startswith(prefix + "_"):
            input_data[feature] = 0

    selected_feature = f"{prefix}_{selected}"

    if selected_feature in input_data:
        input_data[selected_feature] = 1


def set_yes_no(prefix, answer):
    set_one_hot(prefix, "yes" if answer == "Yes" else "no")


# ==========================================
# 5. STUDENT INFORMATION FORM
# ==========================================

with st.form("student_prediction_form"):

    # --------------------------------------
    # SECTION A: PERSONAL INFORMATION
    # --------------------------------------

    st.subheader("👤 1. Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Student Age",
            min_value=15,
            max_value=22,
            value=17,
            step=1
        )

    with col2:
        school = st.selectbox(
            "School",
            ["GP", "MS"],
            format_func=lambda x: {
                "GP": "Gabriel Pereira",
                "MS": "Mousinho da Silveira"
            }[x]
        )

    with col3:
        sex = st.selectbox(
            "Gender",
            ["F", "M"],
            format_func=lambda x: {
                "F": "Female",
                "M": "Male"
            }[x]
        )

    col1, col2 = st.columns(2)

    with col1:
        address = st.selectbox(
            "Area of Residence",
            ["U", "R"],
            format_func=lambda x: {
                "U": "Urban",
                "R": "Rural"
            }[x]
        )

    with col2:
        famsize = st.selectbox(
            "Family Size",
            ["GT3", "LE3"],
            format_func=lambda x: {
                "GT3": "More than 3 members",
                "LE3": "3 or fewer members"
            }[x]
        )

    # --------------------------------------
    # SECTION B: FAMILY BACKGROUND
    # --------------------------------------

    st.divider()
    st.subheader("👨‍👩‍👦 2. Family Background")

    education_options = {
        0: "No education",
        1: "Primary education",
        2: "5th to 9th grade",
        3: "Secondary education",
        4: "Higher education"
    }

    col1, col2 = st.columns(2)

    with col1:
        medu = st.selectbox(
            "Mother's Education Level",
            list(education_options.keys()),
            format_func=lambda x: education_options[x]
        )

        mjob = st.selectbox(
            "Mother's Occupation",
            ["at_home", "health", "other", "services", "teacher"],
            format_func=lambda x: x.replace("_", " ").title()
        )

    with col2:
        fedu = st.selectbox(
            "Father's Education Level",
            list(education_options.keys()),
            format_func=lambda x: education_options[x]
        )

        fjob = st.selectbox(
            "Father's Occupation",
            ["at_home", "health", "other", "services", "teacher"],
            format_func=lambda x: x.replace("_", " ").title()
        )

    col1, col2 = st.columns(2)

    with col1:
        pstatus = st.selectbox(
            "Parents' Living Status",
            ["T", "A"],
            format_func=lambda x: {
                "T": "Living together",
                "A": "Living apart"
            }[x]
        )

        guardian = st.selectbox(
            "Primary Guardian",
            ["mother", "father", "other"],
            format_func=lambda x: x.title()
        )

    with col2:
        famrel = st.select_slider(
            "Family Relationship Quality",
            options=[1, 2, 3, 4, 5],
            value=4,
            format_func=lambda x: {
                1: "Very Bad",
                2: "Bad",
                3: "Average",
                4: "Good",
                5: "Excellent"
            }[x]
        )

        reason = st.selectbox(
            "Reason for Choosing School",
            ["course", "home", "reputation", "other"],
            format_func=lambda x: x.title()
        )

    # --------------------------------------
    # SECTION C: ACADEMIC PERFORMANCE
    # --------------------------------------

    st.divider()
    st.subheader("📚 3. Academic Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        g1 = st.number_input(
            "First Period Grade (G1)",
            min_value=0,
            max_value=20,
            value=10
        )

    with col2:
        g2 = st.number_input(
            "Second Period Grade (G2)",
            min_value=0,
            max_value=20,
            value=10
        )

    with col3:
        failures = st.selectbox(
            "Past Class Failures",
            [0, 1, 2, 3],
            format_func=lambda x: "3 or more" if x == 3 else str(x)
        )

    col1, col2 = st.columns(2)

    with col1:
        studytime = st.selectbox(
            "Weekly Study Time",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Less than 2 hours",
                2: "2 to 5 hours",
                3: "5 to 10 hours",
                4: "More than 10 hours"
            }[x]
        )

    with col2:
        absences = st.number_input(
            "Number of School Absences",
            min_value=0,
            max_value=100,
            value=0
        )

    # --------------------------------------
    # SECTION D: STUDY & DAILY HABITS
    # --------------------------------------

    st.divider()
    st.subheader("⏰ 4. Study & Daily Habits")

    col1, col2 = st.columns(2)

    with col1:
        traveltime = st.selectbox(
            "Travel Time to School",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Less than 15 minutes",
                2: "15 to 30 minutes",
                3: "30 minutes to 1 hour",
                4: "More than 1 hour"
            }[x]
        )

        freetime = st.select_slider(
            "Free Time After School",
            options=[1, 2, 3, 4, 5],
            value=3
        )

        goout = st.select_slider(
            "Going Out With Friends",
            options=[1, 2, 3, 4, 5],
            value=3
        )

    with col2:
        dalc = st.select_slider(
            "Weekday Alcohol Consumption",
            options=[1, 2, 3, 4, 5],
            value=1
        )

        walc = st.select_slider(
            "Weekend Alcohol Consumption",
            options=[1, 2, 3, 4, 5],
            value=1
        )

        health = st.select_slider(
            "Current Health Status",
            options=[1, 2, 3, 4, 5],
            value=3
        )

    # --------------------------------------
    # SECTION E: SCHOOL & FAMILY SUPPORT
    # --------------------------------------

    st.divider()
    st.subheader("🏫 5. School & Family Support")

    col1, col2 = st.columns(2)

    with col1:
        schoolsup = st.selectbox(
            "Extra Educational Support",
            ["No", "Yes"]
        )

        famsup = st.selectbox(
            "Family Educational Support",
            ["No", "Yes"]
        )

        paid = st.selectbox(
            "Extra Paid Classes",
            ["No", "Yes"]
        )

        activities = st.selectbox(
            "Extracurricular Activities",
            ["No", "Yes"]
        )

    with col2:
        nursery = st.selectbox(
            "Attended Nursery School",
            ["No", "Yes"]
        )

        higher = st.selectbox(
            "Wants Higher Education",
            ["No", "Yes"]
        )

        internet = st.selectbox(
            "Internet Access at Home",
            ["No", "Yes"]
        )

        romantic = st.selectbox(
            "Currently in a Romantic Relationship",
            ["No", "Yes"]
        )

    # --------------------------------------
    # SECTION F: ADDITIONAL INPUTS
    # --------------------------------------

    # Map user-friendly answers to model features.

    # Numeric features
    numeric_values = {
        "age": age,
        "Medu": medu,
        "Fedu": fedu,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": dalc,
        "Walc": walc,
        "health": health,
        "absences": absences,
        "G1": g1,
        "G2": g2
    }

    for feature, value in numeric_values.items():
        set_numeric(feature, value)

    # Categorical features
    set_one_hot("school", school)
    set_one_hot("sex", sex)
    set_one_hot("address", address)
    set_one_hot("famsize", famsize)
    set_one_hot("Pstatus", pstatus)
    set_one_hot("Mjob", mjob)
    set_one_hot("Fjob", fjob)
    set_one_hot("reason", reason)
    set_one_hot("guardian", guardian)

    # Yes / No features
    yes_no_values = {
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic
    }

    for feature, value in yes_no_values.items():
        set_yes_no(feature, value)

    # Additional model features not covered above
    known_prefixes = {
        "school", "sex", "address", "famsize", "Pstatus",
        "Mjob", "Fjob", "reason", "guardian",
        "schoolsup", "famsup", "paid", "activities",
        "nursery", "higher", "internet", "romantic"
    }

    known_numeric = set(numeric_values.keys())

    additional_features = [
        f for f in feature_names
        if f not in known_numeric
        and not any(f.startswith(p + "_") for p in known_prefixes)
    ]

    if additional_features:
        st.divider()
        with st.expander("Additional Model Inputs"):
            st.caption(
                "These fields are included because they are required "
                "by your trained model."
            )
            for feature in additional_features:
                input_data[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=0.0,
                    key="extra_" + feature
                )

    # --------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------

    st.divider()

    submitted = st.form_submit_button(
        "🚀 Predict Student Performance"
    )


# ==========================================
# 6. PREDICTION RESULT
# ==========================================

if submitted:

    try:
        input_df = pd.DataFrame(
            [input_data],
            columns=feature_names
        )

        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Final Marks",
                f"{float(prediction):.2f} / 20"
            )

        with col2:
            st.metric(
                "Student's Previous Grade (G2)",
                f"{g2} / 20"
            )

        st.success("Student performance prediction completed!")

        st.caption(
            "This is a machine-learning estimate, not a guaranteed final grade."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ==========================================
# 7. FOOTER
# ==========================================

st.divider()

st.markdown("""
<div style="text-align:center; color:#94a3b8;">
    🎓 Student Performance Prediction System<br>
    Developed using Python, Streamlit and Machine Learning
</div>
""", unsafe_allow_html=True)
