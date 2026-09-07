import streamlit as st
import pickle
import numpy as np
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student GPA Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
STUDENT_IMAGE = BASE_DIR / "Assets" / "student.svg"


# ============================================================
# LOAD MODEL & SCALER
# ============================================================

try:

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

except FileNotFoundError as e:

    st.error(
        f"Required file not found: {e}"
    )

    st.info(
        "Make sure model.pkl and scaler.pkl are in the "
        "same folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error(
        f"Error loading the model: {e}"
    )

    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   GLOBAL
============================================================ */

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.stApp {

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(37, 99, 235, 0.15),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(124, 58, 237, 0.14),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #050a17 0%,
            #081225 50%,
            #0d1530 100%
        );

    color: #f8fafc;
}


/* ============================================================
   MAIN CONTAINER
============================================================ */

.block-container {

    max-width: 1180px;

    padding-top: 35px;
    padding-bottom: 50px;
}


/* ============================================================
   TEXT
============================================================ */

h1 {

    color: #f8fafc !important;

    font-size: 46px !important;

    font-weight: 800 !important;

    letter-spacing: -1.5px !important;

    line-height: 1.1 !important;
}

h2 {

    color: #f1f5f9 !important;

    font-weight: 750 !important;
}

h3 {

    color: #e2e8f0 !important;

    font-weight: 700 !important;
}

p {

    color: #94a3b8;

    line-height: 1.7;
}


/* ============================================================
   HERO CONTAINER
============================================================ */

div[data-testid="stVerticalBlockBorderWrapper"] {

    border-color:
        rgba(148, 163, 184, 0.12) !important;

    background:
        rgba(15, 23, 42, 0.55) !important;

    border-radius: 20px !important;
}


/* ============================================================
   IMAGE
============================================================ */

div[data-testid="stImage"] img {

    border-radius: 22px;

    filter:
        drop-shadow(
            0 20px 40px rgba(37, 99, 235, 0.25)
        );
}


/* ============================================================
   INPUTS
============================================================ */

label {

    color: #cbd5e1 !important;

    font-weight: 600 !important;
}

div[data-baseweb="input"] {

    background-color:
        rgba(15, 23, 42, 0.9) !important;

    border:
        1px solid #263653 !important;

    border-radius: 10px !important;
}

div[data-baseweb="select"] > div {

    background-color:
        rgba(15, 23, 42, 0.9) !important;

    border:
        1px solid #263653 !important;

    border-radius: 10px !important;
}

input {

    color: #f8fafc !important;
}


/* ============================================================
   BUTTON
============================================================ */

.stButton > button {

    width: 100%;

    min-height: 50px;

    border-radius: 12px;

    border: none;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #4f46e5
        );

    color: white;

    font-size: 16px;

    font-weight: 700;

    box-shadow:
        0 10px 30px rgba(37, 99, 235, 0.25);

    transition: all 0.2s ease;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 15px 35px rgba(37, 99, 235, 0.38);
}


/* ============================================================
   METRICS
============================================================ */

div[data-testid="stMetric"] {

    background:
        rgba(15, 23, 42, 0.70);

    border:
        1px solid rgba(148, 163, 184, 0.10);

    border-radius: 16px;

    padding: 15px 18px;

    box-shadow:
        0 10px 30px rgba(0, 0, 0, 0.15);
}

div[data-testid="stMetricLabel"] {

    color: #64748b !important;
}

div[data-testid="stMetricValue"] {

    color: #f8fafc !important;
}


/* ============================================================
   PROGRESS BAR
============================================================ */

div[data-testid="stProgressBar"] {

    margin-top: 10px;
}


/* ============================================================
   DIVIDER
============================================================ */

hr {

    border-color:
        rgba(148, 163, 184, 0.10) !important;
}


/* ============================================================
   CAPTION
============================================================ */

.stCaption {

    color: #64748b !important;
}


/* ============================================================
   ALERTS
============================================================ */

div[data-testid="stAlert"] {

    border-radius: 14px;
}


/* ============================================================
   MOBILE
============================================================ */

@media (max-width: 768px) {

    h1 {
        font-size: 36px !important;
    }

}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================

hero_left, hero_right = st.columns(
    [1.65, 1],
    vertical_alignment="center"
)


with hero_left:

    st.caption(
        "MACHINE LEARNING  •  KNN REGRESSION"
    )

    st.title(
        "Student GPA Predictor"
    )

    st.write(
        "Predict a student's academic performance using "
        "study habits, attendance, academic support, "
        "and extracurricular activities."
    )

    st.write("")


with hero_right:

    if STUDENT_IMAGE.exists():

        st.image(
            str(STUDENT_IMAGE),
            width=280
        )

    else:

        st.warning(
            "student.svg not found in Assets folder."
        )


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.markdown("## Model Overview")

st.caption(
    "Machine learning configuration used by this application."
)

info1, info2, info3, info4 = st.columns(4)


with info1:

    st.metric(
        label="Algorithm",
        value="KNN Regression"
    )


with info2:

    st.metric(
        label="Neighbors",
        value="4"
    )


with info3:

    st.metric(
        label="Features",
        value="8"
    )


with info4:

    st.metric(
        label="Preprocessing",
        value="StandardScaler"
    )


# ============================================================
# STUDENT PROFILE
# ============================================================

st.markdown("## Student Profile")

st.caption(
    "Enter the student's information to generate a GPA prediction."
)


left_col, right_col = st.columns(2)


# ============================================================
# ACADEMIC INFORMATION
# ============================================================

with left_col:

    with st.container(border=True):

        st.subheader(
            "📚 Academic Information"
        )

        study_time = st.number_input(
            "Weekly Study Time (hours)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            help="Average number of hours the student studies per week."
        )

        absences = st.number_input(
            "Number of Absences",
            min_value=0,
            max_value=100,
            value=5,
            step=1,
            help="Total number of school absences."
        )

        grade_class = st.number_input(
            "Grade Class",
            min_value=0.0,
            max_value=5.0,
            value=3.0,
            step=1.0,
            help="Grade class value used by the trained model."
        )

        tutoring = st.selectbox(
            "Tutoring",
            options=[0, 1],
            format_func=lambda x:
                "Yes — Receives Tutoring"
                if x == 1
                else "No — No Tutoring"
        )


# ============================================================
# SUPPORT & ACTIVITIES
# ============================================================

with right_col:

    with st.container(border=True):

        st.subheader(
            "🏫 Support & Activities"
        )

        parental_support = st.number_input(
            "Parental Support",
            min_value=0.0,
            max_value=4.0,
            value=2.0,
            step=1.0,
            help="Level of parental support."
        )

        extracurricular = st.selectbox(
            "Extracurricular Activities",
            options=[0, 1],
            format_func=lambda x:
                "Yes — Participates"
                if x == 1
                else "No — Does Not Participate"
        )

        sports = st.selectbox(
            "Sports",
            options=[0, 1],
            format_func=lambda x:
                "Yes — Participates"
                if x == 1
                else "No — Does Not Participate"
        )

        music = st.selectbox(
            "Music",
            options=[0, 1],
            format_func=lambda x:
                "Yes — Participates"
                if x == 1
                else "No — Does Not Participate"
        )


# ============================================================
# BUTTONS
# ============================================================

st.write("")

button_left, button_middle, button_right = st.columns(
    [1, 1.5, 1]
)


with button_middle:

    predict_button = st.button(
        "✨ Predict Student GPA",
        type="primary"
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # INPUT DATA
    # IMPORTANT:
    # Must match the model's training feature order.
    # --------------------------------------------------------

    input_data = np.array([
        [
            study_time,
            absences,
            tutoring,
            parental_support,
            extracurricular,
            sports,
            music,
            grade_class
        ]
    ])


    try:

        # ----------------------------------------------------
        # SCALE INPUT
        # ----------------------------------------------------

        input_scaled = scaler.transform(
            input_data
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(
            input_scaled
        )[0]

        prediction = float(prediction)


        # ----------------------------------------------------
        # PERFORMANCE CLASSIFICATION
        # ----------------------------------------------------

        if prediction >= 3.5:

            level = "Excellent Performance"

            description = (
                "The predicted GPA indicates strong academic "
                "performance."
            )

        elif prediction >= 3.0:

            level = "Good Performance"

            description = (
                "The student is predicted to demonstrate "
                "good academic performance."
            )

        elif prediction >= 2.5:

            level = "Average Performance"

            description = (
                "The prediction indicates average academic "
                "performance with room for improvement."
            )

        elif prediction >= 2.0:

            level = "Below Average Performance"

            description = (
                "Additional academic support may help improve "
                "the student's performance."
            )

        else:

            level = "Needs Improvement"

            description = (
                "The prediction suggests that the student "
                "may benefit from additional academic support."
            )


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.markdown(
            "## 🎯 Prediction Result"
        )

        result_col1, result_col2 = st.columns(
            [1.3, 1],
            vertical_alignment="center"
        )


        # ----------------------------------------------------
        # GPA
        # ----------------------------------------------------

        with result_col1:

            with st.container(border=True):

                st.caption(
                    "PREDICTED GPA"
                )

                st.markdown(
                    f"# {prediction:.2f}"
                )

                st.caption(
                    "Predicted academic performance based on "
                    "the provided student profile."
                )

                # GPA visualization
                progress_value = np.clip(
                    prediction / 4.0,
                    0.0,
                    1.0
                )

                st.progress(
                    progress_value
                )

                st.caption(
                    f"{prediction:.2f} / 4.00 GPA"
                )


        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        with result_col2:

            with st.container(border=True):

                st.subheader(
                    "📈 Performance Assessment"
                )

                st.markdown(
                    f"### {level}"
                )

                st.write(
                    description
                )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.markdown(
            "## 📋 Prediction Summary"
        )

        st.caption(
            "Values used by the machine learning model."
        )


        summary1, summary2, summary3, summary4 = st.columns(4)


        with summary1:

            st.metric(
                "Study Time",
                f"{study_time:.1f} hrs"
            )


        with summary2:

            st.metric(
                "Absences",
                str(absences)
            )


        with summary3:

            st.metric(
                "Tutoring",
                "Yes" if tutoring == 1 else "No"
            )


        with summary4:

            st.metric(
                "Parental Support",
                f"{parental_support:.0f}"
            )


        summary5, summary6, summary7, summary8 = st.columns(4)


        with summary5:

            st.metric(
                "Extracurricular",
                "Yes"
                if extracurricular == 1
                else "No"
            )


        with summary6:

            st.metric(
                "Sports",
                "Yes"
                if sports == 1
                else "No"
            )


        with summary7:

            st.metric(
                "Music",
                "Yes"
                if music == 1
                else "No"
            )


        with summary8:

            st.metric(
                "Grade Class",
                f"{grade_class:.0f}"
            )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

        st.info(
            "Please make sure the input feature order and "
            "model/scaler are consistent with your training notebook."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Student GPA Predictor  •  KNN Regression  •  "
    "Python + Scikit-learn + Streamlit"
)