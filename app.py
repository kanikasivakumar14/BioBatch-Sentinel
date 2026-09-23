
# Import Streamlit for the interactive web interface.
import streamlit as st

# Import Pandas for creating model input tables.
import pandas as pd

# Import NumPy for numerical operations.
import numpy as np

# Import Joblib for loading trained machine-learning models.
import joblib


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

# Configure the browser tab and dashboard layout.
st.set_page_config(
    page_title="BioBatch Sentinel",
    page_icon="🧬",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD TRAINED MODELS
# ---------------------------------------------------------

# Load the trained Random Forest fault classifier.
rf_model = joblib.load(
    "biobatch_random_forest.pkl"
)

# Load the trained penicillin concentration regression model.
yield_model = joblib.load(
    "biobatch_yield_model.pkl"
)

# Load the trained Isolation Forest anomaly detector.
isolation_model = joblib.load(
    "biobatch_isolation_forest.pkl"
)


# ---------------------------------------------------------
# DASHBOARD HEADER
# ---------------------------------------------------------

# Display the project title.
st.title(
    "🧬 BioBatch Sentinel"
)

# Display the project subtitle.
st.subheader(
    "Explainable AI-Based Fermentation Batch Risk Assessment"
)

# Explain what the prototype does.
st.write(
    """
    BioBatch Sentinel is a proof-of-concept AI decision-support
    system developed using IndPenSim-derived fermentation data.
    It estimates batch fault risk, predicts penicillin concentration,
    and detects unusual process behaviour from batch-level
    process measurements.
    """
)

# Display an important prototype notice.
st.info(
    "Research/educational prototype — not a validated GMP process-control system."
)


# ---------------------------------------------------------
# USER INPUT SECTION
# ---------------------------------------------------------

st.header(
    "1. Enter Fermentation Batch Measurements"
)

st.write(
    "Enter the four batch-level measurements used by the trained AI models."
)


# Create two dashboard columns.
input_col1, input_col2 = st.columns(2)


# Create PAA input.
with input_col1:

    paa = st.number_input(
        "PAA Offline Measurement",
        min_value=0.0,
        max_value=15000.0,
        value=1200.0,
        step=10.0
    )


# Create NH3 input.
with input_col1:

    nh3 = st.number_input(
        "NH3 Offline Measurement",
        min_value=0.0,
        max_value=7000.0,
        value=1900.0,
        step=10.0
    )


# Create biomass input.
with input_col2:

    biomass = st.number_input(
        "Biomass Offline Measurement",
        min_value=0.0,
        max_value=50.0,
        value=22.0,
        step=0.1
    )


# Create viscosity input.
with input_col2:

    viscosity = st.number_input(
        "Viscosity Offline Measurement",
        min_value=0.0,
        max_value=150.0,
        value=70.0,
        step=0.1
    )


# ---------------------------------------------------------
# CREATE MODEL INPUT
# ---------------------------------------------------------

# Create a dataframe using exactly the same feature names
# and order that were used during model training.
input_data = pd.DataFrame(
    {
        "PAA_Offline": [paa],
        "NH3_Offline": [nh3],
        "Biomass_Offline": [biomass],
        "Viscosity_Offline": [viscosity]
    }
)


# ---------------------------------------------------------
# DISPLAY ENTERED VALUES
# ---------------------------------------------------------

st.subheader(
    "Entered Process Measurements"
)

st.dataframe(
    input_data,
    use_container_width=True
)


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

if st.button(
    "Analyze Batch",
    type="primary"
):

    # -----------------------------------------------------
    # FAULT-RISK PREDICTION
    # -----------------------------------------------------

    # Predict whether the batch resembles normal or fault data.
    batch_prediction = rf_model.predict(
        input_data
    )[0]


    # Calculate the model probability assigned to the fault class.
    failure_probability = rf_model.predict_proba(
        input_data
    )[0][1]


    # Convert probability into percentage.
    failure_risk = failure_probability * 100


    # -----------------------------------------------------
    # RISK CATEGORY
    # -----------------------------------------------------

    # Assign a prototype dashboard risk category.
    if failure_risk < 30:

        risk_category = "LOW"

    elif failure_risk < 60:

        risk_category = "MODERATE"

    elif failure_risk < 80:

        risk_category = "HIGH"

    else:

        risk_category = "CRITICAL"


    # -----------------------------------------------------
    # PENICILLIN CONCENTRATION PREDICTION
    # -----------------------------------------------------

    # Predict penicillin concentration using the regression model.
    predicted_penicillin = yield_model.predict(
        input_data
    )[0]


    # -----------------------------------------------------
    # ANOMALY DETECTION
    # -----------------------------------------------------

    # Detect whether the entered process pattern is unusual.
    anomaly_prediction = isolation_model.predict(
        input_data
    )[0]


    # Convert Isolation Forest output into readable text.
    if anomaly_prediction == -1:

        anomaly_status = "ANOMALY DETECTED"

    else:

        anomaly_status = "NO ANOMALY DETECTED"


    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    st.header(
        "2. BioBatch Sentinel Results"
    )


    # Create three result columns.
    result_col1, result_col2, result_col3 = st.columns(3)


    # Display fault-risk probability.
    with result_col1:

        st.metric(
            "Predicted Failure Risk",
            f"{failure_risk:.1f}%"
        )


    # Display predicted penicillin concentration.
    with result_col2:

        st.metric(
            "Predicted Penicillin Concentration",
            f"{predicted_penicillin:.2f} g/L"
        )


    # Display anomaly result.
    with result_col3:

        st.metric(
            "Anomaly Detection",
            anomaly_status
        )


    # -----------------------------------------------------
    # BATCH CLASSIFICATION
    # -----------------------------------------------------

    st.subheader(
        "Batch Assessment"
    )


    # Display classifier result.
    if batch_prediction == 1:

        st.error(
            "AI Classification: FAULT / AT-RISK BATCH"
        )

    else:

        st.success(
            "AI Classification: NORMAL-LIKE BATCH"
        )


    # -----------------------------------------------------
    # DISPLAY RISK CATEGORY
    # -----------------------------------------------------

    st.write(
        "**Prototype Risk Category:**",
        risk_category
    )


    # Display a progress bar for the estimated fault probability.
    st.progress(
        int(
            min(
                max(failure_risk, 0),
                100
            )
        )
    )


    # -----------------------------------------------------
    # PROCESS-REVIEW MESSAGE
    # -----------------------------------------------------

    st.subheader(
        "Decision-Support Message"
    )


    # Provide a conservative process-review message.
    if (
        batch_prediction == 1
        or anomaly_prediction == -1
    ):

        st.warning(
            """
            This batch shows characteristics that may warrant
            additional process review. Examine the measured
            variables and compare the batch with established
            process specifications and validated controls.
            """
        )

    else:

        st.success(
            """
            The entered measurements resemble the normal
            patterns represented in the training dataset.
            Continue routine process monitoring and established
            quality-control procedures.
            """
        )


    # -----------------------------------------------------
    # SIMPLE MODEL EXPLANATION
    # -----------------------------------------------------

    st.header(
        "3. Model Explanation"
    )


    # Extract Random Forest feature importance values.
    feature_importances = rf_model.feature_importances_


    # Create an importance table.
    importance_df = pd.DataFrame(
        {
            "Process Variable": input_data.columns,
            "Model Importance": feature_importances
        }
    )


    # Sort features from highest to lowest global importance.
    importance_df = importance_df.sort_values(
        "Model Importance",
        ascending=False
    )


    # Display explanation.
    st.write(
        """
        The chart below shows the Random Forest model's
        global feature importance. It indicates which variables
        were generally most useful to the classifier across its
        training data; it is not a batch-specific causal explanation.
        """
    )


   # Create a clean horizontal feature-importance chart.

chart_data = importance_df.set_index(
    "Process Variable"
)[["Model Importance"]]

st.bar_chart(
    chart_data,
    horizontal=True
)


    # Display importance values.
    st.dataframe(
        importance_df,
        use_container_width=True
    )


    # -----------------------------------------------------
    # FINAL DISCLAIMER
    # -----------------------------------------------------

    st.divider()

    st.caption(
        """
        BioBatch Sentinel is an educational proof-of-concept
        developed using IndPenSim-derived batch-level data.
        Risk categories are prototype presentation thresholds
        and are not validated manufacturing specifications.
        Predictions must not replace validated process controls,
        quality systems, or expert review.
        """
    )
