import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="ChurnNova", page_icon="🚀", layout="wide")

# -------------------------------------------------
# GLOBAL STYLING
# -------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#e0f2fe,#f8fafc);
}

.logo{
font-size:50px;
font-weight:900;
text-align:center;
background: linear-gradient(90deg,#2563eb,#9333ea,#ec4899);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
letter-spacing:2px;
}

.tagline{
text-align:center;
font-size:16px;
color:#374151;
margin-bottom:25px;
}

.card{
background:white;
padding:20px;
border-radius:14px;
box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}

.result-good{
background:linear-gradient(135deg,#10b981,#059669);
padding:30px;
border-radius:14px;
color:white;
font-size:22px;
text-align:center;
font-weight:600;
}

.result-bad{
background:linear-gradient(135deg,#ef4444,#b91c1c);
padding:30px;
border-radius:14px;
color:white;
font-size:22px;
text-align:center;
font-weight:600;
}

.stButton>button{
background:linear-gradient(135deg,#2563eb,#1d4ed8);
color:white;
font-weight:600;
border-radius:10px;
padding:10px 24px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("models/churn_model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    feature_columns = pickle.load(open("models/feature_columns.pkl", "rb"))
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown('<div class="logo">ChurnNova</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">AI Customer Churn Intelligence & Retention Platform</div>', unsafe_allow_html=True)

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
"🔮 Predict Customer",
"📂 Bulk Prediction",
"📊 Churn Analytics",
"🤖 Model Intelligence"
])

# =================================================
# TAB 1 — SINGLE CUSTOMER PREDICTION
# =================================================
with tab1:

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen",["Yes","No"])
        Partner = st.selectbox("Married", ["Yes","No"])
       
    with col2:
        tenure = st.slider("Tenure (months)",0,72,12)
        MonthlyCharges = st.slider("Monthly Charges",18,120,70)
        TotalCharges = st.number_input("Total Charges",0.0,10000.0,1000.0)

    with col3:
        Contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])

        # New HR related features suggested
        pregnant = st.selectbox("Pregnant (Only applicable for Female)", ["No","Yes"])
        maternity_leave = "No"

        if gender == "Female" and pregnant == "Yes":
            maternity_leave = "Yes"


        Dependents = st.selectbox("Dependents", ["Yes","No"])

    if st.button("Predict Churn Risk",use_container_width=True):

        input_df = pd.DataFrame([{
            "gender":gender,
            "SeniorCitizen":SeniorCitizen,
            "Partner":Partner,
            "Dependents":Dependents,
            "tenure":tenure,
            "MonthlyCharges":MonthlyCharges,
            "TotalCharges":TotalCharges,
            "Contract":Contract,

            # Added professional HR features
            "Pregnant":pregnant,
            "MaternityLeave":maternity_leave
        }])

        encoded = pd.get_dummies(input_df)
        encoded = encoded.reindex(columns=feature_columns, fill_value=0)

        scaled = scaler.transform(encoded)

        prob = model.predict_proba(scaled)[0][1]

        # Business rule adjustment (HR logic)
        # Pregnant employees on maternity leave are less likely to churn
        if gender == "Female" and pregnant == "Yes":
            prob = prob * 0.75
        pred = model.predict(scaled)[0]

        colA, colB = st.columns([1,2])

        with colA:
            if pred == 1:
                st.markdown(f'<div class="result-bad">⚠ High Churn Risk<br>{prob*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-good">✅ Likely to Stay<br>{(1-prob)*100:.1f}%</div>', unsafe_allow_html=True)

        with colB:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                number={'suffix':'%'},
                title={'text':'Churn Probability'},
                gauge={'axis':{'range':[0,100]}}
            ))
            st.plotly_chart(fig,use_container_width=True)

# =================================================
# TAB 2 — BULK CSV PREDICTION
# =================================================
with tab2:

    st.subheader("Upload CSV to Predict Multiple Customers")

    file = st.file_uploader("Upload customer dataset",type=["csv"])

    if file:

        df = pd.read_csv(file)
        st.write("Dataset Preview",df.head())

        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        scaled = scaler.transform(df_encoded)

        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:,1]

        df["ChurnPrediction"] = preds
        df["ChurnProbability"] = probs

        # Risk segmentation
        df["RiskLevel"] = pd.cut(
            df["ChurnProbability"],
            bins=[0,0.35,0.65,1],
            labels=["Low Risk","Medium Risk","High Risk"]
        )

        st.write("Prediction Results",df.head())

        churn_rate = preds.mean()*100
        st.metric("Predicted Churn Rate",f"{churn_rate:.2f}%")

        fig = px.histogram(df,x="ChurnProbability",nbins=20,title="Churn Risk Distribution")
        st.plotly_chart(fig,use_container_width=True)

        risk_fig = px.pie(df,names="RiskLevel",title="Customer Risk Segmentation")
        st.plotly_chart(risk_fig,use_container_width=True)

# =================================================
# TAB 3 — CHURN ANALYTICS
# =================================================
with tab3:

    st.subheader("Customer Behavior Analytics")

    demo_data = pd.DataFrame({
        "Contract":["Month-to-month","One year","Two year"],
        "ChurnRate":[42,11,3]
    })

    fig = px.bar(
        demo_data,
        x="Contract",
        y="ChurnRate",
        title="Churn Rate by Contract Type"
    )

    st.plotly_chart(fig,use_container_width=True)

    demo_data2 = pd.DataFrame({
        "InternetService":["DSL","Fiber","No Internet"],
        "ChurnRate":[20,41,7]
    })

    fig2 = px.bar(
        demo_data2,
        x="InternetService",
        y="ChurnRate",
        title="Churn by Internet Service"
    )

    st.plotly_chart(fig2,use_container_width=True)

# =================================================
# TAB 4 — MODEL INTELLIGENCE
# =================================================
with tab4:
    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        "Feature":feature_columns,
        "Importance":model.feature_importances_
    }).sort_values("Importance",ascending=False).head(15)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features Influencing Churn"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("AI Insight")

    top_feature = importance_df.iloc[0]["Feature"]

    st.info(f"Most influential factor affecting churn is **{top_feature}**. Customers with unfavorable values in this feature are more likely to leave.")

    st.subheader("Model Information")

    st.write("Model Type: Random Forest Classifier")
    st.write("Total Features Used:",len(feature_columns))

st.markdown("---")
st.write("By Saptshrungi Kadole • Customer Retention Intelligence System")
