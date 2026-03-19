import streamlit as st
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsureIQ · Claim Predictor",
    page_icon="🛡️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #0a0e1a;
    --card:    #111827;
    --field:   #161d2e;
    --border:  #1e2d45;
    --accent:  #3b82f6;
    --accent2: #06b6d4;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --radius:  12px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 800px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    text-align: center;
}
.hero::before {
    content: "";
    position: absolute; top: -60px; left: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.hero::after {
    content: "";
    position: absolute; bottom: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(6,182,212,0.12) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.35);
    color: #93c5fd;
    font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 0.3rem 0.9rem; border-radius: 999px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem; font-weight: 800; line-height: 1.15;
    margin: 0 0 0.6rem;
    background: linear-gradient(90deg, #e2e8f0 0%, #93c5fd 60%, #67e8f9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: var(--muted); font-size: 0.95rem; margin: 0; }

/* ── Section card ── */
.form-section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.5rem 0.8rem;
    margin-bottom: 0.85rem;
}
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after {
    content: ""; flex: 1; height: 1px; background: var(--border);
}

/* ── Column cleanup — tight equal gap ── */
[data-testid="stHorizontalBlock"] {
    gap: 12px !important;
    align-items: flex-start !important;
}
[data-testid="column"] {
    padding: 0 !important;
    min-width: 0 !important;
}

/* ── Number input ── */
.stNumberInput { margin-bottom: 0.75rem !important; }
.stNumberInput > label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important; font-weight: 500 !important;
    color: #94a3b8 !important; margin-bottom: 0.3rem !important;
}
.stNumberInput > div > div {
    border-radius: var(--radius) !important;
    border: 1.5px solid var(--border) !important;
    background: var(--field) !important;
    overflow: hidden;
}
.stNumberInput > div > div > input {
    background: var(--field) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.5rem 0.75rem !important;
    height: 44px !important;
}
.stNumberInput > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.13) !important;
}

/* ── Selectbox ── */
.stSelectbox { margin-bottom: 0.75rem !important; }
.stSelectbox > label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important; font-weight: 500 !important;
    color: #94a3b8 !important; margin-bottom: 0.3rem !important;
}
.stSelectbox > div > div {
    background: var(--field) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    min-height: 44px !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.13) !important;
}
[data-baseweb="select"] [role="listbox"] {
    background: #1a2235 !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-baseweb="select"] [role="option"] {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}
[data-baseweb="select"] [role="option"]:hover {
    background: rgba(59,130,246,0.12) !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    color: white; border: none;
    border-radius: var(--radius);
    padding: 0.85rem 1.5rem;
    font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 700; letter-spacing: 0.04em;
    cursor: pointer; transition: opacity 0.2s, transform 0.15s;
    box-shadow: 0 4px 24px rgba(59,130,246,0.28);
    margin-top: 0.4rem;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* ── Result cards ── */
.result-card {
    border-radius: 14px; padding: 1.4rem 1.5rem;
    margin-top: 1.2rem;
    display: flex; align-items: center; gap: 1.1rem;
    animation: slideUp 0.35s ease;
}
.result-card.success {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(6,182,212,0.07));
    border: 1px solid rgba(16,185,129,0.3);
}
.result-card.danger {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(245,158,11,0.06));
    border: 1px solid rgba(239,68,68,0.3);
}
.result-card.age-warn {
    background: linear-gradient(135deg, rgba(245,158,11,0.13), rgba(251,191,36,0.06));
    border: 1px solid rgba(245,158,11,0.35);
}
.result-icon { font-size: 2.1rem; flex-shrink: 0; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem; font-weight: 700;
    margin: 0 0 0.22rem; color: var(--text);
}
.result-sub { font-size: 0.82rem; color: var(--muted); margin: 0; line-height: 1.5; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0 0.8rem; }

@keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("carinsuranceclaim.joblib")

newmodel = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🛡️ AI-Powered Underwriting</div>
    <h1>Insurance Claim<br>Predictor</h1>
    <p>Enter customer details to instantly assess claim likelihood</p>
</div>
""", unsafe_allow_html=True)

# ── Section 1 · Personal Info ─────────────────────────────────────────────────
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown('<div class="section-label">👤 Personal Information</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
with c2:
    gender = st.selectbox("Gender", ["Male", "Female"])
gender_male = 1 if gender == "Male" else 0
st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2 · Policy Details ───────────────────────────────────────────────
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📋 Policy Details</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    policy_type = st.selectbox("Policy Type", ["Gold", "Premium", "Other"])
with c4:
    vehicle_type = st.selectbox("Vehicle Type", ["SUV", "Sedan", "Truck"])
policy_type_gold    = 1 if policy_type == "Gold"    else 0
policy_type_premium = 1 if policy_type == "Premium" else 0
vehicle_type_suv   = 1 if vehicle_type == "SUV"   else 0
vehicle_type_sedan = 1 if vehicle_type == "Sedan" else 0
vehicle_type_truck = 1 if vehicle_type == "Truck" else 0

c5, c6 = st.columns(2)
with c5:
    annual_premium = st.number_input("Annual Premium (₹)", min_value=0.0, value=50000.0, step=1000.0)
with c6:
    claim_amount = st.number_input("Claim Amount (₹)", min_value=0.0, value=100000.0, step=5000.0)
st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3 · Incident Details ─────────────────────────────────────────────
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown('<div class="section-label">🚗 Incident Details</div>', unsafe_allow_html=True)
c7, c8 = st.columns(2)
with c7:
    accident_type = st.selectbox("Accident Type", ["Minor", "Total Loss"])
with c8:
    incident_location = st.selectbox("Incident Location", ["Urban", "Rural"])
accident_type_minor      = 1 if accident_type == "Minor"      else 0
accident_type_total_loss = 1 if accident_type == "Total Loss" else 0
incident_location_urban  = 1 if incident_location == "Urban"  else 0

c9, c10 = st.columns(2)
with c9:
    police_report = st.selectbox("Police Report Filed?", ["Yes", "No"])
with c10:
    witness_present = st.selectbox("Witness Present?", ["Yes", "No"])
police_report_yes   = 1 if police_report   == "Yes" else 0
witness_present_yes = 1 if witness_present == "Yes" else 0
st.markdown('</div>', unsafe_allow_html=True)

# ── Section 4 · Claim History ────────────────────────────────────────────────
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📊 Claim History</div>', unsafe_allow_html=True)
c11, c12 = st.columns(2)
with c11:
    past_claims = st.number_input("Past Claims", min_value=0, value=0)
with c12:
    days_to_claim = st.number_input("Days to Claim", min_value=0, value=10)
st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

if st.button("⚡  Run Prediction"):
    if age < 18:
        st.markdown("""
        <div class="result-card age-warn">
            <div class="result-icon">⚠️</div>
            <div>
                <p class="result-title">Ineligible — Age Restriction</p>
                <p class="result-sub">Customer is under 18 and cannot file an insurance claim.</p>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        input_data = np.array([[
            age, annual_premium, claim_amount, past_claims, days_to_claim,
            gender_male, policy_type_gold, policy_type_premium,
            vehicle_type_suv, vehicle_type_sedan, vehicle_type_truck,
            accident_type_minor, accident_type_total_loss,
            police_report_yes, witness_present_yes, incident_location_urban
        ]], dtype=float)

        prediction = newmodel.predict(input_data)

        if prediction[0] == 1:
            st.markdown("""
            <div class="result-card success">
                <div class="result-icon">✅</div>
                <div>
                    <p class="result-title">Claim Likely Approved</p>
                    <p class="result-sub">Based on the provided details, this customer is predicted to file an insurance claim.</p>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card danger">
                <div class="result-icon">❌</div>
                <div>
                    <p class="result-title">Claim Unlikely</p>
                    <p class="result-sub">Based on the provided details, this customer is predicted NOT to file an insurance claim.</p>
                </div>
            </div>""", unsafe_allow_html=True)
    
    