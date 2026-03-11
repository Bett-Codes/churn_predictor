import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay, roc_curve)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Clean & Professional
# ─────────────────────────────────────────────
st.markdown("""
<style>
    body { font-family: 'Segoe UI', sans-serif; }
    .main { background-color: #f8f9fb; }
    .block-container { padding: 2rem 3rem; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { color: #6b7280; margin: 0; font-size: 0.9rem; }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
        border-left: 4px solid #3b82f6;
        padding-left: 0.6rem;
    }
    .risk-high   { color: #dc2626; font-weight: 700; font-size: 1.4rem; }
    .risk-medium { color: #f59e0b; font-weight: 700; font-size: 1.4rem; }
    .risk-low    { color: #16a34a; font-weight: 700; font-size: 1.4rem; }

    div[data-testid="stSidebar"] { background-color: #1e293b; }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & MODEL PIPELINE (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on Telco dataset...")
def load_and_train():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Clean
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df.drop(columns=['customerID'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    df_model = df.copy()

    # Feature engineering
    df_model['AvgMonthlySpend'] = df_model['TotalCharges'] / (df_model['tenure'] + 1)
    df_model['TenureBucket'] = pd.cut(
        df_model['tenure'], bins=[0, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4+yr']
    )

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'TenureBucket']
    le = LabelEncoder()
    for col in binary_cols:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']
    df_model = pd.get_dummies(df_model, columns=multi_cols, drop_first=True)

    X = df_model.drop(columns=['Churn'])
    y = df_model['Churn']

    scaler = StandardScaler()
    num_feats = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend']
    X[num_feats] = scaler.fit_transform(X[num_feats])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
        'XGBoost':             XGBClassifier(n_estimators=200, use_label_encoder=False,
                                              eval_metric='logloss', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc     = roc_auc_score(y_test, y_proba)
        results[name] = {'model': model, 'y_pred': y_pred, 'y_proba': y_proba, 'auc': auc}

    best_name = max(results, key=lambda k: results[k]['auc'])

    return df, df_model, X, y, X_train_res, y_train_res, X_test, y_test, \
           scaler, num_feats, results, best_name


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 📉 Customer Churn Predictor")
st.markdown("An end-to-end ML app to predict telecom customer churn with explainability.")
st.divider()

# ─────────────────────────────────────────────
# LOAD DATA & TRAIN
# ─────────────────────────────────────────────
if not os.path.exists('WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    st.error("⚠️ Dataset not found! Place **WA_Fn-UseC_-Telco-Customer-Churn.csv** in the same folder as app.py")
    st.stop()

(df, df_model, X, y, X_train_res, y_train_res, X_test, y_test,
 scaler, num_feats, results, best_name) = load_and_train()

best_model  = results[best_name]['model']
best_proba  = results[best_name]['y_proba']
best_pred   = results[best_name]['y_pred']


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict Churn", "📊 Model Performance", "🔍 SHAP Explainability"])


# ════════════════════════════════════════════
# TAB 1 — PREDICT CHURN
# ════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Customer Details")
    st.markdown("Fill in the form below to get an instant churn prediction.")
    st.write("")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<p class="section-title">Demographics</p>', unsafe_allow_html=True)
            gender         = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner        = st.selectbox("Has Partner", ["Yes", "No"])
            dependents     = st.selectbox("Has Dependents", ["Yes", "No"])

        with col2:
            st.markdown('<p class="section-title">Services</p>', unsafe_allow_html=True)
            phone_service    = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines   = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security  = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup    = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            tech_support     = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        with col3:
            st.markdown('<p class="section-title">Billing & Contract</p>', unsafe_allow_html=True)
            contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method  = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
            tenure          = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total_charges   = st.number_input("Total Charges ($)", min_value=0.0,
                                               value=float(monthly_charges * tenure + 1))

        submitted = st.form_submit_button("🔮 Predict Churn", width="stretch")

    # ── Prediction Logic ──
    if submitted:
        # Build a single-row dataframe matching training columns
        input_dict = {
            'gender': 1 if gender == 'Male' else 0,
            'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
            'Partner': 1 if partner == 'Yes' else 0,
            'Dependents': 1 if dependents == 'Yes' else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone_service == 'Yes' else 0,
            'PaperlessBilling': 1 if paperless == 'Yes' else 0,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'AvgMonthlySpend': total_charges / (tenure + 1),
            'TenureBucket': pd.cut([tenure], bins=[0, 12, 24, 48, 72],
                                    labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])[0]
        }

        # One-hot encode using training columns
        sample = pd.DataFrame([input_dict])
        le = LabelEncoder()
        sample['TenureBucket'] = le.fit_transform(sample['TenureBucket'].astype(str))

        # Add all one-hot columns present in X, default 0
        for col in X.columns:
            if col not in sample.columns:
                sample[col] = 0

        # Set one-hot flags
        ml_map = {
            f'MultipleLines_{multiple_lines}':    1,
            f'InternetService_{internet_service}': 1,
            f'OnlineSecurity_{online_security}':  1,
            f'OnlineBackup_{online_backup}':      1,
            f'TechSupport_{tech_support}':        1,
            f'Contract_{contract}':               1,
            f'PaymentMethod_{payment_method}':    1,
        }
        for k, v in ml_map.items():
            if k in sample.columns:
                sample[k] = v

        sample = sample[X.columns]
        sample[num_feats] = scaler.transform(sample[num_feats])

        prob = best_model.predict_proba(sample)[0][1]

        # ── Gauge Chart ──
        st.write("")
        st.markdown("### 🎯 Prediction Result")
        r1, r2, r3 = st.columns([1, 2, 1])

        with r2:
            if prob >= 0.7:
                risk_label = "HIGH RISK"
                risk_color = "#dc2626"
                needle_color = "#dc2626"
            elif prob >= 0.4:
                risk_label = "MEDIUM RISK"
                risk_color = "#f59e0b"
                needle_color = "#f59e0b"
            else:
                risk_label = "LOW RISK"
                risk_color = "#16a34a"
                needle_color = "#16a34a"

            # Draw gauge
            fig, ax = plt.subplots(figsize=(6, 3.5), subplot_kw={'aspect': 'equal'})
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-0.2, 1.3)
            ax.axis('off')

            # Background arc segments
            from matplotlib.patches import Wedge
            colors_gauge = ['#16a34a', '#84cc16', '#f59e0b', '#ef4444', '#dc2626']
            for i, c in enumerate(colors_gauge):
                theta1 = 180 - i * 36
                theta2 = 180 - (i + 1) * 36
                wedge = Wedge((0, 0), 1.0, theta2, theta1, width=0.35, color=c, alpha=0.85)
                ax.add_patch(wedge)

            # Needle
            angle_deg = 180 - prob * 180
            angle_rad = np.radians(angle_deg)
            ax.annotate('', xy=(0.7 * np.cos(angle_rad), 0.7 * np.sin(angle_rad)),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=needle_color,
                                        lw=3, mutation_scale=20))
            ax.add_patch(plt.Circle((0, 0), 0.07, color='#1f2937', zorder=5))

            ax.text(0, -0.15, f'{prob:.1%}', ha='center', va='center',
                    fontsize=22, fontweight='bold', color=needle_color)
            ax.text(0, 0.55, 'Churn Probability', ha='center', fontsize=10, color='#6b7280')
            ax.text(-1.1, -0.1, 'Low', fontsize=9, color='#16a34a', fontweight='bold')
            ax.text(0.85, -0.1, 'High', fontsize=9, color='#dc2626', fontweight='bold')

            st.pyplot(fig, width="stretch")
            plt.close()

            if prob >= 0.7:
                st.error(f"⚠️ {risk_label} — This customer is likely to churn. Consider a retention offer.")
            elif prob >= 0.4:
                st.warning(f"🟡 {risk_label} — Monitor this customer closely.")
            else:
                st.success(f"✅ {risk_label} — This customer is unlikely to churn.")

        # ── SHAP Local Explanation ──
        st.write("")
        st.markdown("### 🔍 Why this prediction?")
        explainer   = shap.TreeExplainer(best_model)
        shap_vals   = explainer.shap_values(sample)
        sv_sample   = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

        shap_df = pd.DataFrame({
            'Feature': X.columns,
            'SHAP Value': sv_sample
        }).reindex(pd.Series(sv_sample).abs().sort_values(ascending=False).index)
        shap_df = shap_df.head(10).sort_values('SHAP Value')

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors_bar = ['#dc2626' if v > 0 else '#16a34a' for v in shap_df['SHAP Value']]
        ax2.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors_bar, edgecolor='white')
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.set_title('Top 10 Factors Driving This Prediction', fontsize=12, fontweight='bold')
        ax2.set_xlabel('SHAP Value (impact on churn probability)')
        red_patch   = mpatches.Patch(color='#dc2626', label='Increases churn risk')
        green_patch = mpatches.Patch(color='#16a34a', label='Decreases churn risk')
        ax2.legend(handles=[red_patch, green_patch], loc='lower right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2, width="stretch")
        plt.close()


# ════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Performance Summary")
    st.write("")

    # ── AUC Score Cards ──
    cols = st.columns(4)
    colors_card = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            badge = "🏆 " if name == best_name else ""
            st.markdown(f"""
            <div class="metric-card">
                <p>{badge}{name}</p>
                <h2 style="color:{colors_card[i]}">{res['auc']:.4f}</h2>
                <p>ROC-AUC Score</p>
            </div>""", unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)

    # ── ROC Curves ──
    with c1:
        st.markdown('<p class="section-title">ROC Curves</p>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        line_colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
        for (name, res), lc in zip(results.items(), line_colors):
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
            lw = 2.5 if name == best_name else 1.5
            ls = '-' if name == best_name else '--'
            ax3.plot(fpr, tpr, label=f"{name} ({res['auc']:.3f})",
                     color=lc, lw=lw, linestyle=ls)
        ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curves — All Models', fontweight='bold')
        ax3.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig3, width="stretch")
        plt.close()

    # ── Confusion Matrix ──
    with c2:
        st.markdown('<p class="section-title">Confusion Matrix — Best Model</p>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, best_pred)
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
        disp.plot(ax=ax4, cmap='Blues', colorbar=False)
        ax4.set_title(f'{best_name}', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig4, width="stretch")
        plt.close()

    # ── Classification Report ──
    st.write("")
    st.markdown('<p class="section-title">Classification Report — Best Model</p>', unsafe_allow_html=True)
    report = classification_report(y_test, best_pred,
                                    target_names=['No Churn', 'Churn'],
                                    output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']),
                 width="stretch")


# ════════════════════════════════════════════
# TAB 3 — SHAP EXPLAINABILITY
# ════════════════════════════════════════════
with tab3:
    st.markdown("### 🔍 SHAP Global Feature Importance")
    st.write("Understanding **why** the model makes predictions — not just what it predicts.")
    st.write("")

    with st.spinner("Computing SHAP values..."):
        explainer_global = shap.TreeExplainer(best_model)
        # Use a sample of 300 rows for speed
        X_sample = X_test.sample(300, random_state=42)
        shap_vals_global = explainer_global.shap_values(X_sample)
        sv_global = shap_vals_global[1] if isinstance(shap_vals_global, list) else shap_vals_global

    g1, g2 = st.columns(2)

    with g1:
        st.markdown('<p class="section-title">Beeswarm Plot</p>', unsafe_allow_html=True)
        plt.figure(figsize=(7, 6))
        shap.summary_plot(sv_global, X_sample, plot_type='dot',
                           show=False, max_display=12)
        plt.title('Feature Impact on Churn Prediction', fontweight='bold', fontsize=11)
        plt.tight_layout()
        st.pyplot(plt.gcf(), width="stretch")
        plt.close()

    with g2:
        st.markdown('<p class="section-title">Mean |SHAP| Feature Importance</p>', unsafe_allow_html=True)
        mean_shap = np.abs(sv_global).mean(axis=0)
        shap_importance = pd.DataFrame({
            'Feature': X_sample.columns,
            'Importance': mean_shap
        }).sort_values('Importance', ascending=True).tail(12)

        fig6, ax6 = plt.subplots(figsize=(7, 6))
        bars = ax6.barh(shap_importance['Feature'], shap_importance['Importance'],
                        color='#3b82f6', edgecolor='white', alpha=0.85)
        ax6.set_title('Top 12 Features by Mean |SHAP|', fontweight='bold', fontsize=11)
        ax6.set_xlabel('Mean |SHAP Value|')
        for bar in bars:
            ax6.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{bar.get_width():.3f}', va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig6, width="stretch")
        plt.close()

    # ── Key Insights ──
    st.write("")
    st.markdown('<p class="section-title">💡 Key Insights</p>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        st.info("**Tenure** is the strongest predictor — newer customers churn far more.")
    with i2:
        st.warning("**Month-to-month contracts** significantly increase churn probability.")
    with i3:
        st.error("**High monthly charges** without added-value services drive customers away.")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.85rem;'>"
    "Built with Python · Scikit-learn · XGBoost · SHAP · Streamlit &nbsp;|&nbsp; Customer Churn Portfolio Project"
    "</p>", unsafe_allow_html=True
)
