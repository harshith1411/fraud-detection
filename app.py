import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
import shap
from streamlit_shap import st_shap

# Title of the App
st.title("Fraud Detection System with Explanations V2")

# Step 1: Upload CSV File
uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load the CSV file into a DataFrame
        data = pd.read_csv(uploaded_file, parse_dates=['transmission_date_time'], dayfirst=True)
        st.write("### Uploaded Data Preview")
        st.dataframe(data.head())

        # Step 2: Preprocessing and Feature Engineering
        required_columns = [
            'user_id', 'mcc', 'amount', 'transmission_date_time',
            'txn_currency_code', 'channel', 'pan_entry_mode'
        ]

        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        # Fill missing values and preprocess MCC codes
        data['amount'] = data['amount'].fillna(0).abs()  # Ensure positive amounts
        data['mcc'] = data['mcc'].astype(str).str.zfill(4)  # Ensure MCC codes are 4 digits

        # Feature Engineering (include all relevant columns)
        data['hour'] = data['transmission_date_time'].dt.hour
        data['day_of_week'] = data['transmission_date_time'].dt.dayofweek

        features = pd.get_dummies(
            data[['amount', 'mcc', 'txn_currency_code', 'channel', 'pan_entry_mode', 'hour', 'day_of_week']],
            columns=['mcc', 'txn_currency_code', 'channel', 'pan_entry_mode']
        )

        # Step 3: Train Isolation Forest Model
        model = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0
        )
        model.fit(features)

        # Step 4: Detect Anomalies
        data['anomaly_score'] = model.decision_function(features)
        data['is_anomaly'] = model.predict(features)
        anomalies = data[data['is_anomaly'] == -1]

        if not anomalies.empty:
            # Step 5: Generate Explanations with SHAP and Reasons

            # Create user profiles for reasoning
            user_profiles = data.groupby('user_id').agg(
                avg_amount=('amount', 'mean'),
                common_mcc=('mcc', lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
                common_currency=('txn_currency_code', lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
                common_channel=('channel', lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
                common_pan_entry_mode=('pan_entry_mode', lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
                common_hour=('hour', lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
            )

            # Generate reasons for flagged anomalies
            reasons = []
            for idx, row in anomalies.iterrows():
                profile = user_profiles.loc[row['user_id']]
                reasons.append(
                    f"Amount ₹{row['amount']:.2f} (vs usual ₹{profile['avg_amount']:.2f}) | "
                    f"MCC {row['mcc']} (common: {profile['common_mcc']}) | "
                    f"Currency {row['txn_currency_code']} (common: {profile['common_currency']}) | "
                    f"Channel {row['channel']} (common: {profile['common_channel']}) | "
                    f"Entry Mode {row['pan_entry_mode']} (common: {profile['common_pan_entry_mode']}) | "
                    f"Hour {row['hour']} (common: {profile['common_hour']})"
                )

            anomalies.insert(0, 'reason', reasons)

            st.write("## 🚨 Detected Anomalies")
            st.dataframe(anomalies[['reason', 'amount', 'mcc', 'channel', 'pan_entry_mode']])

            # SHAP Explanations with Index Alignment
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(features)

                # Reset indices to ensure alignment
                features.reset_index(drop=True, inplace=True)
                anomalies.reset_index(drop=True, inplace=True)

                st.write("## 📊 Explanation Dashboard")
                with st.expander("Transaction Breakdown", expanded=True):
                    for i in range(len(anomalies)):
                        pos_idx = i  # Positional index after reset

                        st.markdown(f"### Anomaly {i+1}")
                        try:
                            st_shap(shap.force_plot(
                                explainer.expected_value[0],
                                shap_values[0][pos_idx],
                                features.iloc[pos_idx],
                                feature_names=features.columns.tolist(),
                                matplotlib=True
                            ))
                        except Exception as e:
                            st.warning(f"Visualization failed for anomaly {i+1}. Showing contributions:")
                            contribs = pd.Series(
                                np.abs(shap_values[0][pos_idx]),
                                index=features.columns
                            ).nlargest(5)
                            st.bar_chart(contribs)

                        st.markdown("---")

                anomalies.to_csv("flagged_transactions_with_reasons_v2.csv", index=False)
                st.success("Results saved to flagged_transactions_with_reasons.csv")

            except Exception as e:
                st.error(f"SHAP explanation failed: {str(e)}")

        else:
            st.success("✅ No anomalies found!")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    st.info("📁 Please upload a CSV file to begin analysis.")
