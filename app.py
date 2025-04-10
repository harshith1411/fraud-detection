import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
import shap
from streamlit_shap import st_shap

# Title of the App
st.title("Advanced Fraud Detection System with Explanations")

# Step 1: Upload CSV File
uploaded_file = st.file_uploader("Upload transaction CSV file", type=["csv"])

if uploaded_file:
    try:
        # Define a custom date parser to avoid inconsistent parsing
        def custom_date_parser(date_str):
            return pd.to_datetime(date_str, format='%Y-%m-%dT%H:%M:%S', errors='coerce')

        # Load the CSV file into a DataFrame with consistent date parsing
        data = pd.read_csv(
            uploaded_file,
            parse_dates=['transmission_date_time'],
            date_parser=custom_date_parser
        )
        st.write("### Uploaded Data Preview")
        st.dataframe(data.head())

        # Validate required columns
        required_columns = [
            'user_id', 'mcc', 'amount', 'transmission_date_time',
            'txn_currency_code', 'channel', 'pan_entry_mode',
            'a_id', 'm_id', 'happay_fee_amount'
        ]
        
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        # Feature Engineering
        data['amount'] = data['amount'].fillna(0).abs()  # Ensure positive amounts
        data['happay_fee_amount'] = data['happay_fee_amount'].fillna(0).abs()
        data['mcc'] = data['mcc'].astype(str).str.zfill(4)
        data['a_id'] = data['a_id'].astype(str)
        data['m_id'] = data['m_id'].astype(str)
        data['pan_entry_mode'] = data['pan_entry_mode'].astype(str)

        # Add temporal features
        data['hour'] = data['transmission_date_time'].dt.hour
        data['day_of_week'] = data['transmission_date_time'].dt.dayofweek

        # Create feature matrix
        features = pd.get_dummies(
            data[['amount', 'happay_fee_amount', 'mcc', 'txn_currency_code',
                  'channel', 'pan_entry_mode', 'a_id', 'm_id', 'hour', 'day_of_week']],
            columns=['mcc', 'txn_currency_code', 'channel', 'pan_entry_mode', 'a_id', 'm_id']
        )

        # Train Isolation Forest model
        model = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=150,
            max_samples=0.8
        )
        model.fit(features)

        # Detect anomalies
        data['anomaly_score'] = model.decision_function(features)
        data['is_anomaly'] = np.where(data['anomaly_score'] < np.percentile(data['anomaly_score'], 1), -1, 1)
        anomalies = data[data['is_anomaly'] == -1]

        if not anomalies.empty:
            # Generate explanations with SHAP and reasons
            user_profiles = data.groupby('user_id').agg({
                'amount': ['mean'],
                'happay_fee_amount': ['mean'],
                'mcc': lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
                'txn_currency_code': lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
                'channel': lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
                'pan_entry_mode': lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
                'hour': lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
            }).droplevel(1, axis=1)

            reasons = []
            for idx, row in anomalies.iterrows():
                profile = user_profiles.loc[row['user_id']]
                reasons.append(
                    f"Amount ₹{row['amount']:.2f} (avg: ₹{profile['amount']:.2f}) | "
                    f"Fee ₹{row['happay_fee_amount']:.2f} | "
                    f"MCC {row['mcc']} | "
                    f"Currency {row['txn_currency_code']} | "
                    f"Channel {row['channel']} | "
                    f"Entry Mode {row['pan_entry_mode']} | "
                    f"Hour {row['hour']} (common: {profile['hour']})"
                )

            anomalies.insert(0, 'reason', reasons)

            st.write("## 🚨 Detected Anomalies")
            st.dataframe(anomalies[['reason', 'amount', 'happay_fee_amount', 'mcc',
                                     'txn_currency_code', 'channel', 'pan_entry_mode']])

            # Add download button before Explanation Dashboard section
            csv_data = anomalies.to_csv(index=False).encode('utf-8')
            # st.write("### Download Anomaly Report")
            st.download_button(
                label="Download Report",
                data=csv_data,
                file_name="flagged_transactions_with_reasons.csv",
                mime="text/csv"
            )

            # SHAP Explanations with Index Alignment
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(features)

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

            except Exception as e:
                st.error(f"SHAP explanation failed: {str(e)}")

        else:
            st.success("✅ No anomalies found!")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    st.info("📁 Please upload a CSV file to begin analysis.")
