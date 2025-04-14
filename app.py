import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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
        data['amount'] = data['amount'].fillna(0).abs()
        data['happay_fee_amount'] = data['happay_fee_amount'].fillna(0).abs()
        data['mcc'] = data['mcc'].astype(str).str.zfill(4)
        data['a_id'] = data['a_id'].astype(str)
        data['m_id'] = data['m_id'].astype(str)
        data['pan_entry_mode'] = data['pan_entry_mode'].astype(str)
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
        # Flag the bottom 5% of scores as anomalies
        threshold = np.percentile(data['anomaly_score'], 1)
        data['is_anomaly'] = np.where(data['anomaly_score'] < threshold, -1, 1)
        # Get all anomalies (no filtering)
        anomalies = data[data['is_anomaly'] == -1].sort_values('anomaly_score')

        if not anomalies.empty:
            # Show total count
            st.info(f"Found {len(anomalies)} anomalies.")

            # Generate explanations
            user_profiles = data.groupby('user_id').agg({
                'amount': ['mean'],
                'happay_fee_amount': ['mean'],
                'mcc': lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
                'txn_currency_code': lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
                'channel': lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
                'pan_entry_mode': lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
                'hour': lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
            }).droplevel(1, axis=1)

            # Modified code to show only the most significant reason
            reasons = []
            for idx, row in anomalies.iterrows():
                profile = user_profiles.loc[row['user_id']]
                
                # Calculate relative deviations for numerical columns
                amount_deviation = abs((row['amount'] - profile['amount']) / max(profile['amount'], 1))
                fee_deviation = abs((row['happay_fee_amount'] - profile['happay_fee_amount']) / max(profile['happay_fee_amount'], 1))
                
                # Check for mismatches in categorical columns
                mcc_mismatch = row['mcc'] != profile['mcc']
                currency_mismatch = row['txn_currency_code'] != profile['txn_currency_code']
                channel_mismatch = row['channel'] != profile['channel']
                entry_mode_mismatch = row['pan_entry_mode'] != profile['pan_entry_mode']
                
                # Determine the hour difference (circular)
                hour_diff = min((row['hour'] - profile['hour']) % 24, (profile['hour'] - row['hour']) % 24)
                hour_deviation = hour_diff / 12  # Normalize to [0, 1] range
                
                # Create a dictionary of deviations with descriptive reasons
                deviations = {
                    f"Unusual amount: ₹{row['amount']:.2f} vs avg ₹{profile['amount']:.2f}": amount_deviation,
                    f"Unusual fee: ₹{row['happay_fee_amount']:.2f} vs avg ₹{profile['happay_fee_amount']:.2f}": fee_deviation,
                    f"Unusual merchant category: {row['mcc']}": 1.0 if mcc_mismatch else 0.0,
                    f"Unusual currency: {row['txn_currency_code']}": 1.0 if currency_mismatch else 0.0,
                    f"Unusual channel: {row['channel']}": 1.0 if channel_mismatch else 0.0,
                    f"Unusual entry mode: {row['pan_entry_mode']}": 1.0 if entry_mode_mismatch else 0.0,
                    f"Unusual time: {row['hour']}:00 vs typical {profile['hour']}:00": hour_deviation
                }
                
                # Find the most significant reason
                most_significant_reason = max(deviations, key=deviations.get)
                reasons.append(most_significant_reason)

            # Add the significant reason column to anomalies DataFrame
            anomalies.insert(0, 'reason', reasons)

            # Display results
            st.write("## 🚨 Detected Anomalies")
            st.dataframe(anomalies[['reason', 'amount', 'happay_fee_amount', 'mcc']])

            # Add download button before Transaction Explanations section
            csv_data = anomalies.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Report",
                data=csv_data,
                file_name="fraud_report.csv",
                mime="text/csv"
            )

            # Compute feature importance using custom implementation
            st.write("## 📊 Transaction Explanations")
            with st.expander("Detailed Analysis", expanded=True):
                # Get anomaly indices
                anomaly_indices = anomalies.index.tolist()
                
                # Compute normal profiles
                normal_data = data[data['is_anomaly'] == 1]
                normal_features = features.loc[normal_data.index]
                normal_mean = normal_features.mean()
                normal_std = normal_features.std()
                
                for i, anomaly_idx in enumerate(anomaly_indices, 1):
                    st.markdown(f"### Anomaly {i}")
                    
                    # Get this specific anomaly's features
                    anomaly_features = features.loc[anomaly_idx]
                    
                    # Calculate z-scores to identify deviations
                    z_scores = (anomaly_features - normal_mean) / normal_std.replace(0, 1)
                    
                    # Get top contributing features (highest absolute z-scores)
                    top_features = z_scores.abs().sort_values(ascending=False).head(10)
                    
                    # Display top features and their contribution
                    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
                    colors = ['red' if val > 0 else 'blue' for val in top_features.values]
                    bars = ax.barh(
                        top_features.index,
                        top_features.values,
                        color=colors
                    )
                    
                    # Add value labels
                    for bar in bars:
                        width = bar.get_width()
                        label_x_pos = width if width > 0 else width - 0.5
                        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                                f'{width:.2f}', va='center')
                    
                    ax.set_title(f"Feature Contributions for Anomaly {i}")
                    ax.set_xlabel("Z-Score (Standard Deviations from Normal)")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Transaction details
                    st.write("#### Transaction Details")
                    anomaly_row = anomalies.loc[anomaly_idx]
                    st.write(f"- Amount: ₹{anomaly_row['amount']:.2f}")
                    st.write(f"- Fee: ₹{anomaly_row['happay_fee_amount']:.2f}")
                    st.write(f"- MCC: {anomaly_row['mcc']}")
                    st.write(f"- Date/Time: {anomaly_row['transmission_date_time']}")
                    st.write(f"- Channel: {anomaly_row['channel']}")
                    st.write(f"- Currency: {anomaly_row['txn_currency_code']}")
                    
                    # User comparison
                    st.write("#### Comparison to User History")
                    user_id = anomaly_row['user_id']
                    user_txns = data[data['user_id'] == user_id]
                    
                    if len(user_txns) > 1:  # Only if user has multiple transactions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Amount history
                            fig, ax = plt.figure(figsize=(8, 4)), plt.gca()
                            sns.histplot(user_txns['amount'], bins=10, kde=True, ax=ax)
                            ax.axvline(anomaly_row['amount'], color='red', linestyle='--',
                                      label=f'Current (₹{anomaly_row["amount"]:.2f})')
                            ax.set_title("User Transaction Amount History")
                            ax.legend()
                            st.pyplot(fig)
                            
                        with col2:
                            # Time of day pattern
                            fig, ax = plt.figure(figsize=(8, 4)), plt.gca()
                            sns.histplot(user_txns['hour'], bins=24, kde=True, ax=ax)
                            ax.axvline(anomaly_row['hour'], color='red', linestyle='--',
                                      label=f'Current ({anomaly_row["hour"]}:00)')
                            ax.set_title("User Transaction Time Pattern")
                            ax.set_xticks(range(0, 24, 3))
                            ax.legend()
                            st.pyplot(fig)
                    
                    st.markdown("---")

            # Overall feature importance
            st.write("## 📈 Overall Anomaly Analysis")
            
            # PCA visualization of anomalies
            try:
                st.write("### Visualization of Anomalies")
                
                # Use PCA to reduce dimensionality to 2D
                pca = PCA(n_components=2)
                features_reduced = pca.fit_transform(features)
                
                # Create a DataFrame with the PCA results
                pca_df = pd.DataFrame(features_reduced, columns=['PC1', 'PC2'])
                pca_df['is_anomaly'] = data['is_anomaly']
                
                # Plot
                fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
                scatter = ax.scatter(
                    pca_df['PC1'],
                    pca_df['PC2'],
                    c=pca_df['is_anomaly'],
                    cmap='coolwarm',
                    alpha=0.6
                )
                
                # Highlight the anomalies specifically
                anomaly_points = pca_df.loc[anomalies.index]
                ax.scatter(
                    anomaly_points['PC1'],
                    anomaly_points['PC2'],
                    color='red',
                    s=100,
                    edgecolors='black',
                    label='Anomalies'
                )
                
                ax.set_title("PCA Visualization of Transactions")
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"PCA visualization failed: {str(e)}")
        else:
            st.success("✅ No anomalies found!")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("📁 Please upload a CSV file to begin analysis.")
