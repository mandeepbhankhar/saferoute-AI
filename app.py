
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="SafeRoute AI",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 SafeRoute AI")
st.subheader("Smart Road Safety Analytics System")
st.caption("Predict Risks. Prevent Accidents. Protect Lives.")

st.info(
    "Demo mode: Sample data is synthetic. "
    "It does not represent verified real-world accident statistics."
)

# ---------------- LOAD DATA ----------------
uploaded_file = st.sidebar.file_uploader(
    "Upload accident CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Uploaded CSV loaded.")
    except Exception as error:
        st.error(f"CSV could not be read: {error}")
        st.stop()
else:
    sample_path = Path(__file__).parent / "sample_accidents.csv"

    if sample_path.exists():
        df = pd.read_csv(sample_path)
        st.sidebar.success("Sample data loaded automatically.")
    else:
        st.warning(
            "sample_accidents.csv not found. "
            "Upload a CSV file using the sidebar."
        )
        st.stop()

# Clean column names and validate essential columns
df.columns = df.columns.str.strip()

required_columns = ["location", "severity"]
missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    st.error(f"Required columns missing: {missing}")
    st.write("Available columns:", list(df.columns))
    st.stop()

# Add optional columns if the uploaded CSV does not contain them
optional_defaults = {
    "date": "",
    "weather": "Unknown",
    "road_condition": "Unknown",
    "time_of_day": "Unknown",
    "fatalities": 0,
    "injuries": 0,
    "vehicles_involved": 0,
    "accident_id": ""
}

for column, default in optional_defaults.items():
    if column not in df.columns:
        df[column] = default

# Standardize text columns
for column in [
    "location", "severity", "weather",
    "road_condition", "time_of_day"
]:
    df[column] = df[column].fillna("Unknown").astype(str).str.strip()
    df[column] = df[column].replace("", "Unknown")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

for column in ["fatalities", "injuries", "vehicles_involved"]:
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("SafeRoute AI")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Accident Analysis",
        "Accident Trends",
        "Weather Analysis",
        "Hotspot Analysis",
        "ML Insights",
        "Reports"
    ]
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.header("Road Safety Dashboard")

    total_accidents = len(df)
    fatal_accidents = int(
        df["severity"].str.lower().eq("fatal").sum()
    )
    locations_count = df["location"].nunique()
    total_injuries = int(df["injuries"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", total_accidents)
    c2.metric("Fatal-Severity Records", fatal_accidents)
    c3.metric("Locations Analysed", locations_count)
    c4.metric("Reported Injuries", total_injuries)

    st.caption(
        "Fatal-severity records count rows labelled 'Fatal'. "
        "They are not necessarily the same as total fatalities."
    )

    st.subheader("Accidents by Severity")
    severity_counts = df["severity"].value_counts()

    fig, ax = plt.subplots()
    severity_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Severity")
    ax.set_ylabel("Number of Records")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Accidents by Time of Day")
    time_counts = df["time_of_day"].value_counts()

    fig, ax = plt.subplots()
    time_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Number of Records")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------- ACCIDENT ANALYSIS ----------------
elif page == "Accident Analysis":
    st.header("Accident Data Analysis")

    locations = sorted(df["location"].unique().tolist())
    selected_locations = st.multiselect(
        "Filter by location",
        locations,
        default=locations
    )

    severities = sorted(df["severity"].unique().tolist())
    selected_severities = st.multiselect(
        "Filter by severity",
        severities,
        default=severities
    )

    filtered = df[
        df["location"].isin(selected_locations)
        & df["severity"].isin(selected_severities)
    ]

    st.metric("Matching Records", len(filtered))
    st.dataframe(filtered, width="stretch")

    st.subheader("Missing Values")
    missing_values = filtered.isnull().sum().reset_index()
    missing_values.columns = ["Column", "Missing Values"]
    st.dataframe(missing_values, width="stretch")

# ---------------- ACCIDENT TRENDS ----------------
elif page == "Accident Trends":
    st.header("📈 Accident Trends")

    dated_df = df.dropna(subset=["date"]).copy()

    if dated_df.empty:
        st.warning(
            "Valid dates were not found in this CSV. "
            "Check that the date column contains dates."
        )
    else:
        dated_df["day"] = dated_df["date"].dt.date
        daily = (
            dated_df.groupby("day")
            .size()
            .reset_index(name="Accident Records")
        )

        fig, ax = plt.subplots()
        ax.plot(
            daily["day"],
            daily["Accident Records"],
            marker="o"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Accident Records")
        ax.set_title("Daily Recorded Accidents")
        fig.autofmt_xdate()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        dated_df["month"] = dated_df["date"].dt.to_period("M").astype(str)
        monthly = (
            dated_df.groupby("month")
            .size()
            .reset_index(name="Accident Records")
        )

        st.subheader("Monthly Accident Records")
        st.dataframe(monthly, width="stretch")

        fig, ax = plt.subplots()
        ax.bar(monthly["month"], monthly["Accident Records"])
        ax.set_xlabel("Month")
        ax.set_ylabel("Accident Records")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "These charts show records present in the dataset, "
            "not a prediction of future accidents."
        )

# ---------------- WEATHER ANALYSIS ----------------
elif page == "Weather Analysis":
    st.header("🌦️ Weather & Road Condition Analysis")

    st.subheader("Records by Weather")
    weather_counts = (
        df["weather"].value_counts()
        .rename_axis("Weather")
        .reset_index(name="Accident Records")
    )
    st.dataframe(weather_counts, width="stretch")

    fig, ax = plt.subplots()
    ax.bar(
        weather_counts["Weather"],
        weather_counts["Accident Records"]
    )
    ax.set_xlabel("Weather")
    ax.set_ylabel("Accident Records")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Records by Road Condition")
    road_counts = (
        df["road_condition"].value_counts()
        .rename_axis("Road Condition")
        .reset_index(name="Accident Records")
    )
    st.dataframe(road_counts, width="stretch")

    fig, ax = plt.subplots()
    ax.bar(
        road_counts["Road Condition"],
        road_counts["Accident Records"]
    )
    ax.set_xlabel("Road Condition")
    ax.set_ylabel("Accident Records")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "This is a comparison of recorded cases only. "
        "It does not prove that weather or road conditions caused accidents."
    )

# ---------------- HOTSPOT ANALYSIS ----------------
elif page == "Hotspot Analysis":
    st.header("📍 Historical Accident Hotspots")

    hotspots = (
        df.groupby("location")
        .size()
        .reset_index(name="Recorded Accidents")
        .sort_values("Recorded Accidents", ascending=False)
    )

    st.write(
        "Locations ranked by the number of records in this dataset."
    )
    st.dataframe(hotspots, width="stretch")

    fig, ax = plt.subplots()
    ax.barh(
        hotspots["location"],
        hotspots["Recorded Accidents"]
    )
    ax.set_xlabel("Recorded Accidents")
    ax.set_ylabel("Location")
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "A higher count does not necessarily mean a higher risk: "
        "traffic volume and observation period are not accounted for."
    )
    
# ---------------- MACHINE LEARNING INSIGHTS ----------------
elif page == "ML Insights":
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    st.header("🤖 Machine Learning Insights")
    st.write(
        "K-Means clustering groups similar accident records "
        "based on their recorded conditions."
    )

    st.warning(
        "Demo analysis only: the sample dataset is synthetic. "
        "Clusters are not predictions of future accidents."
    )

    feature_columns = [
        "weather",
        "road_condition",
        "time_of_day",
        "severity"
    ]

    ml_data = df[feature_columns].copy()

    # Replace missing values and encode text categories
    for col in feature_columns:
        ml_data[col] = (
            ml_data[col]
            .fillna("Unknown")
            .astype(str)
        )

    encoded = pd.get_dummies(ml_data, dtype=float)

    if len(df) < 3 or encoded.shape[1] == 0:
        st.error(
            "Clustering ke liye kam se kam 3 records chahiye."
        )
    else:
        number_of_clusters = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=min(6, len(df) - 1),
            value=3
        )

        # Scale features before clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(encoded)

        # Train the K-Means model
        model = KMeans(
            n_clusters=number_of_clusters,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(scaled_data)

        results = df.copy()
        results["ML_Cluster"] = labels + 1

        st.subheader("Cluster Summary")

        summary = (
            results.groupby("ML_Cluster")
            .agg(
                Records=("location", "size"),
                Locations=("location", "nunique")
            )
            .reset_index()
        )

        st.dataframe(summary, width="stretch")

        st.subheader("Records per Cluster")

        cluster_counts = (
            results["ML_Cluster"]
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots()
        cluster_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Records")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Common Conditions in Each Cluster")

        for cluster_id in sorted(results["ML_Cluster"].unique()):
            cluster_df = results[
                results["ML_Cluster"] == cluster_id
            ]

            with st.expander(
                f"Cluster {cluster_id} — "
                f"{len(cluster_df)} records"
            ):
                for col in feature_columns:
                    common_value = (
                        cluster_df[col]
                        .mode()
                        .iloc[0]
                    )
                    st.write(f"**{col.replace('_', ' ').title()}:** {common_value}")

                st.dataframe(
                    cluster_df[
                        [
                            "location",
                            "severity",
                            "weather",
                            "road_condition",
                            "time_of_day"
                        ]
                    ].head(10),
                    width="stretch"
                )

        st.subheader("Download ML Results")

        st.download_button(
            "Download Cluster Results CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="saferoute_ml_clusters.csv",
            mime="text/csv"
        )


# ---------------- REPORTS ----------------
elif page == "Reports":
    st.header("📄 Download Reports")

    st.write(
        "Download the loaded dataset or summary reports "
        "for your project presentation."
    )

    st.subheader("1. Full Dataset")
    full_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Dataset (CSV)",
        data=full_csv,
        file_name="saferoute_dataset.csv",
        mime="text/csv"
    )

    st.subheader("2. Severity Summary")
    severity_report = (
        df["severity"].value_counts()
        .rename_axis("Severity")
        .reset_index(name="Accident Records")
    )

    st.dataframe(severity_report, width="stretch")

    st.download_button(
        label="Download Severity Report",
        data=severity_report.to_csv(index=False).encode("utf-8"),
        file_name="saferoute_severity_report.csv",
        mime="text/csv"
    )

    st.subheader("3. Location Summary")
    location_report = (
        df.groupby("location")
        .size()
        .reset_index(name="Recorded Accidents")
        .sort_values("Recorded Accidents", ascending=False)
    )

    st.dataframe(location_report, width="stretch")

    st.download_button(
        label="Download Location Report",
        data=location_report.to_csv(index=False).encode("utf-8"),
        file_name="saferoute_location_report.csv",
        mime="text/csv"
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "SafeRoute AI | Python | Streamlit | Pandas | Matplotlib"
)
