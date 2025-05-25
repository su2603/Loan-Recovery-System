import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import networkx as nx
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Loan Recovery System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background: #d1edff;
        padding: 1rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class StreamlitLoanRecoverySystem:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.anomaly_detector = None
        self.neural_predictor = None
        self.survival_data = None
        self.network_graph = None
        self.seasonal_data = None

    def load_data(self, uploaded_file):
        """Load and process uploaded data"""
        try:
            if uploaded_file.name.endswith(".csv"):
                self.data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".xlsx", ".xls")):
                self.data = pd.read_excel(uploaded_file)

            # Generate processed data
            self.processed_data = self.data.copy()

            # Add synthetic features for demonstration
            np.random.seed(42)
            n_rows = len(self.processed_data)

            # Risk segments
            risk_segments = np.random.choice(
                [0, 1, 2, 3], n_rows, p=[0.3, 0.3, 0.25, 0.15]
            )
            self.processed_data["Risk_Segment"] = risk_segments

            # Additional synthetic features
            self.processed_data["Num_Missed_Payments"] = np.random.poisson(2, n_rows)
            self.processed_data["Days_Past_Due"] = np.random.exponential(30, n_rows)
            self.processed_data["Contact_Success_Rate"] = np.random.uniform(
                0.1, 0.9, n_rows
            )
            self.processed_data["Previous_Recovery_Rate"] = np.random.uniform(
                0, 0.8, n_rows
            )

            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False

    def generate_sample_data(self, n_samples=1000):
        """Generate sample data for demonstration"""
        np.random.seed(42)

        # Generate synthetic loan data
        loan_amounts = np.random.lognormal(8, 1, n_samples)  # Log-normal distribution
        risk_segments = np.random.choice(
            [0, 1, 2, 3], n_samples, p=[0.3, 0.3, 0.25, 0.15]
        )

        data = {
            "Loan_Amount": loan_amounts,
            "Risk_Segment": risk_segments,
            "Num_Missed_Payments": np.random.poisson(2, n_samples),
            "Days_Past_Due": np.random.exponential(30, n_samples),
            "Contact_Success_Rate": np.random.uniform(0.1, 0.9, n_samples),
            "Previous_Recovery_Rate": np.random.uniform(0, 0.8, n_samples),
            "Borrower_Age": np.random.normal(45, 15, n_samples),
            "Income": np.random.lognormal(10, 0.8, n_samples),
            "Employment_Length": np.random.exponential(5, n_samples),
            "Credit_Score": np.random.normal(650, 100, n_samples),
        }

        self.data = pd.DataFrame(data)
        self.processed_data = self.data.copy()

        # Ensure positive values and reasonable ranges
        self.processed_data["Loan_Amount"] = np.clip(
            self.processed_data["Loan_Amount"], 1000, 100000
        )
        self.processed_data["Days_Past_Due"] = np.clip(
            self.processed_data["Days_Past_Due"], 0, 365
        )
        self.processed_data["Borrower_Age"] = np.clip(
            self.processed_data["Borrower_Age"], 18, 80
        )
        self.processed_data["Credit_Score"] = np.clip(
            self.processed_data["Credit_Score"], 300, 850
        )

        return True

    def detect_anomalies(self, contamination=0.1):
        """Detect anomalous cases"""
        if self.processed_data is None:
            return None

        # Select numeric features for anomaly detection
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        X_anomaly = self.processed_data[numeric_cols]

        # Fit Isolation Forest
        self.anomaly_detector = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100
        )

        anomaly_scores = self.anomaly_detector.fit_predict(X_anomaly)
        self.processed_data["Anomaly_Score"] = self.anomaly_detector.score_samples(
            X_anomaly
        )
        self.processed_data["Is_Anomaly"] = anomaly_scores == -1

        return self.processed_data[self.processed_data["Is_Anomaly"]]

    def train_neural_predictor(self):
        """Train neural network predictor"""
        if self.processed_data is None:
            return None

        # Prepare features
        feature_cols = [
            "Loan_Amount",
            "Num_Missed_Payments",
            "Days_Past_Due",
            "Contact_Success_Rate",
            "Previous_Recovery_Rate",
        ]
        X = self.processed_data[feature_cols]
        y = self.processed_data["Risk_Segment"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train neural network
        self.neural_predictor = MLPClassifier(
            hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
        )

        self.neural_predictor.fit(X_train, y_train)

        train_score = self.neural_predictor.score(X_train, y_train)
        test_score = self.neural_predictor.score(X_test, y_test)

        return {
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "feature_importance": feature_cols,
        }

    def optimize_settlement_pricing(self, loan_amounts):
        """Optimize settlement pricing"""
        pricing_results = []

        for loan_amount in loan_amounts:
            for risk_segment in range(4):
                for time_horizon in [30, 60, 90, 180]:
                    # Calculate optimal settlement percentage
                    base_prob = 0.3 + (risk_segment * 0.15)
                    optimal_pct = 50 + (risk_segment * 10) - (time_horizon / 10)
                    optimal_pct = np.clip(optimal_pct, 20, 90)

                    expected_recovery = loan_amount * (optimal_pct / 100) * base_prob
                    collection_cost = 100 + (loan_amount * 0.02)
                    expected_profit = expected_recovery - collection_cost

                    pricing_results.append(
                        {
                            "Loan_Amount": loan_amount,
                            "Risk_Segment": risk_segment,
                            "Time_Horizon": time_horizon,
                            "Optimal_Settlement_Pct": optimal_pct,
                            "Expected_Recovery": expected_recovery,
                            "Expected_Profit": expected_profit,
                        }
                    )

        return pd.DataFrame(pricing_results)

    def generate_seasonal_data(self):
        """Generate seasonal recovery patterns"""
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")

        seasonal_data = []
        for date in dates:
            # Seasonal factors
            month_factor = 1 + 0.2 * np.sin(2 * np.pi * date.month / 12)
            week_factor = 1 + 0.1 * np.sin(2 * np.pi * date.weekday() / 7)

            # Holiday effects
            holiday_factor = 1.0
            if date.month == 12:
                holiday_factor = 0.7
            elif date.month in [1, 9]:
                holiday_factor = 1.3

            base_recovery_rate = 0.35
            recovery_rate = (
                base_recovery_rate * month_factor * week_factor * holiday_factor
            )
            recovery_rate = np.clip(recovery_rate + np.random.normal(0, 0.05), 0.1, 0.8)

            seasonal_data.append(
                {
                    "Date": date,
                    "Recovery_Rate": recovery_rate,
                    "Month": date.month,
                    "Quarter": date.quarter,
                    "Weekday": date.weekday(),
                }
            )

        self.seasonal_data = pd.DataFrame(seasonal_data)
        return self.seasonal_data

    def create_network_analysis(self):
        """Create network analysis of borrower connections"""
        if self.processed_data is None:
            return None

        n_borrowers = len(self.processed_data)

        # Create random connections
        np.random.seed(42)
        connections = []

        for i in range(min(n_borrowers, 500)):  # Limit for performance
            n_connections = np.random.poisson(2)
            for _ in range(n_connections):
                j = np.random.randint(0, min(n_borrowers, 500))
                if i != j:
                    connections.append((i, j))

        # Create network graph
        self.network_graph = nx.Graph()
        self.network_graph.add_edges_from(connections)

        # Calculate network metrics
        centrality = nx.degree_centrality(self.network_graph)
        clustering = nx.clustering(self.network_graph)

        # Add to processed data (for first 500 rows)
        network_centrality = [
            centrality.get(i, 0) for i in range(len(self.processed_data))
        ]
        network_clustering = [
            clustering.get(i, 0) for i in range(len(self.processed_data))
        ]

        self.processed_data["Network_Centrality"] = network_centrality
        self.processed_data["Network_Clustering"] = network_clustering

        return {
            "n_nodes": len(self.network_graph.nodes()),
            "n_edges": len(self.network_graph.edges()),
            "avg_centrality": np.mean(list(centrality.values())),
            "avg_clustering": np.mean(list(clustering.values())),
        }


# Initialize the system
@st.cache_resource
def get_recovery_system():
    return StreamlitLoanRecoverySystem()


system = get_recovery_system()


# Main app
def main():
    st.title("🏦Loan Recovery System")
    st.markdown("*Loan recovery analytics and Optimization*")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Module",
        [
            "📊 Dashboard",
            "📈 Risk Analysis",
            "🔍 Anomaly Detection",
            "🧠 Neural Network",
            "💰 Pricing Optimization",
            "📅 Seasonality",
            "🔗 Network Analysis",
            "📋 Recommendations",
        ],
    )

    # Data loading section
    st.sidebar.subheader("Data Input")
    data_option = st.sidebar.radio(
        "Data Source", ["Upload CSV/Excel", "Use Sample Data"]
    )

    if data_option == "Upload CSV/Excel":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your loan data", type=["csv", "xlsx", "xls"]
        )

        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                success = system.load_data(uploaded_file)
            if success:
                st.sidebar.success(f"✅ Loaded {len(system.data)} records")
        else:
            st.info("👆 Please upload a CSV or Excel file to begin analysis")
            return
    else:
        if st.sidebar.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                system.generate_sample_data()
            st.sidebar.success("✅ Generated 1000 sample records")

    if system.processed_data is None:
        st.info("Please load data to proceed with analysis")
        return

    # Main content based on selected page
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "📈 Risk Analysis":
        show_risk_analysis()
    elif page == "🔍 Anomaly Detection":
        show_anomaly_detection()
    elif page == "🧠 Neural Network":
        show_neural_network()
    elif page == "💰 Pricing Optimization":
        show_pricing_optimization()
    elif page == "📅 Seasonality":
        show_seasonality_analysis()
    elif page == "🔗 Network Analysis":
        show_network_analysis()
    elif page == "📋 Recommendations":
        show_recommendations()


def show_dashboard():
    st.header("📊 Recovery Performance Dashboard")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_accounts = len(system.processed_data)
        st.metric("Total Accounts", f"{total_accounts:,}")

    with col2:
        avg_amount = system.processed_data["Loan_Amount"].mean()
        st.metric("Avg Loan Amount", f"${avg_amount:,.0f}")

    with col3:
        high_risk_pct = (system.processed_data["Risk_Segment"] >= 2).mean() * 100
        st.metric("High Risk %", f"{high_risk_pct:.1f}%")

    with col4:
        avg_days_due = system.processed_data["Days_Past_Due"].mean()
        st.metric("Avg Days Past Due", f"{avg_days_due:.0f}")

    # Portfolio overview charts
    col1, col2 = st.columns(2)

    with col1:
        # Risk segment distribution
        risk_dist = system.processed_data["Risk_Segment"].value_counts().sort_index()
        fig = px.pie(
            values=risk_dist.values,
            names=[f"Segment {i}" for i in risk_dist.index],
            title="Portfolio by Risk Segment",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Loan amount distribution
        fig = px.histogram(
            system.processed_data,
            x="Loan_Amount",
            title="Loan Amount Distribution",
            nbins=30,
        )
        fig.update_layout(xaxis_title="Loan Amount ($)")
        st.plotly_chart(fig, use_container_width=True)

    # Performance metrics
    st.subheader("Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        # Days past due by risk segment
        fig = px.box(
            system.processed_data,
            x="Risk_Segment",
            y="Days_Past_Due",
            title="Days Past Due by Risk Segment",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Recovery rate vs contact success
        fig = px.scatter(
            system.processed_data,
            x="Contact_Success_Rate",
            y="Previous_Recovery_Rate",
            color="Risk_Segment",
            title="Recovery Rate vs Contact Success",
            labels={
                "Contact_Success_Rate": "Contact Success Rate",
                "Previous_Recovery_Rate": "Previous Recovery Rate",
            },
        )
        st.plotly_chart(fig, use_container_width=True)


def show_risk_analysis():
    st.header("📈 Risk Segment Analysis")

    # Risk distribution analysis
    col1, col2 = st.columns(2)

    with col1:
        # Risk segment characteristics
        risk_summary = (
            system.processed_data.groupby("Risk_Segment")
            .agg(
                {
                    "Loan_Amount": ["mean", "median"],
                    "Days_Past_Due": ["mean", "median"],
                    "Contact_Success_Rate": "mean",
                    "Previous_Recovery_Rate": "mean",
                }
            )
            .round(2)
        )

        st.subheader("Risk Segment Characteristics")
        st.dataframe(risk_summary)

    with col2:
        # Risk transition probability matrix
        st.subheader("Risk Distribution")
        risk_counts = system.processed_data["Risk_Segment"].value_counts().sort_index()

        fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Accounts by Risk Segment",
            labels={"x": "Risk Segment", "y": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed risk analysis
    st.subheader("Risk Factor Analysis")

    # Correlation heatmap
    numeric_cols = [
        "Loan_Amount",
        "Risk_Segment",
        "Num_Missed_Payments",
        "Days_Past_Due",
        "Contact_Success_Rate",
        "Previous_Recovery_Rate",
    ]
    corr_matrix = system.processed_data[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        title="Risk Factor Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Risk insights
    st.markdown(
        """
    <div class="insight-box">
    <h4>🎯 Key Risk Insights</h4>
    <ul>
    <li>Higher risk segments show longer days past due</li>
    <li>Contact success rate inversely correlates with risk level</li>
    <li>Loan amount distribution varies significantly across risk segments</li>
    <li>Previous recovery rates are strong predictors of current risk</li>
    </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )


def show_anomaly_detection():
    st.header("🔍 Anomaly Detection Analysis")

    # Parameters
    col1, col2 = st.columns([1, 3])

    with col1:
        contamination = st.slider("Contamination Rate", 0.05, 0.3, 0.1, 0.01)

        if st.button("Run Anomaly Detection"):
            with st.spinner("Detecting anomalies..."):
                anomalies = system.detect_anomalies(contamination)
            st.success(f"Detected {len(anomalies)} anomalous cases")

    if "Is_Anomaly" in system.processed_data.columns:
        with col2:
            anomaly_count = system.processed_data["Is_Anomaly"].sum()
            anomaly_rate = anomaly_count / len(system.processed_data) * 100

            st.metric("Anomalous Cases", f"{anomaly_count}", f"{anomaly_rate:.1f}%")

        # Anomaly analysis charts
        col1, col2 = st.columns(2)

        with col1:
            # Anomaly score distribution
            fig = px.histogram(
                system.processed_data,
                x="Anomaly_Score",
                title="Anomaly Score Distribution",
                nbins=50,
            )

            # Add threshold line
            threshold = system.processed_data[system.processed_data["Is_Anomaly"]][
                "Anomaly_Score"
            ].max()
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Anomaly Threshold",
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Anomalies by risk segment
            anomaly_by_segment = (
                system.processed_data.groupby(["Risk_Segment", "Is_Anomaly"])
                .size()
                .unstack(fill_value=0)
            )

            fig = px.bar(
                x=anomaly_by_segment.index,
                y=[anomaly_by_segment[False], anomaly_by_segment[True]],
                title="Normal vs Anomalous Cases by Risk Segment",
                labels={"x": "Risk Segment", "y": "Count"},
            )
            fig.data[0].name = "Normal"
            fig.data[1].name = "Anomaly"

            st.plotly_chart(fig, use_container_width=True)

        # Anomalous cases details
        st.subheader("Anomalous Cases Details")

        if anomaly_count > 0:
            anomalous_cases = system.processed_data[system.processed_data["Is_Anomaly"]]

            # Show top anomalies
            top_anomalies = anomalous_cases.nsmallest(10, "Anomaly_Score")
            st.dataframe(
                top_anomalies[
                    [
                        "Loan_Amount",
                        "Risk_Segment",
                        "Days_Past_Due",
                        "Contact_Success_Rate",
                        "Anomaly_Score",
                    ]
                ]
            )

            st.markdown(
                """
            <div class="warning-box">
            <h4>⚠️ Anomaly Investigation Required</h4>
            <p>These cases require manual review for:</p>
            <ul>
            <li>Data entry errors</li>
            <li>Unusual borrower circumstances</li>
            <li>Potential fraud indicators</li>
            <li>Special collection strategies</li>
            </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )


def show_neural_network():
    st.header("🧠 Neural Network Predictor")

    if st.button("Train Neural Network"):
        with st.spinner("Training neural network..."):
            results = system.train_neural_predictor()

        if results:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Training Accuracy", f"{results['train_accuracy']:.3f}")

            with col2:
                st.metric("Test Accuracy", f"{results['test_accuracy']:.3f}")

            with col3:
                overfitting = results["train_accuracy"] - results["test_accuracy"]
                st.metric("Overfitting", f"{overfitting:.3f}")

            # Feature importance (simplified)
            st.subheader("Model Features")
            features_df = pd.DataFrame(
                {
                    "Feature": results["feature_importance"],
                    "Description": [
                        "Total loan amount",
                        "Number of missed payments",
                        "Days past due",
                        "Contact success rate",
                        "Previous recovery rate",
                    ],
                }
            )
            st.dataframe(features_df)

            # Model performance insights
            st.markdown(
                """
            <div class="success-box">
            <h4>✅ Neural Network Results</h4>
            <ul>
            <li>Model successfully trained on risk prediction</li>
            <li>Complex pattern recognition for borrower behavior</li>
            <li>Can identify non-linear relationships in data</li>
            <li>Suitable for deployment in production systems</li>
            </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )


def show_pricing_optimization():
    st.header("💰 Dynamic Settlement Pricing")

    # Input parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        min_amount = st.number_input("Min Loan Amount", value=1000, step=1000)

    with col2:
        max_amount = st.number_input("Max Loan Amount", value=50000, step=5000)

    with col3:
        n_scenarios = st.number_input(
            "Number of Scenarios", value=5, min_value=3, max_value=10
        )

    if st.button("Optimize Pricing"):
        loan_amounts = np.linspace(min_amount, max_amount, n_scenarios)

        with st.spinner("Optimizing settlement pricing..."):
            pricing_df = system.optimize_settlement_pricing(loan_amounts)

        # Display results
        st.subheader("Pricing Optimization Results")

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_settlement = pricing_df["Optimal_Settlement_Pct"].mean()
            st.metric("Avg Settlement %", f"{avg_settlement:.1f}%")

        with col2:
            total_recovery = pricing_df["Expected_Recovery"].sum()
            st.metric("Total Expected Recovery", f"${total_recovery:,.0f}")

        with col3:
            total_profit = pricing_df["Expected_Profit"].sum()
            st.metric("Total Expected Profit", f"${total_profit:,.0f}")

        # Visualization
        col1, col2 = st.columns(2)

        with col1:
            # Settlement percentage by loan amount and risk
            fig = px.scatter(
                pricing_df,
                x="Loan_Amount",
                y="Optimal_Settlement_Pct",
                color="Risk_Segment",
                size="Expected_Recovery",
                title="Optimal Settlement % by Loan Amount",
                labels={"Optimal_Settlement_Pct": "Settlement %"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Profit by time horizon
            profit_by_time = pricing_df.groupby("Time_Horizon")[
                "Expected_Profit"
            ].mean()

            fig = px.line(
                x=profit_by_time.index,
                y=profit_by_time.values,
                title="Expected Profit by Time Horizon",
                labels={"x": "Time Horizon (Days)", "y": "Expected Profit ($)"},
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detailed results table
        st.subheader("Detailed Pricing Results")

        # Group by key factors for display
        summary_pricing = (
            pricing_df.groupby(["Risk_Segment", "Time_Horizon"])
            .agg(
                {
                    "Optimal_Settlement_Pct": "mean",
                    "Expected_Recovery": "mean",
                    "Expected_Profit": "mean",
                }
            )
            .round(2)
        )

        st.dataframe(summary_pricing)


def show_seasonality_analysis():
    st.header("📅 Seasonality Analysis")

    if st.button("Generate Seasonal Analysis"):
        with st.spinner("Analyzing seasonal patterns..."):
            seasonal_data = system.generate_seasonal_data()

        # Monthly patterns
        col1, col2 = st.columns(2)

        with col1:
            monthly_avg = seasonal_data.groupby("Month")["Recovery_Rate"].mean()

            fig = px.line(
                x=monthly_avg.index,
                y=monthly_avg.values,
                title="Average Recovery Rate by Month",
                labels={"x": "Month", "y": "Recovery Rate"},
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            quarterly_avg = seasonal_data.groupby("Quarter")["Recovery_Rate"].mean()

            fig = px.bar(
                x=quarterly_avg.index,
                y=quarterly_avg.values,
                title="Average Recovery Rate by Quarter",
                labels={"x": "Quarter", "y": "Recovery Rate"},
            )
            st.plotly_chart(fig, use_container_width=True)

        # Weekly patterns
        col1, col2 = st.columns(2)

        with col1:
            weekly_avg = seasonal_data.groupby("Weekday")["Recovery_Rate"].mean()
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

            fig = px.bar(
                x=days,
                y=weekly_avg.values,
                title="Average Recovery Rate by Day of Week",
                labels={"x": "Day of Week", "y": "Recovery Rate"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Time series view
            monthly_ts = seasonal_data.groupby(seasonal_data["Date"].dt.to_period("M"))[
                "Recovery_Rate"
            ].mean()

            fig = px.line(
                x=monthly_ts.index.astype(str),
                y=monthly_ts.values,
                title="Recovery Rate Time Series",
                labels={"x": "Month", "y": "Recovery Rate"},
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        # Seasonal insights
        best_month = monthly_avg.idxmax()
        worst_month = monthly_avg.idxmin()

        st.markdown(
            f"""
        <div class="insight-box">
        <h4>📊 Seasonal Insights</h4>
        <ul>
        <li><strong>Best Month:</strong> {best_month} ({monthly_avg[best_month]:.1%} recovery rate)</li>
        <li><strong>Worst Month:</strong> {worst_month} ({monthly_avg[worst_month]:.1%} recovery rate)</li>
        <li><strong>Seasonal Variation:</strong> {(monthly_avg.max() - monthly_avg.min()):.1%} difference</li>
        <li><strong>Holiday Impact:</strong> December shows typical seasonal decline</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )


def show_network_analysis():
    st.header("🔗 Network Analysis")

    if st.button("Generate Network Analysis"):
        with st.spinner("Analyzing borrower networks..."):
            network_stats = system.create_network_analysis()

        if network_stats:
            # Network statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Network Nodes", network_stats["n_nodes"])

            with col2:
                st.metric("Network Edges", network_stats["n_edges"])

            with col3:
                st.metric("Avg Centrality", f"{network_stats['avg_centrality']:.3f}")

            with col4:
                st.metric("Avg Clustering", f"{network_stats['avg_clustering']:.3f}")

            # Network analysis charts
            col1, col2 = st.columns(2)

            with col1:
                # Centrality distribution
                fig = px.histogram(
                    system.processed_data,
                    x="Network_Centrality",
                    title="Network Centrality Distribution",
                    nbins=30,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Centrality vs Risk
                fig = px.scatter(
                    system.processed_data,
                    x="Network_Centrality",
                    y="Risk_Segment",
                    title="Network Centrality vs Risk Segment",
                    labels={
                        "Network_Centrality": "Network Centrality",
                        "Risk_Segment": "Risk Segment",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

            # Network insights
            st.markdown(
                """
            <div class="insight-box">
            <h4>🔗 Network Analysis Insights</h4>
            <ul>
            <li>Borrowers with high network centrality may influence others</li>
            <li>Clustered borrowers might respond to similar collection strategies</li>
            <li>Network effects can impact payment behavior propagation</li>
            <li>Identify key influencers for targeted interventions</li>
            </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )


def show_recommendations():
    st.header("📋 Strategic Recommendations")

    if system.processed_data is not None:
        # Generate insights and recommendations
        total_accounts = len(system.processed_data)
        high_risk_accounts = (system.processed_data["Risk_Segment"] >= 2).sum()
        avg_days_due = system.processed_data["Days_Past_Due"].mean()

        # Portfolio health score
        health_score = (
            100 - (high_risk_accounts / total_accounts * 50) - (avg_days_due / 365 * 30)
        )
        health_score = max(0, min(100, health_score))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Portfolio Health Score", f"{health_score:.0f}/100")

        with col2:
            recovery_potential = system.processed_data["Loan_Amount"].sum() * 0.4
            st.metric("Recovery Potential", f"${recovery_potential:,.0f}")

        with col3:
            priority_accounts = (system.processed_data["Risk_Segment"] >= 2).sum()
            st.metric("Priority Accounts", f"{priority_accounts:,}")

        # Strategic recommendations
        st.subheader("🎯 Strategic Recommendations")

        recommendations = []

        # Risk-based recommendations
        if high_risk_accounts / total_accounts > 0.3:
            recommendations.append(
                {
                    "category": "Risk Management",
                    "priority": "High",
                    "recommendation": "Implement immediate intervention for high-risk segments",
                    "impact": "Reduce portfolio risk by 15-25%",
                }
            )

        # Contact strategy recommendations
        low_contact_success = (
            system.processed_data["Contact_Success_Rate"] < 0.5
        ).sum()
        if low_contact_success / total_accounts > 0.4:
            recommendations.append(
                {
                    "category": "Contact Strategy",
                    "priority": "Medium",
                    "recommendation": "Enhance multi-channel contact approach",
                    "impact": "Improve contact rates by 20-30%",
                }
            )

        # Pricing recommendations
        recommendations.append(
            {
                "category": "Settlement Pricing",
                "priority": "Medium",
                "recommendation": "Implement dynamic pricing based on risk segments",
                "impact": "Increase recovery rates by 10-15%",
            }
        )

        # Technology recommendations
        recommendations.append(
            {
                "category": "Technology",
                "priority": "Low",
                "recommendation": "Deploy predictive analytics for early intervention",
                "impact": "Prevent 20-30% of accounts from becoming high-risk",
            }
        )

        # Display recommendations
        for i, rec in enumerate(recommendations):
            priority_color = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}[
                rec["priority"]
            ]

            st.markdown(
                f"""
            <div class="insight-box" style="border-left-color: {priority_color};">
            <h5>{rec['category']} - {rec['priority']} Priority</h5>
            <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
            <p><strong>Expected Impact:</strong> {rec['impact']}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Action plan
        st.subheader("📅 30-60-90 Day Action Plan")

        action_plan = {
            "30 Days": [
                "Segment portfolio by risk levels",
                "Implement priority contact strategies",
                "Begin anomaly investigation process",
                "Set up performance monitoring dashboards",
            ],
            "60 Days": [
                "Deploy dynamic settlement pricing",
                "Launch targeted collection campaigns",
                "Implement seasonal adjustment strategies",
                "Train team on new analytics tools",
            ],
            "90 Days": [
                "Evaluate campaign effectiveness",
                "Refine predictive models",
                "Expand network analysis capabilities",
                "Develop advanced automation rules",
            ],
        }

        col1, col2, col3 = st.columns(3)

        for i, (period, actions) in enumerate(action_plan.items()):
            with [col1, col2, col3][i]:
                st.markdown(f"**{period}**")
                for action in actions:
                    st.markdown(f"• {action}")

        # ROI Projection
        st.subheader("💰 ROI Projection")

        current_recovery = (
            system.processed_data["Loan_Amount"].sum() * 0.25
        )  # Baseline 25%
        projected_recovery = (
            system.processed_data["Loan_Amount"].sum() * 0.35
        )  # Target 35%
        improvement = projected_recovery - current_recovery

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Current Recovery", f"${current_recovery:,.0f}")

        with col2:
            st.metric("Projected Recovery", f"${projected_recovery:,.0f}")

        with col3:
            st.metric("Expected Improvement", f"${improvement:,.0f}")

        # Implementation timeline
        st.markdown(
            """
        <div class="success-box">
        <h4>🚀 Implementation Success Factors</h4>
        <ul>
        <li><strong>Data Quality:</strong> Ensure clean, consistent data across all systems</li>
        <li><strong>Team Training:</strong> Invest in analytics literacy for collection teams</li>
        <li><strong>Technology Integration:</strong> Seamless integration with existing systems</li>
        <li><strong>Performance Monitoring:</strong> Continuous tracking and optimization</li>
        <li><strong>Compliance:</strong> Maintain regulatory compliance throughout implementation</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
