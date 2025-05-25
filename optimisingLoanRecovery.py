import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")


class LoanRecoveryStrategyModel:
    def __init__(self, data_path=None):
        """
        Initialize the Loan Recovery Strategy Model
        """
        self.data = None
        self.processed_data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.kmeans = None
        self.recovery_model = None
        self.early_warning_model = None
        self.cost_matrix = self._define_cost_matrix()

    def _define_cost_matrix(self):
        """
        Define costs and recovery rates for different collection methods
        """
        return {
            "Phone_Call": {"cost": 25, "avg_recovery_rate": 0.15, "time_days": 7},
            "Email": {"cost": 5, "avg_recovery_rate": 0.08, "time_days": 3},
            "Debt_Collector": {"cost": 200, "avg_recovery_rate": 0.45, "time_days": 30},
            "Legal_Action": {"cost": 1500, "avg_recovery_rate": 0.65, "time_days": 90},
            "Settlement_Offer": {
                "cost": 100,
                "avg_recovery_rate": 0.35,
                "time_days": 14,
            },
        }

    def load_and_prepare_data(self, df):
        """
        Load and preprocess the loan recovery data
        """
        self.data = df.copy()
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")

        # Create processed copy
        self.processed_data = self.data.copy()

        # Drop ID columns if they exist
        id_cols = [
            col for col in ["Customer_ID", "Loan_ID"] if col in self.data.columns
        ]
        self.data.drop(columns=id_cols, inplace=True)

        # Handle missing values
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        categorical_cols = self.data.select_dtypes(include="object").columns

        # Fill missing numeric with median
        self.data[numeric_cols] = self.data[numeric_cols].fillna(
            self.data[numeric_cols].median()
        )

        # Fill missing categorical with mode
        for col in categorical_cols:
            if self.data[col].isnull().any():
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)

        # Proceed with encoding
        self.label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le

        # Scale numeric features
        self.scaler = StandardScaler()
        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])

        self.processed_data = self.data.copy()
        print("Data preprocessing completed!")
        return self.processed_data

    def segment_borrowers(self, n_clusters=4):
        """
        Segment borrowers into risk categories using clustering
        """
        # Select features for clustering (adjust based on available columns)
        available_cols = self.processed_data.columns.tolist()

        # Define potential clustering features
        potential_features = [
            "Loan_Amount",
            "Monthly_Income",
            "Num_Missed_Payments",
            "Days_Past_Due",
            "Payment_History",
            "Age",
            "Credit_Score",
        ]

        # Use available features
        cluster_features = [col for col in potential_features if col in available_cols]

        if not cluster_features:
            # Use first few numerical columns if specific ones aren't available
            cluster_features = available_cols[:5]

        print(f"Using features for clustering: {cluster_features}")

        X_cluster = self.processed_data[cluster_features]

        # Apply KMeans clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.processed_data["Risk_Segment"] = self.kmeans.fit_predict(X_cluster)

        # Analyze segments
        # Build aggregation dictionary dynamically
        agg_dict = {}
        if len(cluster_features) > 1:
            agg_dict[cluster_features[0]] = ["mean", "std"]
            agg_dict[cluster_features[1]] = ["mean", "std", "count"]
        else:
            agg_dict[cluster_features[0]] = ["mean", "std", "count"]

        segment_analysis = (
            self.processed_data.groupby("Risk_Segment").agg(agg_dict).round(2)
        )

        print("Borrower Segmentation Analysis:")
        print(segment_analysis)

        # Visualize clusters
        self._visualize_segments(X_cluster)

        return segment_analysis

    def _visualize_segments(self, X_cluster):
        """
        Visualize borrower segments using PCA
        """
        # Reduce dimensions for visualization
        pca = PCA(n_components=2)
        cluster_2d = pca.fit_transform(X_cluster)

        plt.figure(figsize=(12, 5))

        # Plot 1: Cluster visualization
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(
            cluster_2d[:, 0],
            cluster_2d[:, 1],
            c=self.processed_data["Risk_Segment"],
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(scatter)
        plt.title("Borrower Risk Segments\n(PCA Projection)")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")

        # Plot 2: Segment distribution
        plt.subplot(1, 2, 2)
        segment_counts = self.processed_data["Risk_Segment"].value_counts().sort_index()
        plt.bar(range(len(segment_counts)), segment_counts.values)
        plt.title("Distribution of Risk Segments")
        plt.xlabel("Risk Segment")
        plt.ylabel("Number of Borrowers")
        plt.xticks(
            range(len(segment_counts)), [f"Segment {i}" for i in segment_counts.index]
        )

        plt.tight_layout()
        plt.show()

    def train_recovery_optimization_model(self, target_col=None):
        """
        Train model to predict optimal collection method
        """
        # Try to identify target column
        possible_targets = [
            "Collection_Method",
            "Recovery_Method",
            "collection_method",
            "recovery_method",
        ]

        if target_col is None:
            for col in possible_targets:
                if col in self.processed_data.columns:
                    target_col = col
                    break

        if target_col is None or target_col not in self.processed_data.columns:
            print("No collection method column found. Creating synthetic target...")
            # Create synthetic target based on risk segments and other factors
            self.processed_data["Optimal_Method"] = self._create_synthetic_target()
            target_col = "Optimal_Method"

        # Prepare features and target
        X = self.processed_data.drop(columns=[target_col])
        y = self.processed_data[target_col]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Random Forest model
        self.recovery_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
        )

        self.recovery_model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = self.recovery_model.predict(X_test)
        y_prob = self.recovery_model.predict_proba(X_test)

        # Print results
        print("Collection Method Optimization Model Results:")
        print("=" * 50)
        print(classification_report(y_test, y_pred))

        # Feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": self.recovery_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))

        self._plot_model_performance(y_test, y_pred, feature_importance)

        return self.recovery_model

    def _create_synthetic_target(self):
        """
        Create synthetic optimal collection method based on borrower characteristics
        """
        methods = []
        for _, row in self.processed_data.iterrows():
            # Simple rule-based assignment (you can make this more sophisticated)
            risk_segment = row["Risk_Segment"]

            # Assume higher risk segments need more aggressive methods
            if risk_segment == 0:  # Low risk
                methods.append(0)  # Phone_Call
            elif risk_segment == 1:  # Medium-low risk
                methods.append(1)  # Email or Phone_Call
            elif risk_segment == 2:  # Medium-high risk
                methods.append(2)  # Debt_Collector
            else:  # High risk
                methods.append(3)  # Legal_Action

        return methods

    def _plot_model_performance(self, y_test, y_pred, feature_importance):
        """
        Plot model performance metrics
        """
        plt.figure(figsize=(15, 5))

        # Plot 1: Confusion Matrix
        plt.subplot(1, 3, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        # Plot 2: Feature Importance
        plt.subplot(1, 3, 2)
        top_features = feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.title("Top 10 Feature Importance")
        plt.xlabel("Importance")

        # Plot 3: Class Distribution
        plt.subplot(1, 3, 3)
        unique, counts = np.unique(y_test, return_counts=True)
        plt.bar(unique, counts, alpha=0.7, label="Actual")
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        plt.bar(unique_pred, counts_pred, alpha=0.7, label="Predicted")
        plt.title("Class Distribution")
        plt.xlabel("Collection Method")
        plt.ylabel("Count")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def develop_early_warning_system(self):
        """
        Develop early warning system to predict default risk
        """
        # Create default risk target (synthetic if not available)
        if "Recovery_Status" in self.processed_data.columns:
            # Assume Recovery_Status indicates success/failure
            y_default = (self.processed_data["Recovery_Status"] == 0).astype(int)
        else:
            # Create synthetic default risk based on risk segments
            y_default = (self.processed_data["Risk_Segment"] >= 2).astype(int)

        # Features for early warning
        warning_features = [
            col
            for col in self.processed_data.columns
            if col not in ["Recovery_Status", "Risk_Segment"]
        ]

        X_warning = self.processed_data[warning_features]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_warning, y_default, test_size=0.2, random_state=42
        )

        # Train Gradient Boosting model for early warning
        self.early_warning_model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )

        self.early_warning_model.fit(X_train, y_train)

        # Predictions
        y_pred = self.early_warning_model.predict(X_test)
        y_prob = self.early_warning_model.predict_proba(X_test)[:, 1]

        # Calculate AUC score
        auc_score = roc_auc_score(y_test, y_prob)

        print("Early Warning System Results:")
        print("=" * 40)
        print(f"AUC Score: {auc_score:.3f}")
        print(classification_report(y_test, y_pred))

        # Feature importance for early warning
        warning_importance = pd.DataFrame(
            {
                "feature": warning_features,
                "importance": self.early_warning_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print("\nEarly Warning - Top Risk Indicators:")
        print(warning_importance.head(10))

        return self.early_warning_model

    def calculate_roi_optimization(self):
        """
        Calculate ROI for different collection strategies
        """
        print("ROI Analysis for Collection Methods:")
        print("=" * 50)

        roi_analysis = []

        for method, params in self.cost_matrix.items():
            # Simulate different loan amounts
            loan_amounts = [1000, 5000, 10000, 25000, 50000]

            for loan_amount in loan_amounts:
                recovered_amount = loan_amount * params["avg_recovery_rate"]
                net_recovery = recovered_amount - params["cost"]
                roi = (net_recovery / params["cost"]) * 100 if params["cost"] > 0 else 0

                roi_analysis.append(
                    {
                        "Method": method,
                        "Loan_Amount": loan_amount,
                        "Cost": params["cost"],
                        "Recovery_Rate": params["avg_recovery_rate"],
                        "Recovered_Amount": recovered_amount,
                        "Net_Recovery": net_recovery,
                        "ROI_Percent": roi,
                        "Time_Days": params["time_days"],
                    }
                )

        roi_df = pd.DataFrame(roi_analysis)

        # Plot ROI analysis
        plt.figure(figsize=(15, 10))

        # Plot 1: ROI by Method and Loan Amount
        plt.subplot(2, 2, 1)
        pivot_roi = roi_df.pivot(
            index="Loan_Amount", columns="Method", values="ROI_Percent"
        )
        sns.heatmap(pivot_roi, annot=True, fmt=".1f", cmap="RdYlGn")
        plt.title("ROI (%) by Collection Method and Loan Amount")

        # Plot 2: Net Recovery by Method
        plt.subplot(2, 2, 2)
        avg_net_recovery = roi_df.groupby("Method")["Net_Recovery"].mean()
        plt.bar(range(len(avg_net_recovery)), avg_net_recovery.values)
        plt.xticks(range(len(avg_net_recovery)), avg_net_recovery.index, rotation=45)
        plt.title("Average Net Recovery by Method")
        plt.ylabel("Net Recovery ($)")

        # Plot 3: Cost vs Recovery Rate
        plt.subplot(2, 2, 3)
        method_summary = (
            roi_df.groupby("Method")
            .agg({"Cost": "first", "Recovery_Rate": "first"})
            .reset_index()
        )

        plt.scatter(method_summary["Cost"], method_summary["Recovery_Rate"], s=100)
        for i, method in enumerate(method_summary["Method"]):
            plt.annotate(
                method,
                (
                    method_summary["Cost"].iloc[i],
                    method_summary["Recovery_Rate"].iloc[i],
                ),
            )
        plt.xlabel("Cost ($)")
        plt.ylabel("Recovery Rate")
        plt.title("Cost vs Recovery Rate by Method")

        # Plot 4: Time vs ROI
        plt.subplot(2, 2, 4)
        avg_roi_time = (
            roi_df.groupby("Method")
            .agg({"ROI_Percent": "mean", "Time_Days": "first"})
            .reset_index()
        )

        plt.scatter(avg_roi_time["Time_Days"], avg_roi_time["ROI_Percent"], s=100)
        for i, method in enumerate(avg_roi_time["Method"]):
            plt.annotate(
                method,
                (
                    avg_roi_time["Time_Days"].iloc[i],
                    avg_roi_time["ROI_Percent"].iloc[i],
                ),
            )
        plt.xlabel("Time to Complete (Days)")
        plt.ylabel("Average ROI (%)")
        plt.title("Time vs ROI by Method")

        plt.tight_layout()
        plt.show()

        return roi_df

    def predict_optimal_strategy(self, borrower_data):
        """
        Predict optimal collection strategy for a specific borrower
        """
        if self.recovery_model is None:
            print("Please train the recovery model first!")
            return None

        # Ensure borrower_data is properly formatted
        if isinstance(borrower_data, dict):
            borrower_data = pd.DataFrame([borrower_data])

        # Predict optimal method
        optimal_method = self.recovery_model.predict(borrower_data)[0]
        method_probs = self.recovery_model.predict_proba(borrower_data)[0]

        # Predict default risk if early warning model exists
        default_risk = None
        if self.early_warning_model is not None:
            default_risk = self.early_warning_model.predict_proba(borrower_data)[0][1]

        return {
            "optimal_method": optimal_method,
            "method_probabilities": method_probs,
            "default_risk": default_risk,
        }

    def generate_strategy_report(self):
        """
        Generate comprehensive strategy report
        """
        print("LOAN RECOVERY STRATEGY REPORT")
        print("=" * 60)

        if self.processed_data is not None:
            print(f"Total Borrowers Analyzed: {len(self.processed_data)}")
            print(
                f"Risk Segments Identified: {self.processed_data['Risk_Segment'].nunique()}"
            )

            # Risk segment distribution
            segment_dist = (
                self.processed_data["Risk_Segment"].value_counts().sort_index()
            )
            print("\nRisk Segment Distribution:")
            for segment, count in segment_dist.items():
                percentage = (count / len(self.processed_data)) * 100
                print(f"  Segment {segment}: {count} borrowers ({percentage:.1f}%)")

        print("\nCollection Method Cost Analysis:")
        for method, params in self.cost_matrix.items():
            print(
                f"  {method}: ${params['cost']} cost, {params['avg_recovery_rate']:.1%} recovery rate"
            )

        print("\nRECOMMENDations:")
        print("1. Focus on high-ROI methods for each risk segment")
        print("2. Implement early warning system to prevent defaults")
        print("3. Personalize collection strategies based on borrower profiles")
        print("4. Monitor and adjust strategies based on performance metrics")


# Example usage function
def run_loan_recovery_analysis(df):
    """
    Run complete loan recovery analysis
    """
    # Initialize model
    model = LoanRecoveryStrategyModel()

    # Load and preprocess data
    processed_data = model.load_and_prepare_data(df)

    # Segment borrowers
    segments = model.segment_borrowers(n_clusters=4)

    # Train recovery optimization model
    recovery_model = model.train_recovery_optimization_model()

    # Develop early warning system
    warning_model = model.develop_early_warning_system()

    # Calculate ROI optimization
    roi_analysis = model.calculate_roi_optimization()

    # Generate final report
    model.generate_strategy_report()

    return model, roi_analysis


# Usage example:
# Load your DataFrame (df) with loan data
df = pd.read_csv("loan-recovery.csv")

model, roi_analysis = run_loan_recovery_analysis(df)
