# ============================================================
# swiftmltoolz.py
# Custom Machine Learning Utilities & Models
# ============================================================


"""
swiftmltoolz
------------

A lightweight machine learning toolkit containing:

Models:
    - LogisticRegressionGD (Mini-batch Gradient Descent)

Preprocessing:
    - Z_Score_Normalizer

Evaluation:
    - plot_roc_comparison
    - plot_importance
    - plot_decision_boundary
    - plot_lin
    - plot_corr_heatmap

Feature Engineering:
    - get_logreg_importance
    - select_important_features
"""
__version__ = "0.1.0"
__author__ = "Swift"

# ============================================================
# Imports
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import auc
from sklearn.utils import check_X_y, check_array, shuffle
from sklearn.utils.validation import check_is_fitted

# ============================================================
# Models
# ============================================================


class LogisticRegressionGD(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression implemented from scratch using
    Mini-batch Gradient Descent with L2 Regularization.

    Parameters
    ----------
    alpha : float
        Learning rate
    iters : int
        Number of training iterations
    lambda_ : float
        L2 regularization strength
    batch_size : int
        Mini-batch size
    tol : float
        Early stopping tolerance
    plot_cost : bool
        Whether to show convergence plot
    random_state : int or None
        Random seed for reproducibility
    """

    def __init__(self, alpha=0.01, iters=1000, lambda_=0.1, batch_size=32, tol=1e-4, plot_cost=False, random_state=None):
        self.alpha = alpha
        self.iters = iters
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.tol = tol
        self.plot_cost = plot_cost
        self.models_ = {}
        self.random_state = random_state
        self.cost_histories_ = {}  # To store history for plotting

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        # optonal
        self.models_ = {}
        self.cost_histories_ = {}
        # -------------
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        for cls in self.classes_:
            y_binary = (y == cls).astype(int)
            # Capture both weights/bias AND the cost history
            w, b, history = self._train_mini_batch(X, y_binary)
            self.models_[cls] = (w, b)
            self.cost_histories_[cls] = history

        if self.plot_cost:
            self.show_cost_plot()
        return self

    def _train_mini_batch(self, X, y):
        m, n = X.shape
        w, b = np.zeros(n), 0.0
        history = []
        prev_loss = float('inf')
        rng = np.random.RandomState(self.random_state)
        for i in range(self.iters):
            X_shuff, y_shuff = shuffle(X, y, random_state=rng)

            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                xi, yi = X_shuff[start:end], y_shuff[start:end]

                y_cap = self._sigmoid(np.dot(xi, w) + b)
                error = y_cap - yi

                dw = (np.dot(xi.T, error) / len(xi)) + (self.lambda_ / m) * w
                db = np.mean(error)

                w -= self.alpha * dw
                b -= self.alpha * db

            # Record cost every 10 epochs for the history plot
            if i % 10 == 0:
                current_loss = self._compute_cost(X, y, w, b)
                history.append(current_loss)

                # Early Stopping Check
                if abs(prev_loss - current_loss) < self.tol:
                    break
                prev_loss = current_loss

        return w, b, history

    def _compute_cost(self, X, y, w, b):
        z = np.dot(X, w) + b
        y_cap = np.clip(self._sigmoid(z), 1e-15, 1 - 1e-15)
        # Add regularization to the reported cost
        reg_cost = (self.lambda_ / (2 * X.shape[0])) * np.sum(np.square(w))
        return -np.mean(y * np.log(y_cap) + (1 - y) * np.log(1 - y_cap)) + reg_cost

    def predict_proba(self, X):

        check_is_fitted(self, attributes=["models_"])
        X = check_array(X)

        probs = []
        for cls in self.classes_:
            w, b = self.models_[cls]
            probs.append(self._sigmoid(np.dot(X, w) + b))

        probs = np.column_stack(probs)

        # probs = np.column_stack([self._sigmoid(np.dot(X, w) + b) for cls, (w, b) in self.models_.items()])

        if len(self.classes_) == 2:
            return np.column_stack([1 - probs[:, 1], probs[:, 1]])

        # Softmax-like normalization for multiclass
        return probs / np.sum(probs, axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """
        # Ensure input is a proper array and fitted
        check_is_fitted(self, attributes=["models_"])
        X, y = check_X_y(X, y)

        # Get predictions
        y_pred = self.predict(X)

        # Calculate mean accuracy: (Total Correct / Total Samples)
        return np.mean(y_pred == y)

    def show_cost_plot(self):
        plt.style.use('seaborn-v0_8-muted')
        fig, ax = plt.subplots(figsize=(10, 6))
        n_classes = len(self.classes_)

        for cls, hist in self.cost_histories_.items():
            if n_classes <= 2:
                # Plot only the positive class for binary to avoid overlapping lines
                label = "Binary Logistic Regression" if cls == self.classes_[
                    1] else None
                if label is None:
                    continue
            else:
                label = f'Class {cls} vs Rest'

            ax.plot(
                # Multiplied by 10 because we record every 10 epochs
                np.arange(len(hist)) * 10,
                hist,
                label=label,
                linewidth=2,
                marker='o',
                markersize=4,
                markevery=max(1, len(hist)//10)
            )

        ax.set_xlabel('Epochs (Iterations)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cost (Log Loss)', fontsize=12, fontweight='bold')
        title = 'Convergence: Binary' if n_classes <= 2 else 'Convergence: One-vs-Rest'
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        plt.tight_layout()
        plt.show()


# ============================================================
# Preprocessing
# ============================================================

class Z_Score_Normalizer(BaseEstimator, TransformerMixin):
    """
    Standardizes features by removing the mean
    and scaling to unit variance (Z-score normalization).
    """

    def __init__(self):
        """Standardizes features by removing the mean and scaling to unit variance."""
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        # Handle Pandas or NumPy
        X = check_array(X, accept_sparse=False)

        # Calculate mean and std for each column (axis=0)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Prevent division by zero if a feature has zero variance
        self.std_[self.std_ == 0] = 1.0

        return self

    def transform(self, X):
        # Ensure fit was called before transform
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        # Apply the transformation: (X - mean) / std
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X):
        """Convert standardized data back to original scale."""
        check_is_fitted(self)
        return (X * self.std_) + self.mean_

# ============================================================
# Feature Importance Utilities
# ============================================================


def get_logreg_importance(model):
    """
    Extract mean absolute coefficient importance
    from LogisticRegressionGD model.
    """
    # model.models_.values() contains tuples of (w, b)
    # We extract ONLY the 'w' (index 0) from each tuple
    all_weights = np.array([params[0] for params in model.models_.values()])

    # Calculate Mean Absolute Importance across all classes
    avg_importance = np.mean(np.abs(all_weights), axis=0)

    return avg_importance

# ============================================================
# Visualization Utilities
# ============================================================


def plot_roc_comparison(model_results):
    """
    Plots ROC curves for multiple models for comparison.
    parameters: dictionary where keys are model names and values are tuples of (fpr, tpr) arrays.
    usage: model_results = {
    "Model_1": (fpr, tpr),
    "Model_2": (fpr, tpr)
    }
    """
    # Professional Styling
    plt.style.use('seaborn-v0_8-muted')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the Diagonal "Chance" Line (50/50 guess)
    ax.plot([0, 1], [0, 1], color='navy', lw=2,
            linestyle='--', label='Random Guess (AUC = 0.50)')

    # Loop through each model in the dictionary
    for model_name, (fpr, tpr) in model_results.items():
        roc_auc = auc(fpr, tpr)  # Calculate the Area Under Curve

        ax.plot(
            fpr,
            tpr,
            lw=2.5,
            label=f'{model_name} (AUC = {roc_auc:.3f})'
        )

    # Professional Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Comparison',
                 fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc="lower right", frameon=True,
              facecolor='white', edgecolor='gray')

    plt.tight_layout()
    plt.show()


def plot_importance(importances, X, column_names=None, title="Feature Importances", top_n=20):
    # 1. Logic for Feature Names - Now includes a safety check
    n_features = len(importances)

    if column_names is not None and len(column_names) == n_features:
        feature_names = np.array(column_names)
    elif hasattr(X, 'columns') and len(X.columns) == n_features:
        feature_names = np.array(X.columns)
    else:
        # Fallback: Generate generic names if there's a mismatch
        print(
            f"Warning: Column name mismatch. Expected {n_features}, got {len(column_names) if column_names is not None else 0}. Using indices.")
        feature_names = np.array([f"Feature {i}" for i in range(n_features)])

    # 2. Sort and Slice to Top N
    top_n = min(top_n, n_features)
    indices = np.argsort(importances)[-top_n:]
    sorted_importances = importances[indices]
    sorted_names = feature_names[indices]

    # 3. Dynamic Figure Size
    fig, ax = plt.subplots(figsize=(12, 0.45 * top_n))
    # Very light grey background for contrast
    fig.patch.set_facecolor('#fdfdfd')

    # 4. Enhanced Color Mapping
    # Using 'magma' or 'viridis' for a high-end look
    norm = plt.Normalize(vmin=sorted_importances.min(),
                         vmax=sorted_importances.max())
    colors = cm.plasma(norm(sorted_importances))

    # 5. Create Horizontal Bars
    bars = ax.barh(range(top_n), sorted_importances, color=colors,
                   edgecolor='white', linewidth=1, alpha=0.85)

    # 6. Refined Typography & Styling
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(sorted_names, fontsize=11,
                       color='#2C3E50', fontweight='500')

    # Modern value labels
    ax.bar_label(bars, padding=8, fmt='%.3f', fontsize=10,
                 fontweight='bold', color='#34495E')

    # 7. Professional Finishing Touches
    ax.set_title(title.upper(), fontsize=18, fontweight='bold',
                 color='#2C3E50', loc='left', pad=25)

    # Clean up the chart area
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.tick_params(axis='y', which='both', length=0)  # Remove tick marks

    plt.tight_layout()
    plt.show()


def plot_decision_boundary(model, X, y, feature_names):
    """
    Plots the decision boundary for exactly 2 features.

    Parameters:
    model: Your fitted LogisticRegressionGD model.
    X: NumPy array or DataFrame (must have exactly 2 columns).
    y: The true labels.
    feature_names: List of strings [feature1, feature2].
    """
    if X.shape[1] != 2:
        raise ValueError(
            "Decision boundary can only be plotted for 2 features at a time.")

    # 1. Setup the mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 2. Predict across the entire grid
    # We flatten xx and yy to feed them into the model
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # 3. Plotting
    plt.figure(figsize=(10, 7))
    plt.style.use('seaborn-v0_8-muted')

    # Draw the colored regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

    # Draw the actual data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y,
                          edgecolors='k', cmap='coolwarm', s=40)

    # Formatting
    plt.xlabel(feature_names[0], fontsize=12, fontweight='bold')
    plt.ylabel(feature_names[1], fontsize=12, fontweight='bold')
    plt.title(f'Logistic Regression Decision Boundary\n{feature_names[0]} vs {feature_names[1]}',
              fontsize=14, fontweight='bold')

    plt.legend(*scatter.legend_elements(), title="Classes", loc="upper right")
    plt.grid(alpha=0.3)
    plt.show()


def plot_corr_heatmap(data, cmap="coolwarm", title="Feature Correlation Matrix"):
    """
    Computes correlation and plots a high-resolution professional heatmap.

    Recommended CMAPs:
    -----------------
    'RdBu_r'   : Professional standard (Red=Pos, Blue=Neg)
    'coolwarm' : Soft and modern gradient
    'vlag'     : Muted/Academic look
    'magma'    : High-contrast/Dark mode feel
    'mako'     : Sleek teal/ocean tones
    """
    # 1. Compute Correlation
    corr = data.corr(numeric_only=True)

    # 2. Figure Setup
    plt.figure(figsize=(18, 11), facecolor='#FDFDFD')
    sns.set_theme(style="white", font_scale=1.0)

    # 3. Create Heatmap
    # center=0 is crucial: it ensures 0 correlation is the neutral color
    ax = sns.heatmap(
        data=corr,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        center=0,             # Zero is always neutral (white/grey)
        square=True,          # Keeps cells as perfect squares
        linewidths=0.5,      # Subtle separation lines
        linecolor='#EEEEEE',  # Soft grey lines
        cbar_kws={"shrink": .8, "label": "Correlation Strength"},
        annot_kws={"size": 9, "weight": "bold", "color": "#333333"}
    )

    # 4. Professional Formatting
    plt.title(title.upper(), fontsize=20, fontweight='bold',
              pad=30, loc='left', color='#2C3E50')

    # Clean up axis labels
    plt.xticks(rotation=45, ha='right', fontsize=10, color='#34495E')
    plt.yticks(rotation=0, fontsize=10, color='#34495E')

    plt.tight_layout()
    plt.show()


def plot_lin(data, column_names, y):
    """
    plots all the  features x  against y
    usage: plot_lin(data,column_names,y)
    y = target variable name as string
    """
    for column in column_names:
        plt.scatter(data=data, x=data[column], y=y)
        plt.xlabel(f"{column}")
        plt.ylabel(f"{y}")
        plt.show()


def plot_model_pred_corr(models, X, y, cv=5):
    '''
    Computes cross_val_predicts for all models and plots prediction correlation.
    Usage: Aim for a "sweet spot" of correlation [0.85 <---> 0.93] for ensembling.
    models: list of tuples [(model_name, model_instance), ...]
    X: feature data (DataFrame or NumPy array)
    y: target variable
    cv: number of cross-validation folds (default=5)
    '''
    predictions = {}

    for name, model in models:

        predictions[name] = cross_val_predict(model, X=X, y=y, cv=cv)

    final_df = pd.DataFrame(predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(final_df.corr(), annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.title('Prediction Correlation Between Models')
    plt.show()

# ============================================================
# Feature Selection
# ============================================================


def select_important_features(X, importances, threshold=None, top_n=None):
    """Select features by threshold or top_n.
    returns the filtered DataFrame and the list of selected feature names.
    usage: X_selected, selected_names = select_important_features(X, importances, threshold=0.05)
    or: X_selected, selected_names = select_important_features(X, importances, top_n=10)
    """
    # 1. Get feature names
    if hasattr(X, 'columns'):
        feature_names = np.array(X.columns)
    else:
        feature_names = np.array(
            [f"Feature {i}" for i in range(len(importances))])

    # 2. Logic for Top N selection (Prioritize this if top_n is provided)
    if top_n is not None:
        indices = np.argsort(importances)[-top_n:]
        selected_names = feature_names[indices]

    # 3. Logic for Threshold selection
    else:
        # If no threshold is provided by the user, calculate the mean automatically
        curr_threshold = threshold if threshold is not None else np.mean(
            importances)

        selected_names = feature_names[importances >= curr_threshold]

        if threshold is None:
            print(
                f"Using auto-calculated mean threshold: {curr_threshold:.4f}")
        else:
            print(f"Using user-defined threshold: {curr_threshold:.4f}")

    # 4. Filter the DataFrame
    # 4. Filter the data
    if hasattr(X, 'loc'):  # Pandas
        X_selected = X[selected_names]
    else:  # NumPy
        selected_indices = np.where(np.isin(feature_names, selected_names))[0]
        X_selected = X[:, selected_indices]

    print(
        f"Selected {len(selected_names)} features out of {len(importances)}.")
    return X_selected, list(selected_names)


__all__ = [
    "LogisticRegressionGD",
    "Z_Score_Normalizer",
    "get_logreg_importance",
    "plot_roc_comparison",
    "plot_importance",
    "select_important_features",
    "plot_decision_boundary",
    "plot_lin",
    "plot_corr_heatmap",
    "plot_model_pred_corr"
]
