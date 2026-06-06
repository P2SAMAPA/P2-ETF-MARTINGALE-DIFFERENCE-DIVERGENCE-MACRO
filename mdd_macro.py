import numpy as np

def gaussian_kernel(x, y, sigma=1.0):
    """Gaussian kernel on macro features."""
    diff = x - y
    return np.exp(-np.dot(diff, diff) / (2 * sigma**2))

def mdd_statistic(returns, macro_features, sigma=1.0):
    """
    Compute Martingale Difference Divergence statistic:
    MDD = (1/(n-1)) * sum_{i≠j} K(macro_i, macro_j) * ret_i * ret_j
    where K is Gaussian kernel.
    This measures the squared conditional mean of returns given macro.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    # Standardise returns to have zero mean (optional)
    ret_centered = returns - np.mean(returns)
    total = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            kij = gaussian_kernel(macro_features[i], macro_features[j], sigma)
            total += kij * ret_centered[i] * ret_centered[j]
    # Normalise by (n-1) (unbiased)
    mdd = total / (n - 1)
    return float(max(mdd, 0.0))

def mdd_score(returns, macro_df, sigma=1.0):
    """
    Compute MDD for a single ETF using all macro variables.
    Macro features are standardised first.
    """
    if len(returns) != len(macro_df):
        min_len = min(len(returns), len(macro_df))
        returns = returns[:min_len]
        macro_df = macro_df.iloc[:min_len]
    if len(returns) < 2:
        return 0.0
    # Standardise macro features (each column separately)
    macro_scaled = (macro_df - macro_df.mean()) / (macro_df.std() + 1e-12)
    # Convert to numpy array
    macro_features = macro_scaled.values
    # Compute MDD using kernel on macro features
    mdd = mdd_statistic(returns, macro_features, sigma)
    return mdd
