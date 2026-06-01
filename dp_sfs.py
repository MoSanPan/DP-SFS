"""
DP-SFS: Differentially Private Spectral Feature Selection.

Reference
---------
[Your Paper Title], [Conference/Journal], [Year]

Usage
-----
    from dp_sfs import dp_spectral_feature_selection

    # X: (n, d) array, y: (n,) array, k: int, epsilon: float
    selected_idx = dp_spectral_feature_selection(X, y, k=10, epsilon=1.0, seed=42)
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans


def dp_spectral_feature_selection(X, y, k, epsilon, seed=42, M=None, delta=1e-5):
    """
    DP-SFS: Differentially Private Spectral Feature Selection.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Feature matrix (should be standardized).
    y : ndarray of shape (n,)
        Target vector (should be standardized).
    k : int
        Number of features to select.
    epsilon : float
        Privacy budget (smaller → more noise → stronger privacy).
    seed : int, default=42
        Random seed for reproducibility.
    M : float or None, default=None
        Clipping bound. If None, auto-detected from max(abs(X), abs(y)).
    delta : float, default=1e-5
        Privacy parameter delta (typically small, e.g., 1/n).

    Returns
    -------
    selected_idx : ndarray of shape (k,)
        Indices of the selected features.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import StandardScaler
    >>> X = StandardScaler().fit_transform(np.random.randn(100, 20))
    >>> y = X[:, 0] * 2 + X[:, 5] * (-1.5) + np.random.randn(100) * 0.5
    >>> y = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()
    >>> idx = dp_spectral_feature_selection(X, y, k=5, epsilon=1.0)
    >>> print(idx)
    """
    n, d = X.shape

    # ---- Step 1: Clipping ----
    if M is None:
        M = max(np.max(np.abs(X)), np.max(np.abs(y)))
    X_clip = np.clip(X, -M, M)
    y_clip = np.clip(y, -M, M)

    # ---- Step 2: Noisy covariance matrix ----
    Z = np.hstack([X_clip, y_clip.reshape(-1, 1)])
    p = d + 1
    M2 = (Z.T @ Z) / n
    idx_triu = np.triu_indices(p)
    vec = M2[idx_triu]
    m = len(vec)

    # Gaussian mechanism
    l2_sens = (2 * M**2 * np.sqrt(m)) / n
    sigma = l2_sens * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, sigma, m)
    vec_noisy = vec + noise

    M2_noisy = np.zeros((p, p))
    M2_noisy[idx_triu] = vec_noisy
    M2_noisy = M2_noisy + M2_noisy.T - np.diag(M2_noisy.diagonal())

    # ---- Step 3: Correlation matrix & Laplacian ----
    Sigma_noisy = M2_noisy[:-1, :-1]
    sigma_xy_noisy = M2_noisy[:-1, -1]

    # Fixed variance (1.0 after standardization)
    std_x = np.ones(d)
    std_y = 1.0
    R = Sigma_noisy / (np.outer(std_x, std_x) + 1e-8)
    R = np.clip(R, -1, 1)
    R = np.abs(R)
    np.fill_diagonal(R, 0)

    # Importance scores
    imp = np.abs(sigma_xy_noisy / (std_x * std_y + 1e-8))
    imp = np.clip(imp, 0, 1)

    # Normalized Laplacian
    degree = np.sum(R, axis=1)
    degree = np.maximum(degree, 1e-8)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    L = np.eye(d) - D_inv_sqrt @ R @ D_inv_sqrt
    L = (L + L.T) / 2

    # ---- Step 4: Eigendecomposition ----
    _, eigvecs = eigh(L)
    U = eigvecs[:, :k]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)

    # ---- Step 5: KMeans clustering + per-cluster selection ----
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(U)

    selected = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        best = idx[np.argmax(imp[idx])]
        selected.append(best)

    # Pad if fewer than k clusters produced
    while len(selected) < k:
        remaining = list(set(range(d)) - set(selected))
        if not remaining:
            break
        selected.append(rng.choice(remaining))

    return np.array(selected[:k])
