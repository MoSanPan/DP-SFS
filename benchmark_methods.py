"""
Baseline methods for differentially private feature selection.

Methods
-------
- random_selection        : Random baseline
- dpkendall_selection     : DP Kendall correlation-based selection
- joint_topk              : Joint scoring with Gumbel noise
- fedsdg_selection        : FedSDG-FS (gradient-based iterative selection)
"""

import numpy as np
from scipy import stats


# =====================================================================
# Random
# =====================================================================
def random_selection(d, k, seed=42):
    """Randomly select k features from d candidates."""
    rng = np.random.RandomState(seed)
    return rng.choice(d, size=k, replace=False)


# =====================================================================
# Joint
# =====================================================================
def _kendall_score(feature, labels, n):
    eps = 1e-12
    feature = feature + np.random.normal(0, eps, len(feature))
    labels = labels + np.random.normal(0, eps, len(labels))
    tau, _ = stats.kendalltau(feature, labels)
    if np.isnan(tau):
        tau = 0.0
    return np.abs(tau) * (n / 2)


def joint_topk(X, y, k, epsilon):
    """
    Joint scoring with Gumbel noise perturbation.

    Computes base scores from Kendall correlation, adds a redundancy
    penalty from inter-feature correlation, and selects via the
    exponential mechanism (Gumbel noise).
    """
    n, d = X.shape
    base_score = np.array([_kendall_score(X[:, j], y, n) for j in range(d)])
    selected = []
    corr_matrix = np.abs(np.corrcoef(X, rowvar=False))

    for t in range(k):
        if len(selected) == 0:
            scores = base_score.copy()
        else:
            penalty = np.sum(corr_matrix[selected, :], axis=0) / len(selected)
            scores = base_score - penalty

        scores[selected] = -np.inf
        beta = 2 / (epsilon / k)
        noise = np.random.gumbel(0, beta, size=d)
        chosen = int(np.argmax(scores + noise))
        selected.append(chosen)

    return np.array(selected, dtype=np.int64)


# =====================================================================
# DPKendall
# =====================================================================
def _kendall_tau_abs(feature, labels, n):
    eps_noise = 1e-12
    feature = feature + np.random.normal(0, eps_noise, len(feature))
    labels = labels + np.random.normal(0, eps_noise, len(labels))
    tau, _ = stats.kendalltau(feature, labels, method='asymptotic')
    if np.isnan(tau):
        tau = 0.0
    return np.abs((n / 2) * tau)


def dpkendall_selection(X, y, k, epsilon):
    """
    DP-Kendall feature selection.

    Iteratively selects features by computing Kendall's tau correlation
    with the target, applying a redundancy penalty, and perturbing
    scores with Gumbel noise for privacy.
    """
    n, d = X.shape
    label_corr = np.array([_kendall_tau_abs(X[:, j], y, n) for j in range(d)])
    selected = []
    selected_corr = np.zeros((k, d))
    sensitivity = 1.5
    eps_per_round = epsilon / k

    for t in range(k):
        if t == 0:
            scores = label_corr.copy()
        else:
            penalty = np.sum(selected_corr[:t, :], axis=0) / t
            scores = label_corr - penalty
            sensitivity = 3.0

        scores[selected] = -np.inf
        beta = 2 * sensitivity / eps_per_round
        gumbel_noise = np.random.gumbel(0, beta, size=d)
        chosen = np.argmax(scores + gumbel_noise)
        selected.append(chosen)

        new_corr = np.array([_kendall_tau_abs(X[:, chosen], X[:, j], n) for j in range(d)])
        selected_corr[t, :] = new_corr

    return np.array(selected)


# =====================================================================
# FedSDG-FS
# =====================================================================
def _gini_impurity_init(X, y, n_bins=10):
    """Gini impurity initialization for feature importance."""
    n, d = X.shape
    classes = np.unique(y)
    gini_scores = np.zeros(d)

    for j in range(d):
        bins = np.linspace(X[:, j].min(), X[:, j].max(), n_bins + 1)
        digitized = np.digitize(X[:, j], bins[1:-1])
        weighted_gini = 0
        for i in range(n_bins):
            mask = digitized == i
            if mask.sum() == 0:
                continue
            p = np.array([(y[mask] == c).sum() / mask.sum() for c in classes])
            weighted_gini += (mask.sum() / n) * (1 - np.sum(p**2))
        gini_scores[j] = weighted_gini

    return gini_scores


def fedsdg_selection(X, y, k, epsilon, n_iters=100, seed=42):
    """
    FedSDG-FS: Federated Stochastic Dual-Gating Feature Selection.

    Iteratively learns feature importance via differentially private
    gradient descent with random gating. Selects the top-k features
    after convergence.
    """
    from scipy.stats import norm

    rng = np.random.RandomState(seed)
    n, d = X.shape

    # Gini initialization
    gini_scores = _gini_impurity_init(X, y)
    mu = 1.0 - gini_scores / (gini_scores.max() + 1e-8)
    mu = np.clip(mu, 0.1, 0.9)

    sigma_gate = 0.3
    sigma_dp = np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon
    lr = 0.05
    lambda_reg = 0.05

    for _ in range(n_iters):
        # Random gating
        rho = rng.normal(0, sigma_gate, d)
        s = np.clip(mu + rho, 0, 1)

        # Regularization gradient
        phi = norm.pdf(mu / sigma_gate) / sigma_gate
        Phi_val = norm.cdf(mu / sigma_gate) + 1e-8
        grad_reg = lambda_reg * phi / Phi_val

        # Feature importance gradient
        X_sel = X * s.reshape(-1, d)
        w = np.linalg.lstsq(X_sel, y, rcond=None)[0]
        residual = y - X_sel @ w
        grad_importance = np.abs(X_sel.T @ residual) / n
        grad_importance = grad_importance / (grad_importance.max() + 1e-8)

        # DP noisy gradient + update
        grad = grad_importance - grad_reg + rng.normal(0, sigma_dp, d)
        mu = mu + lr * grad
        mu = np.clip(mu, 0.01, 0.99)
        lr *= 0.995

    return np.argsort(mu)[::-1][:k]


# =====================================================================
# Top-R
# =====================================================================
try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False


if _HAS_GUROBI:
    import math
    import mpmath
    from itertools import combinations
    mpmath.mp.dps = 100

    def _power_method(A, max_iter=1000, tol=1e-6):
        n = A.shape[0]
        x = np.random.rand(n)
        x /= np.linalg.norm(x)
        for _ in range(max_iter):
            x_new = np.dot(A, x)
            ev = np.dot(x_new, x)
            x_new /= np.linalg.norm(x_new)
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        return ev

    def _project_l0(x, s):
        idx = np.argsort(np.abs(x))[::-1][:s]
        p = np.zeros_like(x)
        p[idx] = x[idx]
        return p

    def _project_l2(x, r):
        norm = np.linalg.norm(x)
        return x if norm == 0 else (r / max(r, norm)) * x

    def _ls_pgd(X, y, r=1.1, Lambda=0.1, max_iter=1000):
        n, p = X.shape
        A = X.T @ X
        b = X.T @ y
        d_y = y.T @ y
        eig = _power_method(A + Lambda * np.eye(p))
        step = n / eig
        beta = np.zeros(p)
        obj = np.zeros(max_iter)
        for i in range(max_iter):
            grad = (A @ beta - b) / n + Lambda * beta / n
            beta = _project_l2(beta - step * grad, r)
            obj[i] = (beta @ A @ beta - 2 * beta @ b + d_y) / (2 * n) + Lambda * (beta @ beta) / (2 * n)
            if i > 0 and abs(obj[i] - obj[i-1]) / obj[i] < 1e-8:
                break
        return obj[min(i, max_iter - 1)]

    def _iht(s, X, A, b, d_y, eig, zt, Lambda, max_iter=1000):
        n = X.shape[0]
        step = n / eig
        beta = _project_l0(zt, s)
        obj_new = 1000
        for i in range(max_iter):
            grad = (A @ beta - b) / n + Lambda * beta / n
            beta_new = _project_l0(beta - step * grad, s)
            obj = obj_new
            obj_new = (beta_new @ A @ beta_new - 2 * beta_new @ b + d_y) / (2 * n) + Lambda * np.sum(beta_new ** 2) / (2 * n)
            if abs(obj - obj_new) / obj < 1e-3:
                break
            beta = beta_new
        return beta, np.where(beta != 0, 1, beta)

    def _compute_c_grad(X, A, b, d_y, z, r, beta0, Lambda, max_iter=1000):
        n = X.shape[0]
        z_safe = np.where(z == 0, 1e-9, z)
        eig = _power_method(A + Lambda * np.diag(1 / z_safe))
        step = n / eig
        beta = _project_l2(beta0, r)
        obj = np.zeros(max_iter)
        for i in range(max_iter):
            grad = (A @ beta - b) / n + Lambda * (beta / z_safe) / n
            beta = _project_l2(beta - step * grad, r)
            obj[i] = (beta @ A @ beta - 2 * beta @ b + d_y) / (2 * n) + Lambda * np.sum(beta ** 2 / z_safe) / (2 * n)
            if i > 0 and abs(obj[i] - obj[i-1]) / obj[i] < 1e-8:
                break
        c = obj[min(i, max_iter - 1)]
        return c, -Lambda * np.square(beta) / np.square(z_safe) / (2 * n)

    def _add_noise(zt, lb=1e-3, ub=5e-3):
        h = zt.copy()
        zi = np.where(zt == 0)[0]
        h[zi] += np.random.uniform(lb, ub, len(zi))
        return h

    def _outer_approx(y, X, s, r=1.1, Lambda=0.1, tol=5e-3, iter_pgd=1000, max_iter=1000, per1mist=0.1):
        n, p = X.shape
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("mip1", env=env)
        z = model.addMVar(p, vtype=GRB.BINARY, name="z")
        ita = model.addVar(lb=0.0)
        model.setObjective(ita, GRB.MINIMIZE)

        zt = np.zeros(p)
        zt[(p - s):p] = 1
        A = X.T @ X
        b = X.T @ y
        d_y = y.T @ y
        eig = _power_method(A + Lambda * np.eye(p))
        beta, zt = _iht(s, X, A, b, d_y, eig, zt, Lambda, max_iter)
        zt[np.argsort(np.abs(zt))[:p - s]] = 0

        oi = np.nonzero(zt)[0]
        zi = np.where(zt == 0)[0]
        corr_idx = np.argsort(np.abs(b))[::-1]
        top_corr = corr_idx[:max(1, int(per1mist * len(b)))]
        iz = zi[np.isin(zi, top_corr)]
        io_ = oi[np.isin(oi, top_corr)]

        for kk in range(1, s + 1):
            for oii in combinations(io_, kk):
                for zii in combinations(iz, kk):
                    mv = np.copy(zt)
                    mv[list(oii)] = 0
                    mv[list(zii)] = 1
                    mn = _add_noise(mv)
                    c, gc = _compute_c_grad(X, A, b, d_y, mn, r, mv, Lambda, iter_pgd)
                    model.addConstr(c + gc @ (z - mn) <= ita)
            if kk == 1:
                break

        zt_h = _add_noise(zt)
        c, gc = _compute_c_grad(X, A, b, d_y, zt_h, r, beta, Lambda, iter_pgd)
        model.addConstr(c + gc @ (z - zt_h) <= ita)
        model.addConstr(gp.quicksum(z[i] for i in range(p)) <= s)

        oi = np.nonzero(zt)[0]
        Xs = X[:, oi]
        curr_obj = _ls_pgd(Xs, y, r, Lambda)
        ita_t, t = 0, 0
        while np.abs(ita_t - curr_obj) / curr_obj > 1e-3:
            model.optimize()
            if model.status != GRB.OPTIMAL:
                break
            ita_t = model.objVal
            t += 1
            if t > 50:
                break
            zt = np.array(model.getAttr('x'))[:p]
            zt[np.argsort(np.abs(zt))[:p - s]] = 0
            zt_h = _add_noise(zt)
            c, gc = _compute_c_grad(X, A, b, d_y, zt_h, r, zt, Lambda, iter_pgd)
            model.addConstr(c + gc @ (z - zt_h) <= ita)
            oi = np.nonzero(zt)[0]
            Xs = X[:, oi]
            curr_obj = _ls_pgd(Xs, y, r, Lambda)

        zt[np.argsort(np.abs(zt))[:p - s]] = 0
        oi = np.nonzero(zt)[0]
        zi = np.where(zt == 0)[0]
        betas = np.zeros((p, max(2, (p - s) * s + 2)))
        betas[:, 0] = zt
        obj_vals = np.zeros(max(2, (p - s) * s + 2))
        Xs = X[:, oi]
        obj_vals[0] = _ls_pgd(Xs, y, r, Lambda)
        kk = 1
        for o in oi:
            for zz in zi:
                if kk >= len(obj_vals):
                    break
                mv = np.copy(zt)
                mv[o] = 0
                mv[zz] = 1
                betas[:, kk] = mv
                nzi = np.nonzero(mv)[0]
                Xs = X[:, nzi]
                obj_vals[kk] = _ls_pgd(Xs, y, r, Lambda)
                kk += 1

        model.addConstr(gp.quicksum(z[i] for i in oi) <= s - 1.5)
        ita_t, t = 0, 0
        while np.abs(ita_t - curr_obj) / curr_obj > tol:
            model.optimize()
            if model.status != GRB.OPTIMAL:
                break
            ita_t = model.objVal
            t += 1
            if t > 50:
                break
            zt = np.array(model.getAttr('x'))[:p]
            zt[np.argsort(np.abs(zt))[:p - s]] = 0
            zt_h = _add_noise(zt)
            c, gc = _compute_c_grad(X, A, b, d_y, zt_h, r, zt, Lambda, iter_pgd)
            model.addConstr(c + gc @ (z - zt_h) <= ita)
            oi = np.nonzero(zt)[0]
            Xs = X[:, oi]
            curr_obj = _ls_pgd(Xs, y, r, Lambda)

        gap = model.getAttr(GRB.Attr.MIPGap) if model.status == GRB.OPTIMAL else 0
        si = np.argsort(obj_vals)
        return betas[:, si], gap, obj_vals[si]

    def _clip_select(X, y, b_x, b_y):
        return np.clip(X, -b_x, b_x), np.clip(y, -b_y, b_y)

    def _delta_sens(n, s, r, b_x, b_y):
        return (b_y**2 + 2 * b_y * b_x * r * np.sqrt(s) + b_x**2 * r**2 * s) / (2.0 * n)

    def _draw_prob(p, u):
        return int(np.searchsorted(np.cumsum(p), u, side='right'))

    def get_topr_supports(X, y, s, eps, b_clip=3.0, runs=50):
        """Precompute Top-R support sets via Gurobi MIP + exponential mechanism."""
        n, p = X.shape
        Xc, yc = _clip_select(X, y, b_clip, b_clip)
        try:
            betas_sorted, gap, obj_values = _outer_approx(yc, Xc, s)
            if gap == -1:
                return None
            reduced = np.where(betas_sorted != 0, 1, 0)
            Re = (p - s) * s + 2
            delta = _delta_sens(n, s, 1.1, b_clip, b_clip)
            exponents = np.zeros(Re + 1)
            n_obj = min(Re, len(obj_values))
            exponents[:n_obj] = -eps * obj_values[:n_obj] / (2 * delta)
            exp_terms = np.array([mpmath.exp(x) for x in exponents], dtype=object)
            exp_terms[Re] = (math.comb(p, s) - Re) * exp_terms[Re]
            denom = mpmath.fsum(exp_terms)
            prob = np.array(exp_terms, dtype=float) / float(denom)
            topk = np.zeros((p, runs), dtype=int)
            for i in range(runs):
                idx_top = _draw_prob(prob, np.random.uniform(0, 1))
                if idx_top < reduced.shape[1]:
                    topk[:, i] = reduced[:, idx_top]
                else:
                    shuf = np.arange(p)
                    np.random.shuffle(shuf)
                    cand = np.zeros(p, dtype=int)
                    cand[shuf[:s]] = 1
                    topk[:, i] = cand
            return topk
        except Exception:
            return None


def topr_selection(X, y, k, epsilon, supports=None, b_clip=3.0, seed=42):
    """
    Top-R: Exponential-family sampling over near-optimal feature subsets.

    Requires Gurobi for the offline MIP precomputation. If supports are
    precomputed via ``get_topr_supports``, the online query is instant.

    Parameters
    ----------
    X : ndarray (n, d)
    y : ndarray (n,)
    k : int
    epsilon : float
    supports : ndarray or None
        Precomputed support sets from ``get_topr_supports``. If None,
        attempts to compute on-the-fly (requires Gurobi).
    b_clip : float
    seed : int

    Returns
    -------
    selected : ndarray (k,)
    """
    if not _HAS_GUROBI:
        raise ImportError("Top-R requires Gurobi. Please install gurobipy.")

    if supports is None:
        supports = get_topr_supports(X, y, k, epsilon, b_clip=b_clip, runs=1)

    if supports is None:
        # Fallback to random
        rng = np.random.RandomState(seed)
        return rng.choice(X.shape[1], size=k, replace=False)

    rng = np.random.RandomState(seed)
    col = rng.randint(supports.shape[1])
    idx = np.where(supports[:, col] == 1)[0]
    if len(idx) == 0:
        return rng.choice(X.shape[1], size=k, replace=False)
    return idx[:k]
