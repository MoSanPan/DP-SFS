# DP-SFS: Differentially Private Spectral Feature Selection

DP-SFS is a differentially private feature selection algorithm based on spectral clustering. It injects Gaussian noise into the covariance matrix for privacy protection, applies spectral clustering to identify redundant feature structures, and selects the most representative feature from each cluster.

---

## Installation

```bash
pip install numpy scipy scikit-learn
```

## Usage

```python
from dp_sfs import dp_spectral_feature_selection

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | ndarray (n, d) | Feature matrix (should be standardized) |
| `y` | ndarray (n,)  | Target vector (should be standardized) |
| `k` | int | Number of features to select |
| `epsilon` | float | Privacy budget (smaller → stronger privacy) |
| `seed` | int | Random seed (default: 42) |
| `M` | float or None | Clipping bound. If None, auto-detected from max \|X\|, \|y\| |
| `delta` | float | Privacy parameter (default: 1e-5) |

**Returns**: `ndarray` of shape `(k,)` — indices of selected features.

## Algorithm

```
DP-SFS Algorithm
─────────────────
Input:  X (n×d), y (n), k, ε
Output: selected feature indices (k)

1. Clip:        X̂ = clip(X, -M, M),  ŷ = clip(y, -M, M)
2. Covariance:  M₂ = [X̂|ŷ]ᵀ [X̂|ŷ] / n  +  Gaussian noise calibrated to ε
3. Correlation: R = |corr(X̂)|  from noisy M₂
4. Laplacian:   L = I − D⁻¹ᐟ² R D⁻¹ᐟ²
5. Eigens:      U = top-k eigenvectors of L
6. Clustering:  labels = KMeans(U, k)
7. Selection:   for each cluster → argmax importance score
```

## References

**Joint**
@inproceedings{gillenwater2022joint,
  title={A Joint Exponential Mechanism For Differentially Private Top-$ k$},
  author={Gillenwater, Jennifer and Joseph, Matthew and Munoz, Andres and Diaz, Monica Ribero},
  booktitle={International Conference on Machine Learning},
  pages={7570--7582},
  year={2022},
  organization={PMLR}
}
**DPKendall**
@article{dick2023better,
  title={Better private linear regression through better private feature selection},
  author={Dick, Travis and Gillenwater, Jennifer and Joseph, Matthew},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={53457--53474},
  year={2023}
}
**FedSDG-FS**
@ARTICLE{li2024efficient,
  author={Li, Anran and Huang, Jiahui and Jia, Ju and Peng, Hongyi and Zhang, Lan and Tuan, Luu Anh and Yu, Han and Li, Xiang-Yang},
  journal={IEEE Transactions on Mobile Computing}, 
  title={Efficient and Privacy-Preserving Feature Importance-Based Vertical Federated Learning}, 
  year={2024},
  volume={23},
  number={6},
  pages={7238-7255}
  }
**Top-R**
@article{prastakos2026differentially,
  title={Differentially Private High-dimensional Variable Selection via Integer Programming},
  author={Prastakos, Petros and Behdin, Kayhan and Mazumder, Rahul},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={116544--116584},
  year={2026}
}
**Random**
@article{Guyon2003,
  title   = {An introduction to variable and feature selection},
  author  = {Guyon, Isabelle and Elisseeff, Andr{\'e}},
  journal = {Journal of Machine Learning Research},
  volume  = {3},
  number  = {Mar},
  pages   = {1157--1182},
  year    = {2003}
}

## File Structure

```
├── dp_sfs.py              # DP-SFS implementation
├── benchmark_methods.py   # Baseline methods (DPKendall, Joint, FedSDG-FS, Random, Top-R)
└── README.md
```


## License

MIT License.
