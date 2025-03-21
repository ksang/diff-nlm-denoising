# diff-nlm-denoising
Differentiable Non-Local Mean Denoising Implemented in PyTorch

# Setup (uv)

- Install [uv](https://docs.astral.sh/uv/getting-started/) for python environment management, install python version compatible with current version of [PyTorch](https://pytorch.org/get-started/locally/).

- Create virtual environment and install dependencies:

```
git clone https://github.com/ksang/diff-nlm-denoising.git
cd diff-nlm-denosing
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

# References

- Buades, Antoni, Bartomeu Coll, and Jean-Michel Morel. "Non-local means denoising." Image Processing On Line 1 (2011): 208-212. [PDF](https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf)
- [A report on Non-Local Means Denoising](https://samrobbins.uk/essays/a-report-on-non-local-means-denoising)
- [OpenCV documentation](https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html)