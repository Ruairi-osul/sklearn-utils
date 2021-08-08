from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import numpy as np
import pandas as pd


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List) -> None:
        if not isinstance(cols, list):
            self.cols = [cols]
        else:
            self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        return X[self.cols]


class FctLumpTransformer(BaseEstimator, TransformerMixin):
    """Forcats's fct_lump in sklearn!
    credit to Tim Gibson on Kaggle for much of the implemention
    """

    def __init__(self, pct: float = 0.8, other_name: str = "other") -> None:
        """Collapse categories by relative occurance

        Args:
            pct (float, optional): Threshold for colapsing. Defaults to 0.8.
            other_name (str, optional): Name of category given to collapsed categories. Defaults to "other".
        """
        self.pct = pct
        self.other_name = other_name

    def fit(self, X):
        return self

    @staticmethod
    def _fct_lump(
        x: np.ndarray, pct: float = 0.8, other_name: str = "other"
    ) -> np.ndarray:
        categories, counts = np.unique(x, return_counts=True)
        n = np.sum(counts)
        descending_idx = np.flip(np.argsort(counts))
        cumulative_prop_covered = np.cumsum(counts[descending_idx]) / n
        index_covered = np.argmax(cumulative_prop_covered > pct)
        ctg_to_keep = categories[descending_idx][:index_covered]
        mapper = {ctg: ctg if ctg in ctg_to_keep else other_name for ctg in categories}
        return np.array([mapper[val] for val in x])

    def transform(self, X):
        return np.apply_over_axes(self._fct_lump, 0, X)

