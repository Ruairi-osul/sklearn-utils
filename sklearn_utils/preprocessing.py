from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import numpy as np


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
        index_covered = int(np.argmax(cumulative_prop_covered > pct))
        ctg_to_keep = categories[descending_idx][:index_covered]
        mapper = {ctg: ctg if ctg in ctg_to_keep else other_name for ctg in categories}
        return np.array([mapper[val] for val in x])

    def transform(self, X):
        return np.apply_over_axes(self._fct_lump, 0, X)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select columns
    """

    def __init__(self, selected_columns):
        """Select Column

        Args:
            selected_columns (list): List of column names to include
        """
        self.selected_columns = selected_columns

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.selected_columns]


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, to_drop):
        """Drop Columns

        Args:
            to_drop (list): List of columns to drop
        """
        self.to_drop = to_drop

    def fit(self, X):
        return self

    def transform(self, X):
        return X[[c for c in X.columns if c not in self.to_drop]]


class DTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dtype_to_set):
        """Change the dtype of input

        Args:
            dtype_to_set (type): Type to which values will be transformed. Must work with np.astype 
        """
        self.dtype_to_set = dtype_to_set

    def fit(self, X):
        return self

    def transform(self, X):
        return X.astype(self.dtype_to_set)
