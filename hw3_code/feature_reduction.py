import pandas as pd
import statsmodels.api as sm
from typing import List


class FeatureReduction(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            forward_list: (python list) contains significant features. Each feature
            name is a string
        """

        forward_list = []
        while True:
            remaining_features = list(set(data.columns) - set(forward_list))
            new_pval = pd.Series(
                index=remaining_features,
                data=[sm.OLS(target, sm.add_constant(data[forward_list + [f]]).fit().pvalues[f]) for f in remaining_features],
            )
            min_pval = new_pval.min()
            if min_pval < significance_level:
                forward_list.append(new_pval.idxmin())
            else:
                break
        return forward_list

    @staticmethod
    def backward_elimination(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            backward_list: (python list) contains significant features. Each feature
            name is a string
        """
        backward_list = list(data.columns)
        while True:
            pvals = sm.OLS(target, sm.add_constant(data[backward_list]).fit().pvalues[1:])
            max_pval = pvals.max()
            if max_pval > significance_level:
                backward_list.remove(pvals.idxmax())
            else:
                break
        return backward_list
        # raise NotImplementedError
