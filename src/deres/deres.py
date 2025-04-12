import logging
import math
from collections.abc import Sequence
from itertools import zip_longest
from typing import Literal

import adjustText
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.pyplot import Figure
from matplotlib.ticker import MaxNLocator

logger = logging.getLogger("deres")


class DEResult:
    """
    Container to hold a differential expression result and associated metadata.

    Parameters
    ----------
    res
        The data frame with the statistical result. Typically contains a column with p-values
        and a column with some sort of effect size (e.g. fold change). The data frame may contain any additional
        columns
    adata
        associated AnnData object that holds expression values that were used to obtain the statistical results.
        This is optional, and only required for some plot types.
    layer
        layer of AnnData to use (if any). If `None`, use `X`
    p_col
        Column in `res` containing the p-value
    effect_size_col
        Column in `res` containing the effect size (e.g. log fold change)
    contrast_col
        Column in `res` containing the contrast name. Only applicable if results
        from multiple comparisons are stored in the data frame. If it contains only the results
        from a single comparison, just leave this as `None`.
    var_col
        Column in `res` containing the variable name (e.g. gene symbol). If `None`, use the index.
    """

    def __init__(
        self,
        res: pd.DataFrame,
        adata: AnnData | None = None,
        *,
        layer: str | None = None,
        p_col: str = "p_value",
        effect_size_col: str = "log_fc",
        contrast_col: str | None = None,
        var_col: str | None = None,
    ) -> None:
        self._res = res
        self.layer = layer
        self.adata = adata

        self.p_col = p_col
        self.effect_size_col = effect_size_col
        self.contrast_col = contrast_col
        self.var_col = var_col

        for col in [self.p_col, self.effect_size_col, self.contrast_col, self.var_col]:
            if col not in self._res.columns:
                raise ValueError(f"Column {col} does not exist in the results data frame!")

        for col in [self.p_col, self.effect_size_col]:
            if not np.issubdtype(self._res[col].dtype, np.number):  # type: ignore
                raise ValueError(f"Column {col} must be numeric!")

    @property
    def contrasts(self) -> list | None:
        """Get a list of all contrast available in the results df"""
        if self.contrast_col is None:
            return None
        else:
            return self._res[self.contrast_col].unique().tolist()

    def p_adjust(self, method: Literal["fdr"] = "fdr", adj_col_name="adj_p_value") -> None:
        """Multiple testing correction for p-values

        Adds a new column to the results dataframe and updates the pointer `p_col`.

        Parameters
        ----------
        method
            method to use for multiple testing correction. Currently only fdr is implemented.
        adj_col_name
            Col name used for the adjusted p values.
        """
        if method != "fdr":
            raise ValueError("Currently only FDR is implemented")

        try:
            import statsmodels.stats.multitest
        except ImportError:
            raise ImportError(
                "FDR correction requires statsmodels to be installed: run `!pip install statsmodels` and try again!"
            ) from None

        self._res[adj_col_name] = statsmodels.stats.multitest.fdrcorrection(self._res[self.p_col])[1]
        self.p_col = adj_col_name

    def get_df(self, contrast: str | None = None) -> pd.DataFrame:
        """
        Get a copy of the results dataframe for a given contrast

        If contrast is None, return the entire dataframe without filtering.
        """
        if contrast is None:
            return self._res.copy()
        else:
            return self._res.loc[lambda x: x[self.contrast_col] == contrast].copy()

    def summary(self, *, cutoffs: Sequence[float] = (0.1, 0.05, 0.01, 0.001, 0.0001)) -> pd.DataFrame:
        """Obtain a summary data frame of differential expression results"""
        dfs = []
        for contr in [None] if self.contrasts is None else self.contrasts:
            tmp_res = self.get_df(contr)
            dfs.append(
                pd.DataFrame(
                    {
                        "total": [np.sum(tmp_res[self.p_col] < c) for c in cutoffs],
                        "up": [
                            np.sum((tmp_res[self.p_col] < c) & (tmp_res[self.effect_size_col] > 0)) for c in cutoffs
                        ],
                        "down": [
                            np.sum((tmp_res[self.p_col] < c) & (tmp_res[self.effect_size_col] < 0)) for c in cutoffs
                        ],
                        "contrast": contr,
                    },
                    index=[f"p < {c}" for c in cutoffs],
                )
            )
        return pd.concat(dfs)

    def plot_volcano(
        self,
        contrast: str | None = None,
        *,
        pval_thresh: float = 0.05,
        log2fc_thresh: float = 0.75,
        to_label: int | list[str] = 5,
        s_curve: bool | None = False,
        colors: list[str] | None = None,
        color_dict: dict[str, list[str]] | None = None,
        shape_dict: dict[str, list[str]] | None = None,
        size_col: str | None = None,
        fontsize: int = 10,
        top_right_frame: bool = False,
        figsize: tuple[int, int] = (5, 5),
        legend_pos: tuple[float, float] = (1.6, 1),
        point_sizes: tuple[int, int] = (15, 150),
        shapes: list[str] | None = None,
        shape_order: list[str] | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        return_fig: bool = False,
        **kwargs: int,
    ) -> Figure | None:
        """Create a volcano plot from a pandas DataFrame or AnnData.

        Parameters
        ----------
        pval_thresh
            Threshold p value for significance, by default 0.05
        log2fc_thresh
            Threshold for log2 fold change significance, by default 0.75
        to_label
            Number of top genes or list of genes to label, by default 5
        s_curve
            Whether to use a reciprocal threshold for up and down gene determination, by default False
        colors
            Colors for [non-DE, up, down] genes. Defaults to ['gray', '#D62728', '#1F77B4'].
        varm_key
            Key in AnnData.varm slot to use for plotting if an AnnData object was passed.
        color_dict
            Dictionary for coloring dots by categories.
        shape_dict
            Dictionary for shaping dots by categories.
        size_col
            Column name to size points by.
        fontsize
            Size of gene labels, by default 10
        top_right_frame
            Whether to show the top and right frame of the plot, by default False
        figsize
            Size of the figure, by default (5, 5)
        legend_pos
            Position of the legend as determined by matplotlib, by default (1.6, 1)
        point_sizes
            Lower and upper bounds of point sizes, by default (15, 150)
        shapes
            List of matplotlib marker ids.
        shape_order
            Order of categories for shapes.
        x_label
            Label for the x-axis.
        y_label
            Label for the y-axis.
        return_fig
            Whether to return the figure, by default False
        **kwargs
            Additional arguments for seaborn.scatterplot.

        Returns
        -------
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples
        --------
        >>> # Example with EdgeR
        >>> import pertpy as pt
        >>> adata = pt.dt.zhang_2021()
        >>> adata.layers["counts"] = adata.X.copy()
        >>> ps = pt.tl.PseudobulkSpace()
        >>> pdata = ps.compute(
        ...     adata,
        ...     target_col="Patient",
        ...     groups_col="Cluster",
        ...     layer_key="counts",
        ...     mode="sum",
        ...     min_cells=10,
        ...     min_counts=1000,
        ... )
        >>> edgr = pt.tl.EdgeR(pdata, design="~Efficacy+Treatment")
        >>> edgr.fit()
        >>> res_df = edgr.test_contrasts(
        ...     edgr.contrast(column="Treatment", baseline="Chemo", group_to_compare="Anti-PD-L1+Chemo")
        ... )
        >>> edgr.plot_volcano(res_df, log2fc_thresh=0)

        """
        if colors is None:
            colors = ["gray", "#D62728", "#1F77B4"]

        if self.contrasts is not None and contrast is None:
            raise ValueError(
                "If multiple contrasts are stored in the DEResults object, the volcano plot function requires you to specify a contrast"
            )

        def _pval_reciprocal(lfc: float) -> float:
            """
            Function for relating -log10(pvalue) and logfoldchange in a reciprocal.

            Used for plotting the S-curve
            """
            return pval_thresh / (lfc - log2fc_thresh)

        def _map_shape(symbol: str) -> str:
            if shape_dict is not None:
                for k in shape_dict.keys():
                    if shape_dict[k] is not None and symbol in shape_dict[k]:
                        return k
            return "other"

        def _map_genes_categories(
            row: pd.Series,
            log2fc_col: str,
            nlog10_col: str,
            log2fc_thresh: float,
            pval_thresh: float | None = None,
            s_curve: bool = False,
        ) -> str:
            """
            Map genes to categorize based on log2fc and pvalue.

            These categories are used for coloring the dots.
            Used when no color_dict is passed, sets up/down/nonsignificant.
            """
            log2fc = row[log2fc_col]
            nlog10 = row[nlog10_col]

            if s_curve:
                # S-curve condition for Up or Down categorization
                reciprocal_thresh = _pval_reciprocal(abs(log2fc))
                if log2fc > log2fc_thresh and nlog10 > reciprocal_thresh:
                    return "Up"
                elif log2fc < -log2fc_thresh and nlog10 > reciprocal_thresh:
                    return "Down"
                else:
                    return "not DE"
            else:
                # Standard condition for Up or Down categorization
                if log2fc > log2fc_thresh and nlog10 > pval_thresh:
                    return "Up"
                elif log2fc < -log2fc_thresh and nlog10 > pval_thresh:
                    return "Down"
                else:
                    return "not DE"

        def _map_genes_categories_highlight(
            row: pd.Series,
            *,
            log2fc_col: str,
            nlog10_col: str,
            log2fc_thresh: float,
            pval_thresh: float,
            s_curve: bool = False,
            symbol_col: str,
        ) -> str:
            """
            Map genes to categorize based on log2fc and pvalue.

            These categories are used for coloring the dots.
            Used when color_dict is passed, sets DE / not DE for background and user supplied highlight genes.
            """
            log2fc = row[log2fc_col]
            nlog10 = row[nlog10_col]
            symbol = row[symbol_col]

            if color_dict is not None:
                for k in color_dict.keys():
                    if symbol in color_dict[k]:
                        return k

            if s_curve:
                # Use S-curve condition for filtering DE
                if nlog10 > _pval_reciprocal(abs(log2fc)) and abs(log2fc) > log2fc_thresh:
                    return "DE"
                return "not DE"
            else:
                # Use standard condition for filtering DE
                if abs(log2fc) < log2fc_thresh or nlog10 < pval_thresh:
                    return "not DE"
                return "DE"

        df = self.get_df(contrast)

        # clean and replace 0s as they would lead to -inf
        if df[[self.effect_size_col, self.p_col]].isnull().values.any():
            print("NaNs encountered, dropping rows with NaNs")
            df = df.dropna(subset=[self.effect_size_col, self.p_col])

        if df[self.p_col].min() == 0:
            # TODO use np.nextafter https://stackoverflow.com/questions/38477908/smallest-positive-float64-number
            print("0s encountered for p value, replacing with 1e-323")
            df.loc[df[self.p_col] == 0, self.p_col] = 1e-323

        # convert p value threshold to nlog10
        pval_thresh = -np.log10(pval_thresh)
        # make nlog10 column
        df["nlog10"] = -np.log10(df[self.p_col])
        y_max = df["nlog10"].max() + 1
        # make a column to pick top genes
        df["top_genes"] = df["nlog10"] * df[self.effect_size_col]

        # Label everything with assigned color / shape
        if shape_dict or color_dict:
            combined_labels = []
            if isinstance(shape_dict, dict):
                combined_labels.extend([item for sublist in shape_dict.values() for item in sublist])
            if isinstance(color_dict, dict):
                combined_labels.extend([item for sublist in color_dict.values() for item in sublist])
            label_df = df[df[self.var_col].isin(combined_labels)]

        # Label top n_gens
        elif isinstance(to_label, int):
            label_df = pd.concat(
                (
                    df.sort_values("top_genes")[-to_label:],
                    df.sort_values("top_genes")[0:to_label],
                )
            )

        # assume that a list of genes was passed to label
        else:
            label_df = df[df[self.var_col].isin(to_label)]

        # By default mode colors by up/down if no dict is passed

        if color_dict is None:
            df["color"] = df.apply(
                lambda row: _map_genes_categories(
                    row,
                    log2fc_col=self.effect_size_col,
                    nlog10_col="nlog10",
                    log2fc_thresh=log2fc_thresh,
                    pval_thresh=pval_thresh,
                    s_curve=s_curve,
                ),
                axis=1,
            )

            # order of colors
            hues = ["not DE", "Up", "Down"][: len(df.color.unique())]

        else:
            df["color"] = df.apply(
                lambda row: _map_genes_categories_highlight(
                    row,
                    log2fc_col=self.effect_size_col,
                    nlog10_col="nlog10",
                    log2fc_thresh=log2fc_thresh,
                    pval_thresh=pval_thresh,
                    symbol_col=self.var_col,
                    s_curve=s_curve,
                ),
                axis=1,
            )

            user_added_cats = [x for x in df.color.unique() if x not in ["DE", "not DE"]]
            hues = ["DE", "not DE"] + user_added_cats

            # order of colors
            hues = hues[: len(df.color.unique())]
            colors = [
                "dimgrey",
                "lightgrey",
                "tab:blue",
                "tab:orange",
                "tab:green",
                "tab:red",
                "tab:purple",
                "tab:brown",
                "tab:pink",
                "tab:olive",
                "tab:cyan",
            ]

        # coloring if dictionary passed, subtle background + highlight
        # map shapes if dictionary exists
        if shape_dict is not None:
            df["shape"] = df[self.var_col].map(_map_shape)
            user_added_cats = [x for x in df["shape"].unique() if x != "other"]
            shape_order = ["other"] + user_added_cats
            if shapes is None:
                shapes = ["o", "^", "s", "X", "*", "d"]
            shapes = shapes[: len(df["shape"].unique())]
            shape_col = "shape"
        else:
            shape_col = None

        # build palette
        colors = colors[: len(df.color.unique())]

        # We want plot highlighted genes on top + at bigger size, split dataframe
        df_highlight = None
        if shape_dict or color_dict:
            label_genes = label_df[self.var_col].unique()
            df_highlight = df[df[self.var_col].isin(label_genes)]
            df = df[~df[self.var_col].isin(label_genes)]

        plt.figure(figsize=figsize)
        # Plot non-highlighted genes
        ax = sns.scatterplot(
            data=df,
            x=self.effect_size_col,
            y="nlog10",
            hue="color",
            hue_order=hues,
            palette=colors,
            size=size_col,
            sizes=point_sizes,
            style=shape_col,
            style_order=shape_order,
            markers=shapes,
            **kwargs,
        )
        # Plot highlighted genes
        if df_highlight is not None:
            ax = sns.scatterplot(
                data=df_highlight,
                x=self.effect_size_col,
                y="nlog10",
                hue="color",
                hue_order=hues,
                palette=colors,
                size=size_col,
                sizes=point_sizes,
                style=shape_col,
                style_order=shape_order,
                markers=shapes,
                legend=False,
                edgecolor="black",
                linewidth=1,
                **kwargs,
            )

        # plot vertical and horizontal lines
        if s_curve:
            x = np.arange((log2fc_thresh + 0.000001), y_max, 0.01)
            y = _pval_reciprocal(x)
            ax.plot(x, y, zorder=1, c="k", lw=2, ls="--")
            ax.plot(-x, y, zorder=1, c="k", lw=2, ls="--")

        else:
            ax.axhline(pval_thresh, zorder=1, c="k", lw=2, ls="--")
            ax.axvline(log2fc_thresh, zorder=1, c="k", lw=2, ls="--")
            ax.axvline(log2fc_thresh * -1, zorder=1, c="k", lw=2, ls="--")
        plt.ylim(0, y_max)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # make labels
        texts = []
        for i in range(len(label_df)):
            txt = plt.text(
                x=label_df.iloc[i][self.effect_size_col],
                y=label_df.iloc[i].nlog10,
                s=label_df.iloc[i][self.var_col],
                fontsize=fontsize,
            )

            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])
            texts.append(txt)

        adjustText.adjust_text(texts, arrowprops={"arrowstyle": "-", "color": "k", "zorder": 5})

        # make things pretty
        for axis in ["bottom", "left", "top", "right"]:
            ax.spines[axis].set_linewidth(2)

        if not top_right_frame:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax.tick_params(width=2)
        plt.xticks(size=11, fontsize=10)
        plt.yticks(size=11)

        # Set default axis titles
        if x_label is None:
            x_label = self.effect_size_col
        if y_label is None:
            y_label = f"-$log_{{10}}$ {self.p_col}"

        plt.xlabel(x_label, size=15)
        plt.ylabel(y_label, size=15)

        plt.legend(loc=1, bbox_to_anchor=legend_pos, frameon=False)

        if return_fig:
            return plt.gcf()
        plt.show()
        return None

    def plot_paired(
        self,
        groupby: str,
        pairedby: str,
        contrast: str | None = None,
        *,
        groups: Sequence[str] | None = None,
        var_names: Sequence[str] | None = None,
        n_top_vars: int = 15,
        n_cols: int = 4,
        panel_size: tuple[int, int] = (5, 5),
        show_legend: bool = True,
        size: int = 10,
        y_label: str = "expression",
        pvalue_template=lambda x: f"p={x:.2e}",
        boxplot_properties=None,
        palette=None,
        return_fig: bool = False,
    ) -> Figure | None:
        """Creates a pairwise expression plot from a Pandas DataFrame or Anndata.

        Visualizes a panel of paired scatterplots per variable.

        Parameters
        ----------
        groupby
            .obs column containing the grouping. Must contain exactly two different values.
        pairedby
            .obs column containing the pairing (e.g. "patient_id"). If None, an independent t-test is performed.
        contrast
            If multiple contrasts are stored in the results data frame, you need to specify one contrast here.
        groups
            If the AnnData object contains more than two unique values in `pairedby`, you need
            to specify the two categories you'd like to show in the plot.
        var_names
            Variables to plot.
        n_top_vars
            Number of top variables to plot. Default: 15.
        layer
            Layer to use for plotting.
        n_cols
            Number of columns in the plot. Default: 4.
        panel_size
            Size of each panel. Default: (5, 5).
        show_legend
            Whether to show the legend. Default: True.
        size
            Size of the points. Default: 10.
        y_label
            Label for the y-axis. Default: "expression".
        pvalue_template
            Template for the p-value string displayed in the title of each panel.
        boxplot_properties
            Additional properties for the boxplot, passed to seaborn.boxplot.
        palette
            Color palette for the line- and stripplot.
        return_fig
            If True, return the figure. Default: False.

        Returns
        -------
        Figure or None
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples
        --------
        >>> # Example with EdgeR
        >>> import pertpy as pt
        >>> adata = pt.dt.zhang_2021()
        >>> adata.layers["counts"] = adata.X.copy()
        >>> ps = pt.tl.PseudobulkSpace()
        >>> pdata = ps.compute(
        ...     adata,
        ...     target_col="Patient",
        ...     groups_col="Cluster",
        ...     layer_key="counts",
        ...     mode="sum",
        ...     min_cells=10,
        ...     min_counts=1000,
        ... )
        >>> edgr = pt.tl.EdgeR(pdata, design="~Efficacy+Treatment")
        >>> edgr.fit()
        >>> res_df = edgr.test_contrasts(
        ...     edgr.contrast(column="Treatment", baseline="Chemo", group_to_compare="Anti-PD-L1+Chemo")
        ... )
        >>> edgr.plot_paired(pdata, results_df=res_df, n_top_vars=8, groupby="Treatment", pairedby="Efficacy")

        """
        if self.contrasts is not None and contrast is None:
            raise ValueError(
                "If multiple contrasts are stored in the DEResults object, the volcano plot function requires you to specify a contrast"
            )
        if self.adata is None:
            raise ValueError("plot_paired requires that DEResult has been initalized with an AnnData object")
        if groups is not None:
            tmp_adata = self.adata[self.adata.obs[groupby].isin(groups), :].copy()
            tmp_adata.obs[groupby] = tmp_adata.obs[groupby].cat.remove_unused_categories()
        else:
            tmp_adata = self.adata
            groups = tmp_adata.obs[groupby].unique()
        if len(groups) != 2:
            raise ValueError("The number of groups in the group_by column must be exactly 2 to enable paired testing")
        results_df = self.get_df(contrast)

        print(tmp_adata.obs[groupby].cat.categories)

        if boxplot_properties is None:
            boxplot_properties = {}

        if var_names is None:
            var_names = self.get_df(contrast).head(n_top_vars)[self.var_col].tolist()
        tmp_adata = tmp_adata[:, var_names]

        groupby_cols = [pairedby, groupby]
        df = tmp_adata.obs.loc[:, groupby_cols].join(tmp_adata.to_df(self.layer))

        # remove unpaired samples
        paired_samples = set(df[df[groupby] == groups[0]][pairedby]) & set(df[df[groupby] == groups[1]][pairedby])
        df = df[df[pairedby].isin(paired_samples)]
        removed_samples = tmp_adata.obs[pairedby].nunique() - len(df[pairedby].unique())
        if removed_samples > 0:
            logger.warning(f"{removed_samples} unpaired samples removed")

        pvalues = results_df.set_index(self.var_col).loc[var_names, self.p_col].values
        df.reset_index(drop=False, inplace=True)

        # transform data for seaborn
        df_melt = df.melt(
            id_vars=groupby_cols,
            var_name="var",
            value_name="val",
        )

        n_panels = len(var_names)
        nrows = math.ceil(n_panels / n_cols)
        ncols = min(n_cols, n_panels)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * panel_size[0], nrows * panel_size[1]),
            tight_layout=True,
            squeeze=False,
        )
        axes = axes.flatten()
        for i, (var, ax) in enumerate(zip_longest(var_names, axes)):
            if var is not None:
                sns.boxplot(
                    x=groupby,
                    data=df_melt.loc[df_melt["var"] == var],
                    y="val",
                    ax=ax,
                    color="white",
                    fliersize=0,
                    **boxplot_properties,
                )
                if pairedby is not None:
                    sns.lineplot(
                        x=groupby,
                        data=df_melt.loc[df_melt["var"] == var],
                        y="val",
                        ax=ax,
                        hue=pairedby,
                        legend=False,
                        errorbar=None,
                        palette=palette,
                    )
                jitter = 0 if pairedby else True
                sns.stripplot(
                    x=groupby,
                    data=df_melt.loc[df_melt["var"] == var],
                    y="val",
                    ax=ax,
                    hue=pairedby,
                    jitter=jitter,
                    size=size,
                    linewidth=1,
                    palette=palette,
                )

                ax.set_xlabel("")
                ax.tick_params(
                    axis="x",
                    labelsize=15,
                )
                ax.legend().set_visible(False)
                ax.set_ylabel(y_label)
                ax.set_title(f"{var}\n{pvalue_template(pvalues[i])}")
            else:
                ax.set_visible(False)
        fig.tight_layout()

        if show_legend is True:
            axes[n_panels - 1].legend().set_visible(True)
            axes[n_panels - 1].legend(
                bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=tmp_adata.obs[pairedby].nunique()
            )

        plt.tight_layout()
        if return_fig:
            return plt.gcf()
        plt.show()
        return None

    def plot_fold_change(
        self,
        contrast: str | None = None,
        *,
        var_names: Sequence[str] | None = None,
        n_top_vars: int = 15,
        y_label: str = "Log2 fold change",
        figsize: tuple[int, int] = (10, 5),
        return_fig: bool = False,
        **barplot_kwargs,
    ) -> Figure | None:
        """Plot a metric from the results as a bar chart.

        Parameters
        ----------
        var_names
            Variables to plot. If None, the top n_top_vars variables based on the log2 fold change are plotted.
        n_top_vars
            Number of top variables to plot. The top and bottom n_top_vars variables are plotted, respectively.
        y_label
            Label for the y-axis.
        figsize
            Size of the figure.
        return_fig
            If True, return the figure. Default: False.
        **barplot_kwargs
            Additional arguments for seaborn.barplot.

        Returns
        -------
        Figure or None
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples
        --------
        >>> # Example with EdgeR
        >>> import pertpy as pt
        >>> adata = pt.dt.zhang_2021()
        >>> adata.layers["counts"] = adata.X.copy()
        >>> ps = pt.tl.PseudobulkSpace()
        >>> pdata = ps.compute(
        ...     adata,
        ...     target_col="Patient",
        ...     groups_col="Cluster",
        ...     layer_key="counts",
        ...     mode="sum",
        ...     min_cells=10,
        ...     min_counts=1000,
        ... )
        >>> edgr = pt.tl.EdgeR(pdata, design="~Efficacy+Treatment")
        >>> edgr.fit()
        >>> res_df = edgr.test_contrasts(
        ...     edgr.contrast(column="Treatment", baseline="Chemo", group_to_compare="Anti-PD-L1+Chemo")
        ... )
        >>> edgr.plot_fold_change(res_df)

        """
        if self.contrasts is not None and contrast is None:
            raise ValueError(
                "If multiple contrasts are stored in the DEResults object, the volcano plot function requires you to specify a contrast"
            )

        if var_names is None:
            var_names = (
                self._res.sort_values(self.effect_size_col, ascending=False).head(n_top_vars)[self.var_col].tolist()
            )
            var_names += (
                self._res.sort_values(self.effect_size_col, ascending=True).head(n_top_vars)[self.var_col].tolist()
            )
            assert len(var_names) == 2 * n_top_vars

        df = self.get_df(contrast).loc[lambda x: x[self.var_col].isin(var_names)].copy()
        df.sort_values(self.effect_size_col, ascending=False, inplace=True)

        plt.figure(figsize=figsize)
        sns.barplot(
            x=self.var_col,
            y=self.effect_size_col,
            data=df,
            palette="RdBu",
            legend=False,
            **barplot_kwargs,
        )
        plt.xticks(rotation=90)
        plt.xlabel("")
        plt.ylabel(y_label)

        if return_fig:
            return plt.gcf()
        plt.show()
        return None

    def plot_multicomparison_fc(
        self,
        *,
        n_top_vars=15,
        marker_size: int = 100,
        figsize: tuple[int, int] = (10, 2),
        x_label: str = "Contrast",
        y_label: str = "Gene",
        return_fig: bool = False,
        **heatmap_kwargs,
    ) -> Figure | None:
        """Plot a matrix of log2 fold changes from the results.

        Parameters
        ----------
        n_top_vars
            Number of top variables to plot per group. Default: 15.
        marker_size
            Size of the biggest marker for significant variables. Default: 100.
        figsize
            Size of the figure. Default: (10, 2).
        x_label
            Label for the x-axis. Default: "Contrast".
        y_label
            Label for the y-axis. Default: "Gene".
        return_fig
            If True, return the figure, otherwise None. Default: False.
        **heatmap_kwargs
            Additional arguments for seaborn.heatmap.

        Returns
        -------
        If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples
        --------
        >>> # Example with EdgeR
        >>> import pertpy as pt
        >>> adata = pt.dt.zhang_2021()
        >>> adata.layers["counts"] = adata.X.copy()
        >>> ps = pt.tl.PseudobulkSpace()
        >>> pdata = ps.compute(
        ...     adata,
        ...     target_col="Patient",
        ...     groups_col="Cluster",
        ...     layer_key="counts",
        ...     mode="sum",
        ...     min_cells=10,
        ...     min_counts=1000,
        ... )
        >>> edgr = pt.tl.EdgeR(pdata, design="~Efficacy+Treatment")
        >>> res_df = edgr.compare_groups(pdata, column="Efficacy", baseline="SD", groups_to_compare=["PR", "PD"])
        >>> edgr.plot_multicomparison_fc(res_df)

        """
        if self.contrasts is None:
            raise ValueError("The multicomparison plot requires the contrasts column to be set.")

        def _get_significance(p_val):
            if p_val < 0.001:
                return "< 0.001"
            elif p_val < 0.01:
                return "< 0.01"
            elif p_val < 0.1:
                return "< 0.1"
            else:
                return "n.s."

        results_df = self.get_df()
        results_df["abs_log_fc"] = results_df[self.effect_size_col].abs()
        results_df["significance"] = results_df[self.p_col].apply(_get_significance)

        var_names = (
            results_df.groupby(self.contrast_col)
            .apply(lambda x: x.head(n_top_vars))
            .sort_values(self.p_col)
            .drop_duplicates(self.var_col, keep="first")[self.var_col]
        )

        results_df = results_df[results_df[self.var_col].isin(var_names)]
        df = results_df.pivot(index=self.contrast_col, columns=self.var_col, values=self.effect_size_col)[var_names]

        plt.figure(figsize=figsize)
        sns.heatmap(df, **heatmap_kwargs, cmap="coolwarm", center=0, cbar_kws={"label": "Log2 fold change"})

        _size = {"< 0.001": marker_size, "< 0.01": math.floor(marker_size / 2), "< 0.1": math.floor(marker_size / 4)}
        x_locs, x_labels = plt.xticks()[0], [label.get_text() for label in plt.xticks()[1]]
        y_locs, y_labels = plt.yticks()[0], [label.get_text() for label in plt.yticks()[1]]

        for _i, row in results_df.iterrows():
            if row["significance"] != "n.s.":
                plt.scatter(
                    x=x_locs[x_labels.index(row[self.var_col])],
                    y=y_locs[y_labels.index(row[self.contrast_col])],
                    s=_size[row["significance"]],
                    marker="*",
                    c="white",
                )

        plt.scatter([], [], s=marker_size, marker="*", c="black", label="< 0.001")
        plt.scatter([], [], s=math.floor(marker_size / 2), marker="*", c="black", label="< 0.01")
        plt.scatter([], [], s=math.floor(marker_size / 4), marker="*", c="black", label="< 0.1")
        plt.legend(title="Significance", bbox_to_anchor=(1.2, -0.05))

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if return_fig:
            return plt.gcf()
        plt.show()
        return None
