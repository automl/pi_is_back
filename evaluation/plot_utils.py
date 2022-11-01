from __future__ import annotations
from enum import auto

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns


def savefig(fig: plt.Figure, basename: str | Path, extension: str = ".pdf"):
    Path(basename).parent.mkdir(parents=True, exist_ok=True)
    basename = basename.replace(":", "_").replace(" ", "-")
    fig.savefig(str(basename) + extension, bbox_inches="tight", dpi=300)


def plot_return(data: pd.DataFrame, title: str = None, extension: str = ".pdf") -> None:
    group_keys = ["instance", "episode", "policy_name", "seed"]
    groups = data.groupby(group_keys)
    df_return = groups.apply(lambda x: np.sum(a=x["reward"])).reset_index().rename(columns={0: "return"})

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.boxplot(
        data=df_return,
        x="policy_name",
        y="return",
        ax=ax,
        palette=get_color_palette(data),
    )
    ax.set_yscale("log")
    ax.set_title(title)
    fig.set_tight_layout(True)
    plt.show()
    savefig(fig=fig, basename=f"./tmp/figures/boxplot_return_{title}", extension=extension)


def plot_final_reward(data: pd.DataFrame, title: str = None, extension: str = ".pdf") -> None:
    data = data[data["state"] == data["state"].min()]
    group_keys = ["instance", "episode", "policy_name", "seed"]
    groups = data.groupby(group_keys)
    df_return = groups.apply(lambda x: np.sum(a=x["reward"])).reset_index().rename(columns={0: "return"})

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.boxplot(
        data=df_return,
        x="policy_name",
        y="return",
        ax=ax,
        palette=get_color_palette(data),
    )
    # ax.set_yscale("log")
    ax.set_title(title)
    fig.set_tight_layout(True)
    plt.show()
    savefig(fig=fig, basename=f"./tmp/figures/boxplot_incumbentcost_{title}", extension=extension)


def plot_final_regret(
    data: pd.DataFrame, title: str = None, yname: str = "regret", outdir: str = "./tmp/figures", extension: str = ".pdf"
) -> None:
    data = data[data["step"] == data["step"].max()]
    # data = scale(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.violinplot(
        data=data,
        x="policy_name",
        y=yname,
        ax=ax,
        palette=get_color_palette(data),
        cut=0,
    )
    ax = sns.stripplot(x="policy_name", y=yname, data=data, size=2, color="black", linewidth=0, ax=ax)
    # ax = sns.barplot(data=data, x="policy_name", y=yname, ax=ax, palette=get_color_palette(data))
    if not "log" in yname:
        ax.set_yscale("log")
    # else:
    #     ax.set_ylim(0, 1)
    title = title.replace("bbob_", "")
    # ax.set_title(title)
    ax.set_xlabel("schedule")
    ax.set_ylabel("log regret (scaled)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.set_tight_layout(True)
    savefig(fig=fig, basename=f"{outdir}/boxplot/incumbentregret_{title}", extension=extension)
    plt.show()


# Plot action
def plot_action_over_steps(data: pd.DataFrame, hue: Optional[str] = None, extension: str = ".pdf") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.lineplot(data=data, x="step", y="action", hue=hue, ax=ax)
    fig.set_tight_layout(True)
    plt.show()
    savefig(fig=fig, basename="./tmp/figures/action_over_steps", extension=extension)


def plot_reward_over_steps(data: pd.DataFrame, title: str | None = None, extension: str = ".pdf"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.lineplot(data=data, x="step", y="reward", hue="policy_name", ax=ax)
    ax.set_title(title)
    fig.set_tight_layout(True)
    plt.show()
    savefig(fig=fig, basename=f"./tmp/figures/reward_over_steps_{title}", extension=extension)


def plot_regret_over_steps(
    data: pd.DataFrame,
    title: str | None = None,
    yname: str = "regret",
    errorbar: str = "sd",
    outdir: str = "./tmp/figures",
    hue_order: list[str] | None = None,
    extension: str = ".pdf",
    ax: plt.Axes | None = None,
    remove_legend: bool = False
) -> None:
    # data = scale(data)
    fig = None
    if ax is None:
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    n_steps = data["step"].max()
    x = [n_steps * k for k in [0.25, 0.5, 0.75]]

    ax = sns.lineplot(
        data=data,
        x="step",
        y=yname,
        hue="policy_name",
        ax=ax,
        palette=get_color_palette(data, automatic=False),
        errorbar=errorbar,
        hue_order=hue_order,
    )
    if not "log" in yname:
        ax.set_yscale("log")
    # else:
    #     ax.set_ylim(0, 1)
    ymin, ymax = ax.get_ylim()
    # ax.vlines(x=x, ymin=ymin, ymax=ymax, color="grey", alpha=0.5)
    for xi in x:
        ax.axvline(x=xi, color="grey", alpha=0.5)
    xticks = np.linspace(0, n_steps + 1, 5)
    xticks = [int(x) for x in xticks]
    # ax.set_xticks(xticks)
    title = title.replace("bbob_", "")
    # ax.set_title(title)
    ax.set_xlabel("BO evaluations")
    ax.set_ylabel("log regret (scaled)")
    ax.legend(title="schedule")
    plt.setp(ax.get_legend().get_texts(), fontsize="9")
    plt.setp(ax.get_legend().get_title(), fontsize="11")
    if remove_legend:
        ax.get_legend().remove()
    fig.set_tight_layout(True)
    plt.show()
    err = errorbar if errorbar == "_ci" else ""
    basename = f"{outdir}/convergence/regret_over_steps{err}_{title}"
    savefig(fig=fig, basename=basename, extension=extension)
    return ax


def get_group_title(group_keys: list[str], group_id: tuple[Any]) -> str:
    return " ".join([f"{k}:{v}" for k, v in zip(group_keys, group_id)])


def get_color_palette(data: pd.DataFrame, automatic: bool = True):
    names = list(data["policy_name"].unique())
    # names.sort()
    n_colors = len(names)
    colors = sns.color_palette(palette="colorblind", n_colors=n_colors)
    palette = {n: p for n, p in zip(names, colors)}
    # palette["static (EI)"] = "grey"
    # palette["static (PI)"] = "black"
    # palette = "colorblind"
    if not automatic:
        palette = {
            "static (EI)": colors[4],
            "static (PI)": colors[3],
            "round robin": colors[5],
            "random": colors[6],
            "explore-exploit (0.75)": colors[0],
            "explore-exploit (0.5)": colors[1],
            "explore-exploit (0.25)": colors[2],
        }
    return palette


def plot_regret_all_samples(
    data: pd.DataFrame,
    title: str | None = None,
    yname: str = "regret",
    errorbar: str = "sd",
    outdir: str = "./tmp/figures",
    extension: str = ".pdf",
):
    group_df = data
    group_df = group_df.rename(columns={"policy_name": "schedule"})
    grid = sns.FacetGrid(data=group_df, col="schedule")
    grid.map_dataframe(sns.lineplot, x="step", y=yname, hue="seed", errorbar=errorbar)
    grid.set_xlabels("BO evaluations")
    grid.set_ylabels("log regret (scaled)")
    grid.set_titles(template="{col_name}")

    # Reference lines for percentages after which to switch
    n_steps = data["step"].max()
    x = [n_steps * k for k in [0.25, 0.5, 0.75]]
    for xi in x:
        grid.refline(x=xi, color="grey", alpha=0.25, ls="-")

    plt.show()
    basename = f"{outdir}/convergence_perschedule/regret_over_steps_hueseed_{title}"
    fig = grid.figure
    savefig(fig=fig, basename=basename, extension=extension)
