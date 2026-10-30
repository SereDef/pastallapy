"""Tagliatelle: multi-column sankey diagrams.

A tagliatelle plot follows the same observations through two or more
categorical labellings -- before and after, or a whole run of follow-ups -- as
ribbons flowing between neighbouring columns.

    import pandas as pd
    import pastallapy as pp

    pp.tagliatelle(df, "year0", "year5", "year10")

Every stage shares one drawing order and one set of colours, so a category sits
at the same height in every column and whatever did not change runs straight
across.
"""

from __future__ import annotations

import textwrap
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from .style import Style, resolve_colors

__all__ = ["Bar", "agreement", "tagliatelle"]

#: Internal weight column. Never a user column name (checked in `_prepare`).
VALUE = "__weight__"
_SAMPLES = 128  # points along one ribbon edge


@dataclass(frozen=True)
class Bar:
    """One category block, as drawn. ``top`` and ``bottom`` are in weight units."""

    stage: int
    label: Any
    value: float
    top: float
    bottom: float
    color: str

    @property
    def middle(self) -> float:
        return (self.top + self.bottom) / 2


def tagliatelle(
    data: Any,
    *columns: Any,
    ax: Axes | None = None,
    weights: Any = None,
    labels: Sequence[Any] | None = None,
    other: str | None = "Other",
    missing: str | None = None,
    title: str | None = None,
    titles: Any = None,
    show_agreement: bool = False,
    style: Style | None = None,
    **style_options: Any,
) -> Axes:
    """Draw a sankey diagram over two or more stages and return its axes.

    Parameters
    ----------
    data
        Dataframe, or dict of columns, holding every stage.
    *columns
        Column names, in drawing order left to right. At least two. Omit them
        to use every column of ``data`` (bar the ``weights`` one).
    ax
        Axes to draw into. A new figure is created when omitted.
    weights
        Column name or array of weights, for data that is already aggregated.
        ``None`` counts rows.
    labels
        Categories to keep, top to bottom, shared by every stage. Categories
        left out are grouped into ``other``. Without ``labels``, the order
        comes from the dtype's categories if any column is categorical, else
        largest first.
    other
        Label collecting categories not listed in ``labels``; ``None`` drops
        those rows instead.
    missing
        Label for missing values; ``None`` drops rows that have any.
    title
        Title above the plot.
    titles
        Headers over the columns. ``None`` uses the column names, ``False``
        draws none, or pass one string per stage.
    show_agreement
        Write the share of the weight that stays put beneath each gap between
        neighbouring stages (see :func:`agreement`).
    style, **style_options
        A :class:`~pastallapy.Style`, or its options as keywords, e.g.
        ``gap=0.05, ribbon_color="target"``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the diagram was drawn into.

    Examples
    --------
    >>> pp.tagliatelle(df, "before", "after", labels=["low", "mid", "high"])
    >>> pp.tagliatelle(counts, "t0", "t1", "t2", weights="n", curve=0.5)
    """
    style = (Style() if style is None else style).replace(**style_options)

    stages, names = _prepare(data, columns, weights, missing)
    stages, order = _order(stages, names, labels, other, missing)

    total = float(stages[VALUE].sum())
    colors = resolve_colors(
        order,
        style.palette,
        neutral_color=style.neutral_color,
        neutral_labels=[label for label in (other, missing) if label is not None],
    )

    xs = _stage_edges(len(names), style.node_width)
    bars = [
        _stack(stages, name, stage, order, total, style.gap, colors)
        for stage, name in enumerate(names)
    ]
    height = total + style.gap * total * (len(order) - 1)

    own_figure = ax is None
    if own_figure:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

    _draw_ribbons(ax, stages, names, bars, xs, style)
    _draw_bars(ax, bars, xs, total, style)
    headers = _draw_titles(ax, titles, names, xs, height, style)
    if title:
        ax.set_title(
            title,
            fontsize=style.title_size,
            fontweight="bold",
            pad=14 + (1.4 * style.stage_title_size if headers else 0),
        )
    if show_agreement:
        _draw_agreement(ax, stages, names, xs, other, missing, style)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.01 * height, 1.01 * height)
    ax.set_axis_off()
    if own_figure:
        ax.figure.subplots_adjust(
            left=style.margin_x,
            right=1 - style.margin_x,
            top=0.86 if title else 0.92 if headers else 0.97,
            bottom=0.10 if show_agreement else 0.03,
        )
    return ax


def agreement(
    data: Any,
    *columns: Any,
    weights: Any = None,
    exclude: Sequence[Any] = (),
) -> list[float]:
    """Share of the weight that stays in the same category, as a percentage.

    Returns one value per pair of neighbouring stages, so the list is one
    shorter than ``columns``. Rows whose category is listed in ``exclude`` at
    either end of a pair sit out that pair, which is how a catch-all group is
    kept from counting as a match.
    """
    stages, names = _prepare(data, columns, weights, missing=None)
    return _agreements(stages, names, exclude)


def _agreements(
    stages: pd.DataFrame, names: Sequence[Any], exclude: Sequence[Any]
) -> list[float]:
    skip = {label for label in exclude if label is not None}
    values = []
    for a, b in pairwise(names):
        scored = stages.loc[~(stages[a].isin(skip) | stages[b].isin(skip))]
        total = float(scored[VALUE].sum())
        if total == 0:
            values.append(float("nan"))
            continue
        matched = float(scored.loc[scored[a] == scored[b], VALUE].sum())
        values.append(100 * matched / total)
    return values


# ---------------------------------------------------------------------------
# data preparation


def _prepare(
    data: Any, columns: Sequence[Any], weights: Any, missing: str | None
) -> tuple[pd.DataFrame, list[Any]]:
    """Return one row per observation, a column per stage plus ``VALUE``."""
    if isinstance(data, Mapping):
        frame = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        frame = data
    else:
        raise TypeError(
            f"data must be a dataframe or a dict of columns, got {type(data).__name__}"
        )
    if not frame.columns.is_unique:
        raise ValueError("data has duplicate column names; make them unique")

    if columns:
        names = list(columns)
    else:
        names = [col for col in frame.columns if not (col == weights and isinstance(weights, str))]
    if len(names) < 2:
        raise ValueError(
            "name at least two columns to compare, e.g. tagliatelle(df, 'before', 'after')"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"the stage columns must be distinct, got {names}")
    if VALUE in names:
        raise ValueError(f"{VALUE!r} is reserved; rename that column")
    unknown = [col for col in names if col not in frame.columns]
    if unknown:
        raise KeyError(
            f"column(s) {unknown} are not in the data; available columns are "
            f"{list(frame.columns)}"
        )

    stages = frame.loc[:, names].copy()
    stages[VALUE] = _weights(frame, weights, len(frame))

    na = stages[names].isna().any(axis=1)
    if na.any():
        if missing is None:
            warnings.warn(
                f"dropped {int(na.sum())} row(s) with missing values; pass "
                "missing='Missing' to show them instead",
                stacklevel=3,
            )
            stages = stages.loc[~na]
        else:
            for name in names:
                column = stages[name]
                if isinstance(column.dtype, pd.CategoricalDtype):
                    column = column.astype(object)
                stages[name] = column.where(column.notna(), missing)

    stages = stages.loc[stages[VALUE] > 0]
    if stages.empty:
        raise ValueError("no data left to plot")
    return stages.reset_index(drop=True), names


def _weights(frame: pd.DataFrame, weights: Any, n_rows: int) -> np.ndarray:
    if weights is None:
        return np.ones(n_rows)
    if isinstance(weights, str):
        if weights not in frame.columns:
            raise KeyError(
                f"weights column {weights!r} is not in the data; available columns "
                f"are {list(frame.columns)}"
            )
        values = np.asarray(frame[weights], dtype=float)
    else:
        values = np.asarray(weights, dtype=float).ravel()
    if values.size != n_rows:
        raise ValueError(f"weights has {values.size} values but the data has {n_rows} rows")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("weights must be finite and non-negative")
    return values


def _order(
    stages: pd.DataFrame,
    names: Sequence[Any],
    labels: Sequence[Any] | None,
    other: str | None,
    missing: str | None,
) -> tuple[pd.DataFrame, list[Any]]:
    """Settle the one drawing order shared by every stage."""
    if labels is None:
        categories = [
            category
            for name in names
            if isinstance(stages[name].dtype, pd.CategoricalDtype)
            for category in stages[name].dtype.categories
        ]
        totals = _totals(stages, names)
        if categories:
            return stages, [c for c in dict.fromkeys(categories) if c in totals]
        return stages, sorted(totals, key=lambda label: -totals[label])

    if isinstance(labels, (str, bytes, Mapping)):
        raise TypeError(
            f"labels must be a sequence of categories, got {type(labels).__name__}; "
            "pass colours with palette= instead"
        )
    wanted = list(dict.fromkeys(labels))
    keep = set(wanted) | ({missing} if missing is not None else set())
    extra = [label for label in _totals(stages, names) if label not in keep]
    if extra:
        warnings.warn(
            f"{len(extra)} category/ies are not in labels and were "
            + (f'grouped into "{other}"' if other else "dropped")
            + f": {_preview(extra)}",
            stacklevel=3,
        )
        stages = stages.copy()
        if other is None:
            for name in names:
                stages = stages.loc[~stages[name].isin(extra)]
        else:
            renamed = dict.fromkeys(extra, other)
            for name in names:
                stages[name] = stages[name].astype(object).replace(renamed)
        if stages.empty:
            raise ValueError("no data left to plot after applying labels")

    present = _totals(stages, names)
    absent = [label for label in wanted if label not in present]
    if absent:
        warnings.warn(
            f"{len(absent)} label(s) from labels are not in the data and were "
            f"skipped: {_preview(absent)}",
            stacklevel=3,
        )
    order = [label for label in wanted if label in present]
    # Categories created along the way go last.
    order += [
        label
        for label in (other, missing)
        if label is not None and label in present and label not in order
    ]
    if not order:
        raise ValueError("none of the categories in labels is in the data")
    return stages, order


def _totals(stages: pd.DataFrame, names: Sequence[Any]) -> dict[Any, float]:
    """Total weight per category, summed over every stage."""
    totals: dict[Any, float] = {}
    for name in names:
        summed = stages.groupby(name, sort=False, observed=True)[VALUE].sum()
        for label, value in summed.items():
            totals[label] = totals.get(label, 0.0) + float(value)
    return totals


def _preview(labels: Sequence[Any], limit: int = 6) -> str:
    shown = ", ".join(repr(label) for label in list(labels)[:limit])
    return shown if len(labels) <= limit else f"{shown}, ... (+{len(labels) - limit})"


# ---------------------------------------------------------------------------
# layout and drawing


def _stage_edges(n_stages: int, width: float) -> list[float]:
    """Left edge of each stage's bars, spreading the stages across ``[0, 1]``."""
    step = (1 - width) / (n_stages - 1)
    if step <= width:
        raise ValueError(
            f"node_width={width!r} leaves no room for ribbons across {n_stages} "
            f"stages; use less than {1 / n_stages:.3f}"
        )
    return [stage * step for stage in range(n_stages)]


def _stack(
    stages: pd.DataFrame,
    name: Any,
    stage: int,
    order: Sequence[Any],
    total: float,
    gap: float,
    colors: Mapping[Any, str],
) -> list[Bar]:
    """Lay one stage's categories out top to bottom, in the shared order."""
    sums = stages.groupby(name, sort=False, observed=True)[VALUE].sum()
    gap_units = gap * total
    bars, top = [], total + gap_units * (len(order) - 1)
    for label in order:
        value = float(sums.get(label, 0.0))
        bars.append(Bar(stage, label, value, top, top - value, colors[label]))
        top -= value + gap_units
    return bars


def _draw_bars(
    ax: Axes, bars: Sequence[Sequence[Bar]], xs: Sequence[float], total: float, style: Style
) -> None:
    """Draw every bar, labelling the first stage on the left and the last on the right."""
    width = style.node_width
    last = len(bars) - 1
    for stage, (column, x0) in enumerate(zip(bars, xs)):
        for bar in column:
            if bar.value <= 0:
                continue
            ax.add_patch(
                Rectangle(
                    (x0, bar.bottom), width, bar.value, facecolor=bar.color, edgecolor="none"
                )
            )
            if stage not in (0, last):
                continue
            text = _bar_label(bar, total, style)
            if not text:
                continue
            left = stage == 0
            ax.annotate(
                text,
                xy=(x0 if left else x0 + width, bar.middle),
                xytext=(-style.label_pad if left else style.label_pad, 0),
                textcoords="offset points",
                ha="right" if left else "left",
                va="center",
                fontsize=style.font_size,
                annotation_clip=False,
            )


def _draw_ribbons(
    ax: Axes,
    stages: pd.DataFrame,
    names: Sequence[Any],
    bars: Sequence[Sequence[Bar]],
    xs: Sequence[float],
    style: Style,
) -> None:
    """Join neighbouring stages, stacking each ribbon in the order of the other end."""
    width = style.node_width
    u = np.linspace(0, 1, _SAMPLES)
    ease = (1 - style.curve) * u + style.curve * u**2 * (3 - 2 * u)

    for stage, (left, right) in enumerate(pairwise(names)):
        flows = (
            stages.groupby([left, right], sort=False, observed=True)[VALUE].sum().to_dict()
        )
        # One cursor per end: ribbons leave a bar top down in target order and
        # arrive top down in source order, so they never cross within a bar.
        leaving = {bar.label: bar.top for bar in bars[stage]}
        arriving = {bar.label: bar.top for bar in bars[stage + 1]}
        x = xs[stage] + width + (xs[stage + 1] - xs[stage] - width) * u

        for source in bars[stage]:
            for target in bars[stage + 1]:
                value = float(flows.get((source.label, target.label), 0.0))
                if value <= 0:
                    continue
                start = leaving[source.label] - value
                end = arriving[target.label] - value
                leaving[source.label] = start
                arriving[target.label] = end
                lower = start + (end - start) * ease
                ax.fill_between(
                    x,
                    lower,
                    lower + value,
                    facecolor=_ribbon_color(style, source, target),
                    alpha=style.ribbon_alpha,
                    edgecolor="none",
                )


def _ribbon_color(style: Style, source: Bar, target: Bar) -> Any:
    if style.ribbon_color == "source":
        return source.color
    if style.ribbon_color == "target":
        return target.color
    return style.ribbon_color


def _bar_label(bar: Bar, total: float, style: Style) -> str:
    template = style.label_template
    if not template:
        return ""
    label = str(bar.label)
    if style.label_wrap:
        label = textwrap.fill(label, style.label_wrap)
    share = bar.value / total if total else 0.0
    percent = 100 * share
    return template.format(
        label=label,
        value=bar.value,
        share=share,
        percent="<1%" if 0 < percent < 0.5 else f"{percent:.0f}%",
    )


def _draw_titles(
    ax: Axes,
    titles: Any,
    names: Sequence[Any],
    xs: Sequence[float],
    height: float,
    style: Style,
) -> list[Any]:
    if titles is False:
        return []
    if titles is None:
        titles = names
    elif isinstance(titles, (str, bytes)):
        raise ValueError(f"titles must be one string per stage, None or False, got {titles!r}")
    else:
        titles = list(titles)
        if len(titles) != len(names):
            raise ValueError(
                f"titles must have one entry per stage ({len(names)}), got {len(titles)}"
            )
    return [
        ax.annotate(
            str(title),
            xy=(x + style.node_width / 2, height),
            xytext=(0, style.label_pad + 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=style.stage_title_size,
            fontweight="bold",
            annotation_clip=False,
        )
        for x, title in zip(xs, titles)
        if title
    ]


def _draw_agreement(
    ax: Axes,
    stages: pd.DataFrame,
    names: Sequence[Any],
    xs: Sequence[float],
    other: str | None,
    missing: str | None,
    style: Style,
) -> None:
    """Write the unchanged share beneath the gap between each pair of stages."""
    values = _agreements(stages, names, [other, missing])
    for stage, value in enumerate(values):
        middle = (xs[stage] + style.node_width + xs[stage + 1]) / 2
        ax.annotate(
            "match: n/a" if np.isnan(value) else f"match: {value:.0f}%",
            xy=(middle, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -style.label_pad - 4),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=style.annotation_size,
            annotation_clip=False,
        )
