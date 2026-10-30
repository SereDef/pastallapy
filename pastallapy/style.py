"""Look and feel for pastallapy plots.

:class:`Style` holds every appearance option in one place, so a plot is
configured either with keywords::

    pp.tagliatelle(df, "before", "after", gap=0.05, ribbon_color="target")

or with a reusable style object::

    house = pp.Style(palette="tab20", font_size=8)
    pp.tagliatelle(df, "before", "after", style=house)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap, to_hex

__all__ = ["DEFAULT_PALETTE", "NEUTRAL_COLOR", "Style", "resolve_colors"]

#: Qualitative, colour-vision-deficiency friendly default palette (Okabe-Ito based).
DEFAULT_PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky
    "#D55E00",  # vermillion
    "#8C6BB1",  # purple
    "#B2DF8A",  # light green
    "#F0E442",  # yellow
    "#7B4B2A",  # brown
)

#: Colour used for catch-all ("Other") and missing categories.
NEUTRAL_COLOR = "#BDBDBD"

Palette = Mapping[Any, Any] | Sequence[Any] | str | None


@dataclass(frozen=True)
class Style:
    """Appearance of a pastallapy plot.

    Parameters
    ----------
    palette
        Colours for the categories: a ``{label: colour}`` mapping, a sequence
        of colours, the name of a matplotlib colormap, or ``None`` for
        :data:`DEFAULT_PALETTE`. Labels absent from a mapping fall back to the
        default palette, so partial mappings are fine.
    neutral_color
        Colour for the catch-all and missing categories.
    font_size, title_size, stage_title_size, annotation_size
        Text sizes in points.
    label_pad
        Distance between a mark and its label, in points.
    label_wrap
        Wrap labels longer than this many characters. ``None`` leaves them be.
    label_template
        Format string for the bar labels, with the fields ``label``, ``value``,
        ``percent`` and ``share``. Use ``""`` for no labels.
    node_width
        Width of one stage's bars, as a fraction of the plot width. Must leave
        room for the ribbons: less than ``1 / number of stages``.
    gap
        Vertical space between two bars, as a fraction of the total. Being a
        fraction keeps a style usable on data of any size.
    curve
        Ribbon shape, from ``0`` (straight) to ``1`` (fully eased S-curve).
    ribbon_alpha
        Opacity of the ribbons; the bars are always solid.
    ribbon_color
        ``"source"`` or ``"target"`` to follow that end's category colour, or
        any matplotlib colour to paint every ribbon the same.
    figsize, dpi
        Used only when the plotting function has to create a figure itself.
    margin_x
        Fraction of the figure width reserved on each side for the labels that
        sit outside the axes. Widen it for long labels. Only used when the
        plotting function creates the figure itself.
    """

    palette: Palette = None
    neutral_color: Any = NEUTRAL_COLOR
    font_size: float = 10.0
    title_size: float = 15.0
    stage_title_size: float = 11.0
    annotation_size: float = 11.0
    label_pad: float = 7.0
    label_wrap: int | None = None
    label_template: str = "{label} ({percent})"
    node_width: float = 0.07
    gap: float = 0.02
    curve: float = 1.0
    ribbon_alpha: float = 0.45
    ribbon_color: Any = "source"
    figsize: tuple[float, float] = (9.0, 6.0)
    dpi: float | None = None
    margin_x: float = 0.12

    def __post_init__(self) -> None:
        if self.label_wrap is not None and self.label_wrap < 1:
            raise ValueError(f"label_wrap must be >= 1 or None, got {self.label_wrap!r}")
        if not 0 < self.node_width < 1:
            raise ValueError(f"node_width must be in (0, 1), got {self.node_width!r}")
        if self.gap < 0:
            raise ValueError(f"gap must be >= 0, got {self.gap!r}")
        if not 0 <= self.curve <= 1:
            raise ValueError(f"curve must be in [0, 1], got {self.curve!r}")
        if not 0 <= self.margin_x < 0.4:
            raise ValueError(f"margin_x must be in [0, 0.4), got {self.margin_x!r}")

    def replace(self, **changes: Any) -> Style:
        """Return a copy of this style with ``changes`` applied."""
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            raise TypeError(
                f"unknown Style option(s) {sorted(unknown)}; valid options are "
                f"{sorted(f.name for f in fields(self))}"
            )
        return dataclasses.replace(self, **changes) if changes else self


def resolve_colors(
    labels: Sequence[Any],
    palette: Palette = None,
    *,
    neutral_color: Any = NEUTRAL_COLOR,
    neutral_labels: Sequence[Any] = (),
) -> dict[Any, str]:
    """Map every label to a hex colour.

    ``labels`` is the drawing order shared by every stage, so a category keeps
    the same colour wherever it appears. Labels in ``neutral_labels`` get
    ``neutral_color`` unless the palette names them explicitly.
    """
    explicit: dict[Any, Any] = {}
    cycle: Sequence[Any] = DEFAULT_PALETTE

    if palette is None:
        pass
    elif isinstance(palette, Mapping):
        explicit = dict(palette)
    elif isinstance(palette, str):
        cycle = _colormap_colors(palette, len(labels))
    elif isinstance(palette, Sequence):
        if not len(palette):
            raise ValueError("palette sequence is empty")
        cycle = list(palette)
    else:
        raise TypeError(
            "palette must be a mapping, a sequence of colours, a colormap name or None, "
            f"got {palette!r}"
        )

    neutral = set(neutral_labels)
    colors: dict[Any, str] = {}
    for label in labels:
        if label in explicit:
            colors[label] = _to_hex(explicit[label], label)
        elif label in neutral:
            colors[label] = to_hex(neutral_color)
    # Hand out palette colours in label order, skipping the ones already set.
    taken = 0
    for label in labels:
        if label not in colors:
            colors[label] = to_hex(cycle[taken % len(cycle)])
            taken += 1
    return {label: colors[label] for label in labels}


def _to_hex(color: Any, label: Any) -> str:
    try:
        return to_hex(color)
    except (ValueError, TypeError):
        raise ValueError(f"invalid colour {color!r} for label {label!r}") from None


def _colormap_colors(name: str, n: int) -> list[str]:
    try:
        cmap = mpl.colormaps[name]
    except KeyError:
        raise ValueError(
            f"{name!r} is not a matplotlib colormap; pass a mapping or a sequence "
            "of colours instead"
        ) from None
    if isinstance(cmap, ListedColormap) and cmap.N <= 20:
        return [to_hex(c) for c in cmap.colors]
    return [to_hex(c) for c in cmap(np.linspace(0.1, 0.9, max(n, 1)))]
