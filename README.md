# cvplt

Lightweight NumPy-to-OpenCV plotting. Composite line plots and scatter plots directly onto BGR image arrays for real-time display with `cv2.imshow()` -- faster than matplotlib, at a moderate cost to aesthetics.

![depiction 001](https://github.com/benfpv/cvplt/assets/55154673/b530c88e-9a92-4d31-a2aa-99e7ac4c821c)

## Features

- **1-D line plot** -- overlay a NumPy 1-D array onto any BGR or grayscale image.
- **2-D scatter plot** -- overlay an Nx2 coordinate array as filled circles.
- NaN-safe: `np.nan` values in 1-D data are simply skipped.
- Single data points are drawn as dots; adjacent pairs as lines.
- Auto-creates a 640x480 canvas when no `renderArray` is supplied.

## Requirements

- Python >= 3.10
- numpy >= 1.26
- opencv-python >= 4.11

Install with:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from cvplt import cvplt

# Create a 1-D plot on an auto-sized canvas
data = np.random.rand(200) * 100
img = cvplt.draw_plot(data, plotTitle="My Plot")

# Composite a scatter plot onto an existing image
coords = np.array([[10, 20], [30, 40], [50, 10]], dtype="int")
img = cvplt.draw_plot_coords(coords, img, plotBeginXY=[100, 100], plotEndXY=[300, 250])

import cv2
cv2.imshow("result", img)
cv2.waitKey(0)
```

## Demo

```bash
python demo.py
```

Overlays seven example plots onto a 640x480 window and pauses for 30 seconds. Press `Ctrl+C` to exit early.

![dispArray_Resize](https://github.com/benfpv/cvplt/assets/55154673/5c392636-13fb-45b8-88a1-12eb04732261)

## API Reference

All functions are static methods on the `cvplt` class. Functions prefixed with `_` are private helpers and not intended for direct use.

### `cvplt.draw_plot(data, ...)`

Overlay a 1-D data series onto a BGR render array.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `np.ndarray` | *(required)* | 1-D array of values. May contain `np.nan`. |
| `renderArray` | `np.ndarray` | `None` | Destination BGR/grayscale image. Auto-created (min 640x480) if `None`. |
| `plotBeginXY` | `list` | `None` | Top-left `[x, y]` of the plot region. Defaults to full image. |
| `plotEndXY` | `list` | `None` | Bottom-right `[x, y]` of the plot region. Defaults to full image. |
| `plotTitle` | `str` | `""` | Label drawn beside the max value. |
| `plotBackgroundColour` | `list\|int` | `[2,2,2]` | BGR fill colour for the plot area. |
| `plotOutlineColour` | `list\|int` | `[250,250,250]` | BGR border and text colour. |
| `plotValuesColour` | `list\|int` | `[250,250,250]` | BGR colour for data lines/dots. |

**Returns:** `np.ndarray` -- the render array with the plot composited.

**Raises:** `TypeError` if `data` is not a numpy array; `ValueError` if `data` is not 1-D.

### `cvplt.draw_plot_coords(data, ...)`

Overlay 2-D coordinate data as a scatter plot.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `np.ndarray` | *(required)* | Nx2 integer array of `[x, y]` coordinates. No NaNs. |
| `renderArray` | `np.ndarray` | `None` | Destination image. Auto-created (min 640x480) if `None`. |
| `connectDots` | `bool` | `False` | Connect consecutive points with lines. **Not yet implemented** -- raises `NotImplementedError`. |
| `plotBeginXY` | `list` | `None` | Top-left `[x, y]`. |
| `plotEndXY` | `list` | `None` | Bottom-right `[x, y]`. |
| `plotTitle` | `str` | `""` | Label drawn beside the coordinate range. |
| `plotBackgroundColour` | `list\|int` | `[2,2,2]` | BGR fill colour. |
| `plotOutlineColour` | `list\|int` | `[250,250,250]` | BGR border/text colour. |
| `plotValuesColour` | `list\|int` | `[250,250,250]` | BGR dot colour. |

**Returns:** `np.ndarray` -- the render array with the plot composited.

**Raises:** `TypeError` if `data` is not a numpy array; `ValueError` if `data` is not shape (N, 2); `NotImplementedError` if `connectDots=True`.

## Repository Contents

| File | Purpose |
|---|---|
| `cvplt.py` | Core library |
| `demo.py` | Visual demo |
| `tests/test_cvplt.py` | Unit tests |
| `requirements.txt` | Runtime dependencies |
| `pyproject.toml` | Package metadata |
| `LICENSE` | MIT License |

## License

MIT -- see [LICENSE](LICENSE).

## Known Limitations & Future Directions

- Improve sizing heuristics for points/lines across different data densities.
- Implement `connectDots` in `draw_plot_coords()`.
- Add a method to visualise 2-D neural networks.