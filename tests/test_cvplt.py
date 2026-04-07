"""Unit tests for cvplt — covers edge cases and input validation."""
import numpy as np
import pytest
from cvplt import cvplt


# ---------- draw_plot ----------

class TestDrawPlot:
    def test_basic_returns_array(self):
        data = np.array([1.0, 2.0, 3.0])
        result = cvplt.draw_plot(data)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3  # BGR

    def test_with_nans(self):
        data = np.array([1.0, np.nan, 3.0, np.nan])
        result = cvplt.draw_plot(data)
        assert isinstance(result, np.ndarray)

    def test_all_nan_returns_default_canvas(self):
        data = np.array([np.nan, np.nan])
        result = cvplt.draw_plot(data)
        assert result.shape == (480, 640, 3)

    def test_empty_data_returns_default_canvas(self):
        data = np.array([], dtype="float64")
        result = cvplt.draw_plot(data)
        assert result.shape == (480, 640, 3)

    def test_single_point(self):
        data = np.array([5.5])
        result = cvplt.draw_plot(data)
        assert isinstance(result, np.ndarray)

    def test_onto_existing_render_array(self):
        canvas = np.zeros((200, 300, 3), dtype="uint8")
        data = np.array([10.0, 20.0, 30.0])
        result = cvplt.draw_plot(data, canvas, plotBeginXY=[10, 10], plotEndXY=[290, 190])
        assert result.shape == (200, 300, 3)

    def test_grayscale_render_array(self):
        canvas = np.zeros((200, 300), dtype="uint8")
        data = np.array([10.0, 20.0, 30.0])
        result = cvplt.draw_plot(data, canvas, plotBeginXY=[10, 10], plotEndXY=[290, 190])
        assert result.ndim == 2

    def test_zero_size_region_returns_unchanged(self):
        canvas = np.zeros((100, 100, 3), dtype="uint8")
        data = np.array([1.0, 2.0])
        result = cvplt.draw_plot(data, canvas, plotBeginXY=[50, 50], plotEndXY=[50, 50])
        assert np.array_equal(result, canvas)

    def test_rejects_non_array(self):
        with pytest.raises(TypeError):
            cvplt.draw_plot([1, 2, 3])

    def test_rejects_2d_array(self):
        with pytest.raises(ValueError):
            cvplt.draw_plot(np.array([[1, 2], [3, 4]]))

    def test_constant_data(self):
        """All identical values should not crash (dataRange = 0)."""
        data = np.array([5.0, 5.0, 5.0, 5.0])
        result = cvplt.draw_plot(data)
        assert isinstance(result, np.ndarray)


# ---------- draw_plot_coords ----------

class TestDrawPlotCoords:
    def test_basic_returns_array(self):
        data = np.array([[0, 0], [10, 10], [20, 5]], dtype="int")
        result = cvplt.draw_plot_coords(data)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3

    def test_empty_data_returns_default_canvas(self):
        data = np.empty((0, 2), dtype="int")
        result = cvplt.draw_plot_coords(data)
        assert result.shape == (480, 640, 3)

    def test_single_coordinate(self):
        data = np.array([[5, 5]], dtype="int")
        result = cvplt.draw_plot_coords(data)
        assert isinstance(result, np.ndarray)

    def test_identical_coordinates(self):
        """All same coords should not cause div/0."""
        data = np.array([[3, 3], [3, 3], [3, 3]], dtype="int")
        result = cvplt.draw_plot_coords(data)
        assert isinstance(result, np.ndarray)

    def test_onto_existing_render_array(self):
        canvas = np.zeros((200, 300, 3), dtype="uint8")
        data = np.array([[0, 0], [10, 10]], dtype="int")
        result = cvplt.draw_plot_coords(data, canvas, plotBeginXY=[10, 10], plotEndXY=[290, 190])
        assert result.shape == (200, 300, 3)

    def test_rejects_non_array(self):
        with pytest.raises(TypeError):
            cvplt.draw_plot_coords([[0, 0], [1, 1]])

    def test_rejects_1d_array(self):
        with pytest.raises(ValueError):
            cvplt.draw_plot_coords(np.array([1, 2, 3]))

    def test_rejects_wrong_columns(self):
        with pytest.raises(ValueError):
            cvplt.draw_plot_coords(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_connect_dots_raises(self):
        data = np.array([[0, 0], [1, 1]], dtype="int")
        with pytest.raises(NotImplementedError):
            cvplt.draw_plot_coords(data, connectDots=True)
