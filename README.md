# cvplt
cvplt is a small collection of python functions to quickly and easily plot a Numpy Array to a Numpy Image Array (Rows x Columns x Colour[B,G,R]), for the purpose of plotting via opencv-python (i.e., cv2) imshow().

Since drawing a plot with opencv-python is computationally quicker than plotting traditionally with matplotlib, cvplt solutions may be preferred for quick-and-dirty plotting of dynamic data in real-time (at only a moderate cost to aesthetics :)).

Currently supports the following plotting methods:
- Overlay a Numpy 1D Array to a Numpy Image Array, with undefined X-axis.
- Overlay a Numpy Coordinates Array (e.g., [[0,0],[1,1],[9,10]]) to a Numpy Image Array.

Notes:
- How are np.Nan's handled?
	- Numpy 1D Array inputs can include np.Nan's - they are simply omitted from plots.
	- Numpy Coordinate Array inputs should not include any np.Nan's.
- Singular data points are plotted as points (i.e., cv2.circle() function).
- Pairs of data points are plotted as lines (i.e., cv2.line() function).

![depiction 001](https://github.com/benfpv/cvplt/assets/55154673/b530c88e-9a92-4d31-a2aa-99e7ac4c821c)

# Repository Contents
1. demo.py
2. cvplt.py
3. LICENSE
4. README.md

# Requirements
1. python 3.12.10 (probably also works on other versions)
2. numpy
3. cv2 (opencv-python)

# Instructions to Run the Demo
1. Run `demo.py` with Python.
2. See `demo.py` > `Main.loop()` for the inputs used to create the demo plots.
3. Expected outcome: The demo plots should be plotted onto a 640x480 window.
4. Exit the demo by pressing `Ctrl+C` in the command prompt or terminal.

![dispArray_Resize](https://github.com/benfpv/cvplt/assets/55154673/5c392636-13fb-45b8-88a1-12eb04732261)

# Instructions for Custom Use in Your Own Projects
1. Ensure your project environment meets the requirements above.
2. Add `cvplt.py` to your project directory.
3. Import it: `from cvplt import *`
4. Call the public functions listed below.  All other functions are private (prefixed `_`) and are not intended for direct use.

# Functions
1. draw_plot(data, renderArray=None, plotBeginXY=None, plotEndXY=None, plotTitle="", plotBackgroundColour=[2,2,2], plotOutlineColour=[250,250,250], plotValuesColour=[250,250,250])
	- Required:  
		- data: Numpy 1D array (e.g., [0, np.nan, 5, 2.2]) you wish to plot. Must be length > 0.
	- Optional:
		- renderArray: Numpy image array (# of Rows, # of Columns, Colour(B,G,R)), a BGR image which you want to add your plot to. If none provided, defaults to 640x480.
  		- plotBeginXY: Numpy array or List (X, Y), XY coordinates of beginning or top-left of plot (inclusive). Defaults to fit the renderArray.
  		- plotEndXY: Numpy array or List (X, Y), XY coordinates of end or bottom-right of plot (inclusive). Defaults to fit the renderArray.
  		- plotTitle: String, title of the plot you wish to be presented in the top-middle of the plot. Defaults to "".
  		- plotBackgroundColour: Numpy array or List (B,G,R), BGR colour that you want the plot background to be. Defaults to black (e.g., [2,2,2]).
  		- plotOutlineColour: Numpy array or List (B,G,R), BGR colour that you want the plot outline and text to be. Defaults to white (e.g., [250,250,250]).
  		- plotValuesColour: Numpy array or List (B,G,R), BGR colour that you want the plotted data values to be. Defaults to white (e.g., [250,250,250]).

2. draw_plot_coords(data, renderArray=None, connectDots=False, plotBeginXY=None, plotEndXY=None, plotTitle="", plotBackgroundColour=[2,2,2], plotOutlineColour=[250,250,250], plotValuesColour=[250,250,250])
	- Required:  
		- data: Numpy coordinates array (e.g., [[0,0],[2,2] ... [X,Y]]) you wish to plot (each coordinate == one point). Must be length > 0.
	- Optional:
  		- renderArray: Numpy array (# of Rows, # of Columns, Colour(B,G,R)), a BGR image which you want to add your plot to. If none provided, defaults to 640x480.
  		- connectDots: Boolean, whether to connect the dots with lines or not. *** Currently this is not yet working ***.
		- plotBeginXY: Numpy array or List (X, Y), XY coordinates of beginning or top-left of plot (inclusive). Defaults to fit the renderArray.
  		- plotEndXY: Numpy array or List (X, Y), XY coordinates of end or bottom-right of plot (inclusive). Defaults to fit the renderArray.
  		- plotTitle: String, title of the plot you wish to be presented in the top-middle of the plot. Defaults to "".
  		- plotBackgroundColour: Numpy array or List (B,G,R), BGR colour that you want the plot background to be. Defaults to black (e.g., [2,2,2]).
  		- plotOutlineColour: Numpy array or List (B,G,R), BGR colour that you want the plot outline and text to be. Defaults to white (e.g., [250,250,250]).
  		- plotValuesColour: Numpy array or List (B,G,R), BGR colour that you want the plotted data values to be. Defaults to white (e.g., [250,250,250]).

# Known Limitations & Future Directions
- Update README.md assets (screenshots).
- Improve sizing heuristics for points and lines so they look coherent across different data densities (e.g., smaller points when many coordinates are grouped together).
- Implement `connectDots` logic in `draw_plot_coords()` to optionally draw lines between coordinate points.
- Add a method to visualise 2-D neural networks.