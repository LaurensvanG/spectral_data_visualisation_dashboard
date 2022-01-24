### Imports
# Native Python
import re
from functools import partial

# Data storage & processing
import numpy as np
import xarray as xr
from PyMca5.PyMca.FsmReader import parseFile

# Embeddings
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Plotting
import panel as pn
css = '''
.embedding-button {
  border-radius: 5px;
  padding: 3px !important;
  background-color: #ccc;
}
'''
pn.config.raw_css=[css]

import param
import holoviews as hv
from bokeh.themes.theme import Theme
from holoviews.operation.datashader import dynspread, rasterize

# Set holoviews width (%) and backend
hv.extension("bokeh", width=100)
hv.Store.set_current_backend("bokeh")
pn.extension(sizing_mode="stretch_width", comms="vscode")

# Set Bokeh theme and default holoviews *Curve* options
theme = Theme(
    json={
        'attrs' : {
            'Figure' : {
                'background_fill_color': '#f0f0f0',
                'border_fill_color': '#f0f0f0',
            }
        }
    }
)

hv.renderer('bokeh').theme = theme
hv.opts.defaults(
    hv.opts.Curve(tools=["hover"], line_width=.75),
)

# Dashboard
class Opts(param.Parameterized):
    
    xlabel = param.String()
    ylabel = param.String()
    
    xlim = param.NumericTuple()
    ylim = param.NumericTuple()
    
    title = param.String()
    aspect = param.Parameter()
    show_legend = param.Boolean()

    
class DataContainer(param.Parameterized):
    """Container for all the data that is used across the dashboard"""
    
    # Constants
    ONE_WIDGET_WIDTH = 380
    TWO_WIDGET_WIDTH = 150
    FOUR_WIDGET_WIDTH = 65

    # Data import
    filename = param.Filename()
    spectral_data = param.Array()    
    band_min = param.Number()
    band_max = param.Number()
    band_change = param.Number()
    wavenumbers = param.Array()
    
    raw_data = param.Array()
    all_data = param.Array()
    sel_data = param.Array()
    all_norm_data = param.Array()
    
    do_trans_to_abs = param.Boolean(False)
    do_baseline_shift = param.Boolean(False)
    do_normalisation = param.Boolean(False)
    shifting_band = param.Integer(-1)
    shifting_range = param.Integer(100, bounds=(1, None))
    normalisation_band = param.Integer(-1)
    normalisation_range = param.Integer(5, bounds=(1, None))
    shifting_band_selector = pn.widgets.IntInput(name="Shifting band", max_width=TWO_WIDGET_WIDTH)
    normalisation_band_selector = pn.widgets.IntInput(name="Normalisation band", max_width=TWO_WIDGET_WIDTH)
    
    # Region selection    
    x_min = param.Integer(-1)
    x_max = param.Integer(-1)
    y_min = param.Integer(-1)
    y_max = param.Integer(-1)
    sample_selection = hv.link_selections.instance(unselected_alpha=.5, cross_filter_mode="overwrite")
    
    # Plotting
    band_selector = pn.widgets.IntInput(name="Plotting band", max_width=TWO_WIDGET_WIDTH, value=-1)
    band = param.Selector(default=-1)
    band_index = param.Integer()
    quantile = param.Number(.99, bounds=(.5, 1))
    plotting_cols = param.Integer(2, bounds=(2, None))
    
    # Embeddings
    do_pca = param.Boolean()
    do_tsne = param.Boolean()
    do_umap = param.Boolean()
    
    pca_data = param.Dict()
    tsne_data = param.Array()
    umap_data = param.Array()
    no_pca_comps = param.Integer(10, bounds=(1, None))
    
    pcx = param.Integer(1, bounds=(1, 10))
    pcy = param.Integer(2, bounds=(1, 10))
    
    EMPTY_DATA_2D = [[np.NaN, np.NaN], [np.NaN, np.NaN]]
    EMPTY_DATA_3D = [[[np.NaN, np.NaN], [np.NaN, np.NaN]], [[np.NaN, np.NaN], [np.NaN, np.NaN]]]
    exrds = xr.Dataset()
    eds = hv.Dataset([])
    
    current_sel = param.Array()

    calculate_button = pn.widgets.Button(name="Calculate embeddings", button_type="primary", max_width=TWO_WIDGET_WIDTH)
    calculate_button_clicks = param.Integer()

    # Template
    template = pn.template.FastListTemplate(
        title="Spectral visualisation dashboard", 
        header_background="#1B264F",
        accent_base_color="#51617C",
        neutral_color="#fff",
        font='"Open Sans", verdana, arial, sans-serif',
        sidebar_width=350,
        meta_author="Laurens van Giersbergen",
    )

    
class DataProcessor(param.Parameterized):
    """Data importing and basic processing"""

    def __init__(self, datacontainer):
        self.data = datacontainer
        super().__init__()  
    
    @param.depends("data.filename", watch=True)
    def import_data(self):
        """Import data from the specified file, when the selected file changes"""
        spectral_info, spectral_data = parseFile(self.data.filename)

        # Replace NaNs with the lowest non-NaN value
        np.nan_to_num(spectral_data, copy=False, nan=np.nanmin(spectral_data))
        
        self.data.band_min = int(min( abs(spectral_info["zStart"]), abs(spectral_info["zStart"] + spectral_info["zDelta"] * (spectral_info["zLength"] - 1)) ))
        self.data.band_max = int(max( abs(spectral_info["zStart"]), abs(spectral_info["zStart"] + spectral_info["zDelta"] * (spectral_info["zLength"] - 1)) ))
        self.data.band_change = int(spectral_info["zDelta"])

        self.data.wavenumbers = np.arange(self.data.band_max, self.data.band_min+self.data.band_change, self.data.band_change)
        self.data.param.band.objects = self.data.wavenumbers
        self.data.band = self.data.param.band.objects[-1]
        
        self.data.spectral_data = spectral_data
                
    @param.depends("data.spectral_data", "data.do_trans_to_abs", watch=True)
    def transform_data(self):
        """Transform transmission into absorbance"""
        if self.data.do_trans_to_abs:
            # Transform transmission into absorbance
            raw_data = -np.log10(self.data.spectral_data / 100)
            
            # Remove NaN values from logarithm of negative values
            np.nan_to_num(raw_data, copy=False, nan=np.nanmin(raw_data))

            self.data.raw_data = raw_data
        
        else:
            self.data.raw_data = self.data.spectral_data
            
    @param.depends("data.raw_data", "data.do_baseline_shift", "data.shifting_band", "data.shifting_range", watch=True)
    def baseline_shift(self):
        """Carry out a baseline shift"""
        if self.data.do_baseline_shift:
            shift_band_1 = self._get_band(self.data.shifting_band - self.data.shifting_range)
            shift_band_2 = self._get_band(self.data.shifting_band + self.data.shifting_range)
            shift_band_low = min(shift_band_1, shift_band_2)
            shift_band_high = max(shift_band_1, shift_band_2)

            shift_abs = self.data.raw_data[..., np.arange(shift_band_low, shift_band_high)].mean(-1)
            self.data.all_data = self.data.raw_data - np.repeat(shift_abs[..., np.newaxis], self.data.raw_data.shape[-1], axis=-1)
            
        else:
            self.data.all_data = self.data.raw_data
        
    @param.depends("data.all_data", "data.do_normalisation", "data.normalisation_band", "data.normalisation_range", watch=True)
    def check_normalisation(self):
        """Check whether the data should be normalised, if so: normalise"""
        
        # Do nothing if there is no data yet
        if self.data.all_data is not None:
            if self.data.do_normalisation:
                # Normalise the data
                norm_band_1 = self._get_band(self.data.normalisation_band - self.data.normalisation_range)
                norm_band_2 = self._get_band(self.data.normalisation_band + self.data.normalisation_range)
                norm_band_low = min(norm_band_1, norm_band_2)
                norm_band_high = max(norm_band_1, norm_band_2)
                
                # Only calculate the normalisation if it does not exist yet or 
                # is not in the same shape as the unnormalised data (i.e. the 
                # imported data has changed)
                if self.data.all_norm_data == None or self.data.all_norm_data.shape != self.data.all_data.shape:
                    all_norm_data = self._normalise(self.data.all_data, norm_band_low, norm_band_high)
                
                # Save the normalised data
                self.data.exrds = xr.Dataset(
                    data_vars = dict(absorbance=(["y", "x", "band"], all_norm_data)),
                    coords = dict(x=np.arange(all_norm_data.shape[1]), y=np.arange(all_norm_data.shape[0]), band=self.data.wavenumbers),
                ).stack(idx=("x", "y"))
                self.data.all_norm_data = all_norm_data
            else:
                # Save the non-normalised data and set the normalised data 
                # container to None
                self.data.exrds = xr.Dataset(
                    data_vars = dict(absorbance=(["y", "x", "band"], self.data.all_data)),
                    coords = dict(x=np.arange(self.data.all_data.shape[1]), y=np.arange(self.data.all_data.shape[0]), band=self.data.wavenumbers),
                ).stack(idx=("x", "y"))
                self.data.all_norm_data = None

            # If no sample selection is made, select the entire sample for analysis
            if "absorbance" in self.data.exrds and self.data.sample_selection.selection_expr is None:
                self.get_data_bounds()
            
            self.data.eds = hv.Dataset(self.data.exrds, ["band", "idx"])
                
    def _normalise(self, all_data, norm_band_low, norm_band_high):
        """Normalise selected data using the provided bands"""
        # Median absorbtion (over the mean of the selected norm range) to multiply with at the end to approximately maintain the original scaling
        median_mean_abs = np.median(all_data[..., np.arange(norm_band_low, norm_band_high)].mean(-1))
        
        # Mean absorbtion of the selected norm range for every data point, copied to match the data shape
        mean_abs = np.repeat(self.data.all_data[..., np.arange(norm_band_low, norm_band_high)].mean(-1)[..., np.newaxis], 
                            self.data.all_data.shape[-1], axis=-1)
        
        return np.divide(self.data.all_data, mean_abs) * median_mean_abs
        
    @param.depends("data.band_selector.value", watch=True)
    def update_band(self):
        """Update band parameter when selected plotting band changes"""
        if self.data.band_selector.value != -1:
            self.set_closest_band(self.data.band_selector.value)
    
    @param.depends("data.band", watch=True)
    def update_band_selector(self):
        """
        If the band changes by means other than a the band_selector,
        update the band_selector's band value.
        """
        if self.data.band_selector.value != self.data.band:
            self.data.band_selector.value = int(self.data.band)

    @param.depends("data.shifting_band_selector.value", watch=True)
    def update_shifting_band(self):
        self.data.shifting_band = self.data.shifting_band_selector.value
    
    @param.depends("data.normalisation_band_selector.value", watch=True)
    def update_normalisation_band(self):
        self.data.normalisation_band = self.data.normalisation_band_selector.value
    
    @param.depends("data.band", watch=True)
    def calculate_band_index(self):
        """Transform the selected band in wavenumbers into the corresponding index of the"""
        self.data.band_index = self._get_band(self.data.band)
    
    @param.depends("data.spectral_data", "data.shifting_range", "data.normalisation_range", watch=True)
    def update_band_selectors(self):
        self.data.band_selector.start = self.data.band_min
        self.data.band_selector.stop = self.data.band_max
        self.data.band_selector.step = abs(self.data.band_change)
        
        self.data.shifting_band_selector.start = self.data.band_min + self.data.shifting_range + abs(self.data.band_change)
        self.data.shifting_band_selector.stop = self.data.band_max - self.data.shifting_range
        self.data.shifting_band_selector.step = abs(self.data.band_change)
        
        self.data.normalisation_band_selector.start = self.data.band_min + self.data.normalisation_range + abs(self.data.band_change)
        self.data.normalisation_band_selector.stop = self.data.band_max - self.data.normalisation_range
        self.data.normalisation_band_selector.step = abs(self.data.band_change) 
    
    @param.depends("data.no_pca_comps", watch=True)
    def update_max_pca_component(self):
        self.data.param.pcx.bounds = (1, self.data.no_pca_comps)
        self.data.param.pcy.bounds = (1, self.data.no_pca_comps)
    
    def filename_select(self):
        return pn.widgets.TextInput.from_param(self.data.param.filename, max_width=self.data.ONE_WIDGET_WIDTH)
    
    def sidebar(self):
        plotting_information = pn.pane.Markdown("""
        ### Plotting options
        """, margin=0)
        processing_information = pn.pane.Markdown("""
        ### Processing options
        """, margin=0)

        plot_cols_selector = pn.widgets.IntInput.from_param(self.data.param.plotting_cols, name="No. of plots per row", max_width=self.data.ONE_WIDGET_WIDTH)
        ta_toggle = pn.widgets.Checkbox.from_param(self.data.param.do_trans_to_abs, name="Transmission to absorbance", max_width=self.data.TWO_WIDGET_WIDTH)
        baseline_toggle = pn.widgets.Checkbox.from_param(self.data.param.do_baseline_shift, name="Baseline shift", max_width=self.data.TWO_WIDGET_WIDTH)
        normalisation_toggle = pn.widgets.Checkbox.from_param(self.data.param.do_normalisation, name="Normalise", max_width=self.data.TWO_WIDGET_WIDTH)
        shift_range = pn.widgets.IntInput.from_param(self.data.param.shifting_range, max_width=self.data.TWO_WIDGET_WIDTH)
        norm_range = pn.widgets.IntInput.from_param(self.data.param.normalisation_range, max_width=self.data.TWO_WIDGET_WIDTH)
        quantile_selector = pn.widgets.FloatInput.from_param(self.data.param.quantile, max_width=self.data.TWO_WIDGET_WIDTH)

        return pn.Column(
            pn.layout.Divider(),
            plotting_information,
            plot_cols_selector,
            pn.Row(self.data.band_selector, quantile_selector),
            pn.layout.Divider(),
            processing_information,
            ta_toggle,
            pn.Row(
                pn.Column(baseline_toggle, self.data.shifting_band_selector, shift_range),
                pn.Column(normalisation_toggle, self.data.normalisation_band_selector, norm_range),
            ),
            visible=False,
        )
    
    # General utility functions
    def get_coordinates(self, data):
        """
        Create a coordinate space from the provided data.
        
        Args:
            data (np.array): spectral data.
            
        Returns:
            x (list): x coordinates
            y (list): y coordinates
        """
        x = [x % data.shape[1] for x in range(np.product(data.shape[:2]))]
        y = [int(np.floor(y / data.shape[1])) for y in range(np.product(data.shape[:2]))]
        
        return x, y
    
    def get_data_bounds(self):
        self.data.x_min, self.data.x_max, self.data.y_min, self.data.y_max = (
            self.data.exrds.x.min().item(), self.data.exrds.x.max().item(), 
            self.data.exrds.y.min().item(), self.data.exrds.y.max().item()
        )

    def calculate_quantile(self, data, quantile=.99):
        """
        Clip the data using a supplied upper quantile and calculated lower 
            quantile (1 - upper quantile).

        Args:
            data (np.array): spectral data.
            quantile (float): cutoff for the upper quantile (lower than 1)

        Returns:
            (np.array): the clipped spectral data.
        """
        lower_q = np.nanquantile(data, 1 - quantile)
        upper_q = np.nanquantile(data, quantile)

        return np.clip(data, lower_q, upper_q)
    
    def calculate_square_plot_limits(self, x_min, x_max, y_min, y_max):
        """
        Calculate plot ranges given the x and y length of the data, so that the 
            ranges create a plot with square pixels.

        Args:
            x_length (number): length of the data in the x direction
            y_length (number): length of the data in the y direction

        Returns:
            x_min (float): the lower x bounds
            x_max (float): the upper x bounds
            x_min (float): the lower y bounds
            y_max (float): the upper y bounds
        """
        x_length = abs(x_max - x_min)
        y_length = abs(y_max - y_min)
        axis_length = max(x_length, y_length)
        x_min_calc = -axis_length/2 + x_length/2 - 0.5 + x_min
        x_max_calc =  axis_length/2 + x_length/2 + 0.5 + x_min
        y_min_calc = -axis_length/2 + y_length/2 - 0.5 + y_min
        y_max_calc =  axis_length/2 + y_length/2 + 0.5 + y_min
        
        return x_min_calc, x_max_calc, y_min_calc, y_max_calc

    def set_closest_band(self, band):
        """Get the closest band in the data"""
        objs = self.data.param.band.objects
        closest_band = objs[np.abs(objs - band).argmin()]
        self.data.band = closest_band

    def _get_band(self, band):
        """
        Calculate the band index from the supplied band in wavenumbers

        Args:
            band (number): band to find in wavenumbers

        Returns:
            (int): band index
        """
        return int( (band - self.data.band_max) / self.data.band_change )
    

class RegionSelection(param.Parameterized):
    sel_plot_opts = Opts(xlim=(0, 10), ylim=(0, 10))
    
    EMPTY_MAP = ([0, 10], [0, 10], [[None, None], [None, None]])
    PLOT_OPTIONS = dict(framewise=True, responsive=True, min_width=200, aspect=1,
                        cmap="kbc",xlabel="", ylabel="", )
    data_pipe = hv.streams.Pipe(EMPTY_MAP)
    sel_data_pipe = hv.streams.Pipe(EMPTY_MAP)
    
    def __init__(self, datacontainer, dataprocessor):
        self.data = datacontainer
        self.dp = dataprocessor
        super().__init__()    
    
    
    def create_selection_plot(self):
        """Create plot in which you can select regions and view the selection"""
        return self.data.sample_selection(
            hv.DynamicMap(hv.Image, streams=[self.data_pipe]).opts(
                hv.opts.Image(
                    **self.PLOT_OPTIONS,
                    active_tools=["box_select"], title="Region selection",
                ),
            )
        ).redim(x="selection_x", y="selection_y")
    
    def show_selection(self):
        """Create a plot that shows the selected data"""
        return hv.DynamicMap(
            hv.Image, streams=[self.sel_data_pipe]
        ).opts(
            **self.PLOT_OPTIONS, tools=["hover"], title="Selected region", 
        ).redim(x="sel_x", y="sel_y").redim.label(sel_x="x", sel_y="y", z="Absorbance").apply.opts(
            xlim=self.sel_plot_opts.param.xlim, ylim=self.sel_plot_opts.param.ylim
        )

    @param.depends("data.all_data", "data.all_norm_data", "data.band_index", "data.quantile", watch=True)
    def update_data_pipe(self):
        """Update the data pipe for the overall plot when the data changes"""

        # Skip if there is no data yet
        if self.data.all_data is not None:
            if self.data.do_normalisation:
                data = self.dp.calculate_quantile(self.data.all_norm_data[..., self.data.band_index], self.data.quantile)
                
                self.data_pipe.send(
                    (np.arange(data.shape[0]), 
                     np.arange(data.shape[1]), 
                     data) 
                )
                
            else:
                data = self.dp.calculate_quantile(self.data.all_data[..., self.data.band_index], self.data.quantile)
                
                self.data_pipe.send(
                    (np.arange(data.shape[1]), 
                     np.arange(data.shape[0]), 
                     data) 
                )
            
    @param.depends("data.sel_data", "data.band_index", "data.quantile", watch=True)
    def update_sel_data_pipe(self):
        """Update the sample plot's data pipe, if the selected data changes"""

        self.sel_plot_opts.aspect = None
        
        if self.data.sample_selection.selection_expr is None or self.data.sel_data is None:
            self.sel_data_pipe.send(self.EMPTY_MAP)
        
        else:
            data = self.dp.calculate_quantile(self.data.sel_data[..., self.data.band_index], self.data.quantile)

            self.sel_data_pipe.send(
                (np.arange(self.data.x_min, self.data.x_max),
                 np.arange(self.data.y_min, self.data.y_max),
                 data) 
            )
            
            x_min, x_max, y_min, y_max = self.dp.calculate_square_plot_limits(
                self.data.x_min, self.data.x_max, self.data.y_min, self.data.y_max
            )
            
            self.sel_plot_opts.xlim = (x_min, x_max)
            self.sel_plot_opts.ylim = (y_min, y_max)
        self.sel_plot_opts.aspect = "equal"
            
    @param.depends("data.sample_selection.selection_expr", watch=True)
    def update_bounds(self):
        """Update selection plot bounds when the selection boundary is updated"""

        if "absorbance" in self.data.exrds:
            if self.data.sample_selection.selection_expr is None:
                self.dp.get_data_bounds()
            else:
                sel_expr = str(self.data.sample_selection.selection_expr)
                self.data.x_min = max(0, round(float(re.search(r"'x'\)>=(\-*\d+\.*\d*)", sel_expr).group(1))))
                self.data.y_min = max(0, round(float(re.search(r"'y'\)>=(\-*\d+\.*\d*)", sel_expr).group(1))))
                self.data.x_max = min(self.data.all_data.shape[1] - 1, round(float(re.search(r"'x'\)<=(\-*\d+\.*\d*)", sel_expr).group(1))))
                self.data.y_max = min(self.data.all_data.shape[0] - 1, round(float(re.search(r"'y'\)<=(\-*\d+\.*\d*)", sel_expr).group(1))))
    
    @param.depends("data.y_max", "data.all_data", "data.all_norm_data", watch=True)
    def update_selection_data(self):
        """Update the selection data when a change is made"""

        # If there is no selection, or the selection was removed
        if "absorbance" in self.data.exrds and self.data.sample_selection.selection_expr is None:
            self.dp.check_normalisation()
            self.data.sel_data = None

        else:
            if self.data.do_normalisation:
                sel_data = self.data.all_norm_data[self.data.y_min:self.data.y_max, self.data.x_min:self.data.x_max, :]
            else:
                sel_data = self.data.all_data[self.data.y_min:self.data.y_max, self.data.x_min:self.data.x_max, :]
            
            # Replace the old data with the new selection
            self.data.exrds = xr.Dataset(
                data_vars = dict(absorbance=(["y", "x", "band"], sel_data)),
                coords = dict(
                    x=np.arange(self.data.x_min, self.data.x_max), 
                    y=np.arange(self.data.y_min, self.data.y_max), 
                    band=self.data.wavenumbers
                ),
            ).stack(idx=("x", "y"))
            self.data.eds = hv.Dataset(self.data.exrds, ["band", "idx"])
            self.data.sel_data = sel_data
        
    def view(self):
        selection_plot = self.create_selection_plot()
        selected_plot = self.show_selection()
        return pn.GridBox(selection_plot, selected_plot, ncols=2)
    

class EmbeddingsProducer(param.Parameterized):
    
    pca_params = param.Dict(dict(
        whiten=False,
        svd_solver='auto',
        tol=0.0,
        iterated_power='auto',
        random_state=None,
    ))
    
    tsne_distance_metric = param.Selector({
        "Euclidean": "euclidean",
        "Pre-computed": "precomputed",
        "Cosine": "cosine",
        "Correlation": "correlation",
        "Manhattan/City block": "cityblock",
        "Haversine": "haversine",
        "Chebyshev": "chebyshev",
        "Minkowski": "minkowsky",
        "Canberra": "canberra",
        "Bray-Curtis": "braycurtis",
        "Mahalanobis": "mahalanobis",
        "Weighted Minkowsky": "wminkowsky",
        "Std. Euclidean": "seuclidean",
    })
    
    tsne_params = param.Dict(dict(
        n_components=2,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate=200.0,
        n_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-07,
        init='random',
        verbose=0,
        random_state=None,
        method='barnes_hut',
        angle=0.5,
        n_jobs=None,
        square_distances='legacy',
    ))
    
    umap_distance_metric = param.Selector({
        "Euclidean": "euclidean",
        "Cosine": "cosine",
        "Correlation": "correlation",
        "Manhattan/City block": "cityblock",
        "Haversine": "haversine",
        "Chebyshev": "chebyshev",
        "Minkowski": "minkowsky",
        "Canberra": "canberra",
        "Bray-Curtis": "braycurtis",
        "Mahalanobis": "mahalanobis",
        "Weighted Minkowsky": "wminkowsky",
        "Std. Euclidean": "seuclidean",
    })
    
    umap_params = param.Dict(dict(
        n_neighbors=15, n_components=2, metric_kwds=None, output_metric='euclidean', output_metric_kwds=None,
        n_epochs=None, learning_rate=1.0, init='spectral', min_dist=0.1, spread=1.0, low_memory=True,
        n_jobs=-1, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0, negative_sample_rate=5,
        transform_queue_size=4.0, a=None, b=None, random_state=None, angular_rp_forest=False, target_n_neighbors=-1,
        target_metric='categorical', target_metric_kwds=None, target_weight=0.5, transform_seed=42,
        transform_mode='embedding', force_approximation_algorithm=False, verbose=False, unique=False,
        densmap=False, dens_lambda=2.0, dens_frac=0.3, dens_var_shift=0.1, output_dens=False, 
        disconnection_distance=None,
    ))
    
    clear_selection_button = pn.widgets.Button(name="Clear selection")
    
    def __init__(self, datacontainer):
        self.data = datacontainer

        super().__init__()
    
    @param.depends("data.calculate_button.clicks", watch=True)
    def update_button_clicks(self):
        """
        Update the number of clicks, done indirectly to prevent unnecessary 
            updates
        """
        if self.data.calculate_button_clicks != self.data.calculate_button.clicks:
            self.data.calculate_button_clicks = self.data.calculate_button.clicks


    @param.depends("data.calculate_button_clicks", watch=True)
    def calculate_embeddings(self):
        """
        Calculate the selected embeddings, when the calculate button is pressed.
        """
        
        # Skip calculation if there is no data yet
        if len(self.data.exrds) != 0:
            # Flatten the data into a 2D matrix
            # If no region is selected, take all data
            flattened_data = self.data.exrds.absorbance.data.reshape(-1, self.data.exrds.absorbance.shape[-1]).T

            # PCA
            if self.data.do_pca :
                pca = PCA(n_components=self.data.no_pca_comps, **self.pca_params)

                pca_data = dict(
                    scores = pca.fit_transform(flattened_data),
                    loadings = pca.components_.T * np.sqrt(pca.explained_variance_),
                    expl_var = pca.explained_variance_ratio_,
                )

                # columns = [f"PC {i+1}" for i in range(pca_data["scores"].shape[1])]
                self.data.exrds = self.data.exrds.assign({
                    f"PC {i+1}": ("idx", pca_data["scores"][:, i])
                    for i in range(pca_data["scores"].shape[1])
                })
                self.data.eds = hv.Dataset(self.data.exrds, ["band", "idx"])

                self.data.pca_data = pca_data
            else:
                self.data.pca_data = None

            # t-SNE
            if self.data.do_tsne:
                reducer = TSNE(metric=self.tsne_distance_metric, **self.tsne_params)
                tsne_data = reducer.fit_transform(flattened_data)
                self.data.exrds = self.data.exrds.assign({
                    f"t-SNE component {i+1}": ("idx", tsne_data[:, i])
                    for i in range(tsne_data.shape[1])
                })
                self.data.eds = hv.Dataset(self.data.exrds, ["band", "idx"])

                self.data.tsne_data = tsne_data
            else:
                self.data.tsne_data = None

            # UMAP
            if self.data.do_umap:
                reducer = umap.UMAP(metric=self.umap_distance_metric, **self.umap_params)
                umap_data = reducer.fit_transform(flattened_data)
                self.data.exrds = self.data.exrds.assign({
                    f"UMAP component {i+1}": ("idx", umap_data[:, i])
                    for i in range(umap_data.shape[1])
                })
                self.data.eds = hv.Dataset(self.data.exrds, ["band", "idx"])

                self.data.umap_data = umap_data
            else:
                self.data.umap_data = None
    
    def create_modal_button(self):
        modal_button = pn.widgets.Button(name="Adv. options", max_width=self.data.TWO_WIDGET_WIDTH)
        modal_button.on_click(self.open_modal)
        return modal_button
    
    def open_modal(self, event):
        template = self.data.template
        template.open_modal()

    @param.depends("clear_selection_button.clicks", "data.sel_data", watch=True)
    def clear_selection(self):
        """Clear the selection if the clear_selection_button is clicked"""
        self.data.current_sel = None
    
    def sidebar(self):
        embedding_information = pn.pane.Markdown("""
        ### Embedding options
        """, margin=0)

        do_pca = pn.widgets.Checkbox.from_param(self.data.param.do_pca, name="PCA", css_classes=["embedding-button"], max_width=self.data.TWO_WIDGET_WIDTH, sizing_policy="stretch_width")
        pca_comp = pn.widgets.IntInput.from_param(self.data.param.no_pca_comps, name="No. of PCA comps.", max_width=self.data.TWO_WIDGET_WIDTH)
        pcx = pn.widgets.IntInput.from_param(self.data.param.pcx, name="PC x-axis", max_width=self.data.FOUR_WIDGET_WIDTH)
        pcy = pn.widgets.IntInput.from_param(self.data.param.pcy, name="PC y-axis", max_width=self.data.FOUR_WIDGET_WIDTH)
        
        do_tsne = pn.widgets.Checkbox.from_param(self.data.param.do_tsne, name="t-SNE", css_classes=["embedding-button"], max_width=self.data.TWO_WIDGET_WIDTH)
        tsne_dist = pn.widgets.Select.from_param(self.param.tsne_distance_metric, name="t-SNE distance metric", max_width=self.data.TWO_WIDGET_WIDTH)
        
        do_umap = pn.widgets.Checkbox.from_param(self.data.param.do_umap, name="UMAP", css_classes=["embedding-button"], max_width=self.data.TWO_WIDGET_WIDTH)
        umap_dist = pn.widgets.Select.from_param(self.param.umap_distance_metric, name="UMAP distance metric", max_width=self.data.TWO_WIDGET_WIDTH)
        
        calculate_button = self.data.calculate_button
        modal_button = self.create_modal_button

        return pn.Column(
            pn.layout.Divider(),
            embedding_information,
            pn.Row(
                pn.Column(do_umap, umap_dist,),
                pn.Column(do_tsne, tsne_dist),
            ),
            pn.Row(
                pn.Column(do_pca, pca_comp, pn.Row(pcx, pcy)),
            ),
            pn.Row(
                modal_button,
                calculate_button,
            ),
            self.clear_selection_button,
            pn.layout.Divider(),
            visible=False,
        )
    
    def modal(self):
        modal_information2 = pn.pane.HTML("""
        <h3>Advanced embedding options</h3>
        For more information see:
        <ul>
            <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html" target="_blank" rel="noopener">PCA</a></li>
            <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html?highlight=t%20sne#sklearn.manifold.TSNE" target="_blank" rel="noopener">t-SNE</a></li>
            <li><a href="https://umap-learn.readthedocs.io/en/latest/api.html" target="_blank" rel="noopener">UMAP</a></li>
        </ul>
        """)

        pca_params = pn.widgets.LiteralInput.from_param(self.param.pca_params, name="Advanced PCA paramterers (dict)", )
        tsne_params = pn.widgets.LiteralInput.from_param(self.param.tsne_params, name="Advanced t-SNE paramterers (dict)", )
        umap_params = pn.widgets.LiteralInput.from_param(self.param.umap_params, name="Advanced UMAP paramterers (dict)", )
        
        return pn.Column(
            modal_information2,
            pca_params,
            tsne_params,
            umap_params,
        )


class ExplorationDashboard(param.Parameterized):
    view_objects = param.Dict({"sample": None})
    
    pca_sel_pipe = hv.streams.Pipe([])
    pca_nonsel_pipe = hv.streams.Pipe([])
    tsne_sel_pipe = hv.streams.Pipe([])
    tsne_nonsel_pipe = hv.streams.Pipe([])
    umap_sel_pipe = hv.streams.Pipe([])
    umap_nonsel_pipe = hv.streams.Pipe([])
    sample_sel_pipe = hv.streams.Pipe([])
    sample_nonsel_pipe = hv.streams.Pipe([])
    spectrum_all_pipe = hv.streams.Pipe([])
    spectrum_sel_pipe = hv.streams.Pipe([])
    spectrum_band_pipe = hv.streams.Pipe(0)
    
    pca_opts = Opts(xlabel="PC 1", ylabel="PC 2")
    tsne_opts = Opts()
    umap_opts = Opts()
    sample_opts = Opts(title="Sample", xlim=(0,1), ylim=(0,1))
    spectrum_opts = Opts(show_legend=True)
    
    plot_options = dict(framewise=True, responsive=True, min_width=200, aspect=1)
    plot_panel_options = dict(sizing_mode="scale_both", aspect_ratio=1,)
    
    pca_selection = hv.link_selections.instance()
    tsne_selection = hv.link_selections.instance()
    umap_selection = hv.link_selections.instance()
    sample_selection = hv.link_selections.instance()

    band_tap = hv.streams.Tap(transient=True)
    prev_tapped_band = None

    def __init__(self, datacontainer, dataprocessor):
        self.data = datacontainer
        self.dp = dataprocessor
        super().__init__()   
    
    @param.depends("data.pcx", "data.pcy", watch=True)
    def update_pca_axes(self):
        """Update the PCA axes if the PCA component to be displayed is changed"""
        if self.pca_opts.xlabel != f"PC {self.data.pcx}":
            self.pca_opts.xlabel = f"PC {self.data.pcx}"
        if self.pca_opts.ylabel != f"PC {self.data.pcy}":
            self.pca_opts.ylabel = f"PC {self.data.pcy}"

    param.depends("data.pcx", "data.pcy", watch=True)
    def get_pca_plot(self, data):
        if self.pca_opts.xlabel in data:
            return hv.Scatter(data.set_coords(f"PC {self.data.pcx}"), kdims=[f"PC {self.data.pcx}"], vdims=[f"PC {self.data.pcy}"], )
        else:
            return hv.Scatter([], kdims=[f"PC {self.data.pcx}"], vdims=[f"PC {self.data.pcy}"], )
    
    def plot_pca(self):
        return self.pca_selection(
            dynspread(rasterize(hv.DynamicMap(self.get_pca_plot, streams=[self.pca_sel_pipe]))).opts(cmap="kbc_r") *
            dynspread(rasterize(hv.DynamicMap(self.get_pca_plot, streams=[self.pca_nonsel_pipe]))).opts(cmap="reds", alpha=0.5)
        ).opts(
                hv.opts.Image(**self.plot_options, active_tools=["lasso_select", "wheel_zoom"], title="PCA"),
        ).apply.opts(xlabel=self.pca_opts.param.xlabel, ylabel=self.pca_opts.param.ylabel, 
                     xlim=self.pca_opts.param.xlim, ylim=self.pca_opts.param.ylim
        )
     
    def plot_tsne(self):
        return self.tsne_selection(
            dynspread(rasterize(hv.DynamicMap(partial(hv.Scatter, kdims="t-SNE component 1", vdims="t-SNE component 2"), streams=[self.tsne_sel_pipe]))).opts(cmap="kbc_r") *
            dynspread(rasterize(hv.DynamicMap(partial(hv.Scatter, kdims="t-SNE component 1", vdims="t-SNE component 2"), streams=[self.tsne_nonsel_pipe]))).opts(cmap="reds", alpha=0.5)
        ).opts(
            hv.opts.Image(**self.plot_options, title="t-SNE embedding", active_tools=["lasso_select", "wheel_zoom"]),
        ).apply.opts(xlim=self.tsne_opts.param.xlim, ylim=self.tsne_opts.param.ylim)
    
    def plot_umap(self):
        return self.umap_selection(
            dynspread(rasterize(hv.DynamicMap(partial(hv.Scatter, kdims=["UMAP component 1"], vdims=["UMAP component 2"]), streams=[self.umap_sel_pipe]))).opts(cmap="kbc_r") *
            dynspread(rasterize(hv.DynamicMap(partial(hv.Scatter, kdims=["UMAP component 1"], vdims=["UMAP component 2"]), streams=[self.umap_nonsel_pipe]))).opts(cmap="reds", alpha=0.5)
        ).opts(
            hv.opts.Image(**self.plot_options, title="UMAP embedding", active_tools=["lasso_select", "wheel_zoom"]),
        ).apply.opts(xlim=self.umap_opts.param.xlim, ylim=self.umap_opts.param.ylim)
    
    def plot_sample_image(self):
        return self.sample_selection(
            hv.DynamicMap(partial(hv.Image, kdims=["x", "y"], vdims=["absorbance"]), streams=[self.sample_sel_pipe]).opts(cmap="kbc") *
            hv.DynamicMap(partial(hv.Image, kdims=["x", "y"], vdims=["absorbance"]), streams=[self.sample_nonsel_pipe]).opts(cmap="reds", alpha=0.2)
        ).opts(
            hv.opts.Image(**self.plot_options, clipping_colors={"NaN": "transparent"}, active_tools=["lasso_select", "wheel_zoom"]),
        ).apply.opts(
            title = self.sample_opts.param.title, aspect=self.sample_opts.param.aspect,
            xlim = self.sample_opts.param.xlim, ylim = self.sample_opts.param.ylim, 
        )

    @param.depends("data.band")
    def get_band_line(self):
        """Returns a HoloViews verticle line object to add to the spectrum plot"""
        return hv.VLine(self.data.band)

    def get_spectrum_plot_or_update_band(self, data, y, x):
        """
        Returns the spectrum plot. 
            If the spectrum plot is tapped/clicked, the x variable will have a
            value and the view band is updated with the nearest value.
        """
        
        # If a tap/click event has fired
        if x:
            # If the selected band is different from the previously tapped band
            if x != self.prev_tapped_band:
                self.prev_tapped_band = x

                # Find the closest band in the data and set it to the current
                # view band
                self.dp.set_closest_band(x)

        return hv.Curve(data, "band", "absorbance", label="All data")

    def plot_spectrum(self):
        return (
            hv.DynamicMap(partial(hv.Curve, kdims=["band"], vdims=["absorbance"], label="Selected data"), streams=[self.spectrum_sel_pipe]).redim(absorbance="Abs (sel)") *
            hv.DynamicMap(self.get_spectrum_plot_or_update_band, streams=[self.spectrum_all_pipe, self.band_tap]).redim(absorbance="Abs (all)") *
            hv.DynamicMap(self.get_band_line)
        ).redim(band="cm⁻¹").opts(
            hv.opts.Curve(**self.plot_options, title="Spectrum", tools=["vline", "tap"], ylabel="Absorbance"),
            hv.opts.VLine(**self.plot_options, line_width=.75, line_color="green")
        ).apply.opts(show_legend=self.spectrum_opts.param.show_legend)
    
    @param.depends("pca_selection.selection_expr", watch=True)
    def update_selection_from_pca(self):
        if self.pca_selection.selection_expr is not None:
            selection_coords = self.data.eds.select(self.pca_selection.selection_expr).data.idx.data.astype("i,i")
            self.update_selection_indices(selection_coords)
            
            self.pca_selection.selection_expr = None
    
    @param.depends("tsne_selection.selection_expr", watch=True)
    def update_selection_from_tsne(self):
        if self.tsne_selection.selection_expr is not None:
            selection_coords = self.data.eds.select(self.tsne_selection.selection_expr).data.idx.data.astype("i,i")
            self.update_selection_indices(selection_coords)
            
            self.tsne_selection.selection_expr = None
            
    @param.depends("umap_selection.selection_expr", watch=True)
    def update_selection_from_umap(self):
        if self.umap_selection.selection_expr is not None:
            selection_coords = self.data.eds.select(self.umap_selection.selection_expr).data.idx.data.astype("i,i")
            self.update_selection_indices(selection_coords)
            
            self.umap_selection.selection_expr = None
            
    @param.depends("sample_selection.selection_expr", watch=True)
    def update_selection_from_sample(self):
        if self.sample_selection.selection_expr is not None:
            data = (self.data.exrds.idx.data.astype("i,i"), self.data.exrds.x.data, self.data.exrds.y.data)
            ds = hv.Dataset(data, ["idx", "x", "y"])
            selection_coords = ds.select(self.sample_selection.selection_expr).data["idx"].to_numpy().astype("i,i")
            self.update_selection_indices(selection_coords)
            
            self.sample_selection.selection_expr = None
    
    def update_selection_indices(self, selection_coords):
        """
        If a selection is made, update the selection indices which allow for 
            replotting the new selection.
        """

        # Create boolean array with the current selection indices as True 
        all_idxs = self.data.exrds.idx.data.astype("i,i")
        new_sel_bool = np.isin(all_idxs, selection_coords)
        
        # If no selection has been made yet, set the selection to the current selection
        # else get the intersection between the previous and current selection
        # N.B. current_sel is in the DataContainer to prevent looping from implicit dependency
        if self.data.current_sel is None:
            updated_sel = new_sel_bool
        else:
            updated_sel = np.logical_and(self.data.current_sel, new_sel_bool)
        
        # Check if the selected indexes have changed, if so update the arrays
        # that contain the selections and update the plots
        if not np.equal(updated_sel, self.data.current_sel).all():
            self.data.current_sel = updated_sel
    
    @param.depends("data.sel_data", "data.pca_data", "data.tsne_data", "data.umap_data", "data.current_sel", "data.pcx", "data.pcy", watch=True)
    def update_embeddings_pipes(self):
        """
        Update the data pipes of the embedding plots, when one of the above 
            parameters changes.
        """

        # Get the data and drop the unnecessary variables
        data = self.data.exrds.drop(["absorbance", "band"])
        sel = data.unstack()

        # If a selection has been made split selected and non-selected data points 
        if self.data.current_sel is not None:
            sel = data[dict(idx=self.data.current_sel)].unstack()
            nonsel = data[dict(idx=~self.data.current_sel)].unstack()

    
        if self.data.do_pca and "PC 1" in data:
            pc_names = [f"PC {i+1}" for i in range(self.data.pca_data["expl_var"].shape[0])]

            self.pca_sel_pipe.send(sel[pc_names])
            if self.data.current_sel is None:
                self.pca_nonsel_pipe.send([])
            else:
                self.pca_nonsel_pipe.send(nonsel[pc_names])
            
            self.pca_opts.xlim = (self.data.exrds[self.pca_opts.xlabel].min().item(), self.data.exrds[self.pca_opts.xlabel].max().item())
            self.pca_opts.ylim = (self.data.exrds[self.pca_opts.ylabel].min().item(), self.data.exrds[self.pca_opts.ylabel].max().item())
        else:
            self.pca_sel_pipe.send([])
            self.pca_nonsel_pipe.send([])
        

        if self.data.do_tsne and "t-SNE component 1" in data:
            tsne_names = [f"t-SNE component {i+1}" for i in range(self.data.tsne_data.shape[1])]
        
            self.tsne_sel_pipe.send(sel[tsne_names].set_coords("t-SNE component 1"))
            if self.data.current_sel is None:
                self.tsne_nonsel_pipe.send([])
            else:
                self.tsne_nonsel_pipe.send(nonsel[tsne_names].set_coords("t-SNE component 1"))
        
            self.tsne_opts.xlim = (self.data.tsne_data[:, 0].min(), self.data.tsne_data[:, 0].max())
            self.tsne_opts.ylim = (self.data.tsne_data[:, 1].min(), self.data.tsne_data[:, 1].max())
        else:
            self.tsne_sel_pipe.send([])
            self.tsne_nonsel_pipe.send([])
            

        if self.data.do_umap and "UMAP component 1" in data:
            umap_names = [f"UMAP component {i+1}" for i in range(self.data.umap_data.shape[1])]
            
            self.umap_sel_pipe.send(sel[umap_names].set_coords("UMAP component 1"))
            if self.data.current_sel is None:
                self.umap_nonsel_pipe.send([])
            else:
                self.umap_nonsel_pipe.send(nonsel[umap_names].set_coords("UMAP component 1"))

            self.umap_opts.xlim = (self.data.umap_data[:, 0].min(), self.data.umap_data[:, 0].max())
            self.umap_opts.ylim = (self.data.umap_data[:, 1].min(), self.data.umap_data[:, 1].max())     
        else:
            self.umap_sel_pipe.send([])
            self.umap_nonsel_pipe.send([])
            
    @param.depends("data.all_data", "data.all_norm_data", "data.sel_data", "data.band", "data.current_sel", watch=True)
    def update_sample_image_pipe(self):
        """
        Update the data pipe of the sample plot, when one of the above 
            parameters changes.
        """

        if "absorbance" not in self.data.exrds:
            self.sample_sel_pipe.send([])
            self.sample_nonsel_pipe.send([])
            
        else:
            # Get the relevant data and take the selected quantile
            data = self.dp.calculate_quantile(self.data.exrds.sel(band=self.data.band).absorbance, self.data.quantile)
            
            sel = data.unstack()
            nonsel = []

            # If a selection has been made split selected and non-selected data points 
            if self.data.current_sel is not None:
                sel = data.where(self.data.current_sel).unstack()
                nonsel = data.where(~self.data.current_sel).unstack()
            
            self.sample_opts.title = f"Absorbance at {self.data.band} cm⁻¹"
            self.sample_opts.aspect = None
           
            x_min, x_max, y_min, y_max = self.dp.calculate_square_plot_limits(
                self.data.x_min, self.data.x_max, self.data.y_min, self.data.y_max
            )
            
            self.sample_opts.xlim = (x_min, x_max)
            self.sample_opts.ylim = (y_min, y_max)

            self.sample_sel_pipe.send(sel)
            self.sample_nonsel_pipe.send(nonsel)

            self.sample_opts.aspect = "equal"

    @param.depends("data.all_data", "data.all_norm_data", "data.sel_data", "data.current_sel", watch=True)
    def update_spectrum_all_pipe(self):
        """
        Update the data pipe of the spectrum plot of all the data (ignoring any 
            possible selections), when one of the above parameters changes.
        """

        if "absorbance" in self.data.exrds:
            self.spectrum_all_pipe.send(self.data.exrds.absorbance.mean("idx"))
            
    @param.depends("data.all_data", "data.sel_data", "data.current_sel", watch=True)
    def update_spectrum_sel_pipe(self):
        """
        Update the data pipe of the spectrum plot of selected data, when one of 
            the above parameters changes.
        """

        if "absorbance" in self.data.exrds:
            data = dv.dc.exrds.absorbance
            
            if self.data.current_sel is None:
                self.spectrum_opts.show_legend = False
                self.spectrum_sel_pipe.send([])
            else: 
                self.spectrum_opts.show_legend = True
                self.spectrum_sel_pipe.send(data.where(self.data.current_sel).mean("idx"))
    
    def get_view_objects(self):
        """
        Get the plot objects, so they can be instantiated as hidden objects later
        """
        
        self.view_objects = {
            "sample": pn.pane.HoloViews(self.plot_sample_image()),
            "spectrum": pn.pane.HoloViews(self.plot_spectrum()),
            "pca": pn.pane.HoloViews(self.plot_pca()),
            "tsne": pn.pane.HoloViews(self.plot_tsne()),
            "umap": pn.pane.HoloViews(self.plot_umap()),
        }
    
        return self.view_objects


class DashboardView(param.Parameterized):
    
    print("Initialising dashboard...")

    # Get all the relevant elements from the different classes
    dc = DataContainer()
    dp = DataProcessor(dc)
    filename_select = dp.filename_select()
    dp_sidebar = dp.sidebar()
    rs = RegionSelection(dc, dp)
    rs_view = rs.view()
    ep = EmbeddingsProducer(dc)
    ep_sidebar = ep.sidebar()
    ep_modal = ep.modal()
    ed = ExplorationDashboard(dc, dp)

    # Create ed_view
    ed_view_objects = ed.get_view_objects()
    # Copy all the objects and remove the embedding plots that are not initially visible
    ed_view_visible_objects =  ed_view_objects.copy()
    ed_view_visible_objects.pop("pca")
    ed_view_visible_objects.pop("tsne")
    ed_view_visible_objects.pop("umap")
    ed_view = pn.GridBox(*list(ed_view_visible_objects.values()), ncols=dc.plotting_cols, visible=False)

    # Template creation
    dc.template.sidebar.append(filename_select)
    dc.template.sidebar.append(dp_sidebar)
    dc.template.sidebar.append(ep_sidebar)
    
    dc.template.modal.append(ep_modal)

    dc.template.main.append(rs_view)
    dc.template.main.append(ed_view)

    
    @param.depends("dc.filename", watch=True)
    def reveal_hidden_elements(self):
        """If a filename is provided, reveal certain previously hidden elements"""
        if self.dc.filename:
            self.dp_sidebar.visible = True
            self.ep_sidebar.visible = True
            self.ed_view.visible = True
        
    @param.depends("dc.pca_data", watch=True)
    def update_pca_visibility(self):
        if self.dc.pca_data is None:
            self.ed_view_visible_objects.pop("pca")
        else:
            self.ed_view_visible_objects["pca"] = self.ed_view_objects["pca"]
        
        self.ed_view.objects = list(self.ed_view_visible_objects.values())
    
    @param.depends("dc.tsne_data", watch=True)
    def update_tsne_visibility(self):
        if self.dc.tsne_data is None:
            self.ed_view_visible_objects.pop("tsne")
        else:
            self.ed_view_visible_objects["tsne"] = self.ed_view_objects["tsne"]
        
        self.ed_view.objects = list(self.ed_view_visible_objects.values())

    @param.depends("dc.umap_data", watch=True)
    def update_umap_visibility(self):
        if self.dc.umap_data is None:
            self.ed_view_visible_objects.pop("umap")
        else:
            self.ed_view_visible_objects["umap"] = self.ed_view_objects["umap"]
        
        self.replot_objects()
        
    @param.depends("dc.plotting_cols", watch=True)
    def update_grid_cols(self):
        """Update the number of visible columns, when a new number is chosen"""
        self.ed_view.ncols = self.dc.plotting_cols
        self.replot_objects()

    def replot_objects(self):
        """
        Replot the ExplorationDashboard view objects in order for them to update 
            properly after changes have been made
        """
        self.ed_view.objects = list(self.ed_view_visible_objects.values())

    def view(self):
        print("Starting dashboard...")
        return self.dc.template.servable("Spectral data visualisation dashboard")

dv = DashboardView()
dv.view().show()