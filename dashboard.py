# # Imports
# ## Python imports and setup

# Native Python
import re
import socket
import colorsys
from pathlib import Path
from copy import deepcopy
from functools import partial


# Data storage & processing
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from PyMca5.PyMca.FsmReader import parseFile

# Embeddings
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Plotting
import panel as pn
css = '''
.bk.custom-panel {
  background: #f0f0f0;
  padding: 50px;
  border-radius: 10px;
}
.bk.divider {
  border-top: 1px solid #d9d9d9;
  height: 5px;
}
'''
pn.config.raw_css=[css]

# import hsluv
import param
import holoviews as hv
from bokeh.themes.theme import Theme
from holoviews.operation.datashader import datashade, dynspread, rasterize


# To enable saving selections
from numpy import array
from holoviews import dim
from holoviews.element.selection import spatial_select

# Set holoviews width (%) and backend

hv.extension("bokeh", width=100)
hv.Store.set_current_backend("bokeh")
pn.extension(sizing_mode="stretch_width")

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

# Set the default size of the plots, depending on the computer running it

size = 250

if socket.gethostname() == "Laurens-PC":
    size = 300

# ### New Dashboard

class Opts(param.Parameterized):
    
    xlabel = param.String()
    ylabel = param.String()
    
    xlim = param.NumericTuple()
    ylim = param.NumericTuple()
    
    title = param.String()
    aspect = param.String(None, allow_None=True)

# Region selection
class DataContainer(param.Parameterized):
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
    shifting_band_selector = pn.widgets.IntInput(name="Shifting band")
    normalisation_band_selector = pn.widgets.IntInput(name="Normalisation band")
    
    # Region selection    
    x_min = param.Integer(-1)
    x_max = param.Integer(-1)
    y_min = param.Integer(-1)
    y_max = param.Integer(-1)
    sel_plot_opts = Opts(xlim=(0, 10), ylim=(0, 10))
    
    # Plotting
    band_selector = pn.widgets.IntInput(name="Plotting band")
    band = param.Selector()
    band_index = None
    quantile = param.Number(.99, bounds=(.5, 1))
    
    # Embeddings
    pca_data = param.Dict()
    tsne_data = param.Array()
    umap_data = param.Array()
    no_pca_comps = param.Integer(10, bounds=(1, None))
    
    EMPTY_DATA_2D = [[np.NaN, np.NaN], [np.NaN, np.NaN]]
    EMPTY_DATA_3D = [[[np.NaN, np.NaN], [np.NaN, np.NaN]], [[np.NaN, np.NaN], [np.NaN, np.NaN]]]
    exrds = xr.Dataset()
    eds = hv.Dataset([])
    
    calculate_button = pn.widgets.Button(name="Calculate embeddings")
    previous_calc_btn_clicks = param.Integer()
    
    current_sel = param.Array()

    
class DataProcessor(param.Parameterized):
    
    def __init__(self, datacontainer):
        self.data = datacontainer
        super().__init__()  
    
    # Data processing
    @param.depends("data.filename", watch=True)
    def import_data(self):
        """Import data when the selected file changes"""
        spectral_info, spectral_data = parseFile(self.data.filename)
        
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
            self.data.raw_data = -np.log10(self.data.spectral_data / 100)
            
            # Remove NaN values from logarithm of negative values
            np.nan_to_num(self.data.raw_data, copy=False, nan=np.nanmin(self.data.raw_data))
        
        else:
            self.data.raw_data = self.data.spectral_data
            
    @param.depends("data.raw_data", "data.do_baseline_shift", "data.shifting_band", "data.shifting_range", watch=True)
    def baseline_shift(self):
        """Carry out a baseline shift"""
        if self.data.do_baseline_shift:
            shift_band_1 = self.data._get_band(self.data.shifting_band - self.data.shifting_range)
            shift_band_2 = self.data._get_band(self.data.shifting_band + self.data.shifting_range)
            shift_band_low = min(shift_band_1, shift_band_2)
            shift_band_high = max(shift_band_1, shift_band_2)

            shift_abs = self.data.raw_data[..., np.arange(shift_band_low, shift_band_high)].mean(-1)
            self.data.all_data = self.data.raw_data - np.repeat(shift_abs[..., np.newaxis], self.data.raw_data.shape[-1], axis=-1)
            
        else:
            self.data.all_data = self.data.raw_data
        
    @param.depends("data.all_data", "data.do_normalisation", "data.normalisation_band", "data.normalisation_range", watch=True)
    def normalise(self):
        """Add data to the correct data containers and normalise it"""
        
        # Do nothing if there is not data yet
        if self.data.all_data is not None:
            if self.data.do_normalisation:
                norm_band_1 = self.data._get_band(self.data.normalisation_band - self.data.normalisation_range)
                norm_band_2 = self.data._get_band(self.data.normalisation_band + self.data.normalisation_range)
                norm_band_low = min(norm_band_1, norm_band_2)
                norm_band_high = max(norm_band_1, norm_band_2)
                
                norm_mean = np.median(self.data.all_data[..., np.arange(norm_band_low, norm_band_high)].mean(-1))

                self.data.all_norm_data = np.divide(self.data.all_data, np.repeat(self.data.all_data[..., np.arange(norm_band_low, norm_band_high)].mean(-1)[..., np.newaxis], self.data.all_data.shape[-1], axis=-1)) * norm_mean
                self.data.exrds = xr.Dataset(
                    data_vars = dict(absorbance=(["x", "y", "band"], self.data.all_norm_data)),
                    coords = dict(x=np.arange(self.data.all_norm_data.shape[0]), y=np.arange(self.data.all_norm_data.shape[1]), band=self.data.wavenumbers),
                ).stack(idx=("x", "y"))
            else:
                self.data.exrds = xr.Dataset(
                    data_vars = dict(absorbance=(["x", "y", "band"], self.data.all_data)),
                    coords = dict(x=np.arange(self.data.all_data.shape[0]), y=np.arange(self.data.all_data.shape[1]), band=self.data.wavenumbers),
                ).stack(idx=("x", "y"))
                
            self.data.eds = hv.Dataset(self.data.exrds, ["band", "idx"])

    @param.depends("data.band_selector.value", watch=True)
    def update_band(self):
        # Initialise band_selector first with data
        if self.data.band_selector.value != 0:
            self.data.band = self.data.band_selector.value
    
    @param.depends("data.shifting_band_selector.value", watch=True)
    def update_shifting_band(self):
        self.data.shifting_band = self.data.shifting_band_selector.value
    
    @param.depends("data.normalisation_band_selector.value", watch=True)
    def update_normalisation_band(self):
        self.data.normalisation_band = self.data.normalisation_band_selector.value
    
    @param.depends("data.band", watch=True)
    def calculate_band_index(self):
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
    
    def filename_select(self):
        return pn.widgets.TextInput.from_param(self.data.param.filename, value=r"C:\Users\gieri\OneDrive\Documenten\Course manuals\Stage\Nachtwacht\SK-C-5_047\SK-C-5_047_FTIR-ATR_4cm_2021-04-12_raw.fs")
    
    def sidebar(self):
        ta_toggle = pn.widgets.Checkbox.from_param(self.data.param.do_trans_to_abs)
        baseline_toggle = pn.widgets.Checkbox.from_param(self.data.param.do_baseline_shift)
        normalisation_toggle = pn.widgets.Checkbox.from_param(self.data.param.do_normalisation)
        shift_range = pn.widgets.IntInput.from_param(self.data.param.shifting_range)
        norm_range = pn.widgets.IntInput.from_param(self.data.param.normalisation_range)
        quantile_selector = pn.widgets.FloatInput.from_param(self.data.param.quantile)

        return pn.Column(
            pn.Row(ta_toggle, baseline_toggle, normalisation_toggle),
            pn.Row(self.data.shifting_band_selector, self.data.normalisation_band_selector),
            pn.Row(shift_range, norm_range),
            pn.Row(self.data.band_selector, quantile_selector),
            visible=False,
        )
    
    def get_coordinates(self, data):
        """
        Create a coordinate space from the provided data.
        
        Args:
            data (np.array): spectral data.
            
        Returns:
            xy (np.array): 2D array with the x- and y- coordinates in 
                a x times y grid.
        """
        x = [x % data.shape[1] for x in range(np.product(data.shape[:2]))]
        y = [int(np.floor(y / data.shape[1])) for y in range(np.product(data.shape[:2]))]
        
        return x, y
    
    def calculate_quantile(self, data, quantile=.99):
        lower_q = np.nanquantile(data, 1 - quantile)
        upper_q = np.nanquantile(data, quantile)

        return np.clip(data, lower_q, upper_q)
            
    def _get_band(self, band):
        return int( (band - self.data.band_max) / self.data.band_change )
    

class RegionSelection(param.Parameterized):
    sample_selection = hv.link_selections.instance(unselected_alpha=.5, cross_filter_mode="overwrite")
    
    EMPTY_MAP = ([0, 10_000], [0, 10_000], [[None, None], [None, None]])
    data_pipe = hv.streams.Pipe(EMPTY_MAP)
    sel_data_pipe = hv.streams.Pipe(EMPTY_MAP)
    
    def __init__(self, datacontainer, dataprocessor):
        self.data = datacontainer
        self.dp = dataprocessor
        super().__init__()    
    
    
    def create_selection_plot(self):
         return self.sample_selection(
             hv.DynamicMap(hv.Image, streams=[self.data_pipe]).opts(
                 hv.opts.Image(
                     cmap="viridis", framewise=True, active_tools=["box_select"], title="Region selection",
                     aspect="equal", responsive=True,
                 ),
             )
         ).redim(x="selection_x", y="selection_y")
    
    def show_selection(self):
        return hv.DynamicMap(
            hv.Image, streams=[self.sel_data_pipe]
        ).opts(
            cmap="viridis", framewise=True, tools=["hover"],
            title="Selected region",  aspect="equal", responsive=True, axiswise=True,
        ).redim(x="sel_x", y="sel_y").redim.label(sel_x="x", sel_y="y", z="Absorbance").apply.opts(
            xlim = self.data.sel_plot_opts.param.xlim, ylim = self.data.sel_plot_opts.param.ylim, 
        )

    @param.depends("data.all_data", "data.all_norm_data", "data.do_normalisation", "data.band", "data.quantile", watch=True)
    def update_data_pipe(self):
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
            
    @param.depends("data.sel_data", "data.band", "data.quantile", watch=True)
    def update_sel_data_pipe(self):
        if self.sample_selection.selection_expr is None or self.data.sel_data is None:
            self.sel_data_pipe.send(self.EMPTY_MAP)
        
        else:
            data = self.dp.calculate_quantile(self.data.sel_data[..., self.data.band_index], self.data.quantile)
                
            self.sel_data_pipe.send(
                (np.arange(data.shape[1]),
                 np.arange(data.shape[0]),
                 data) 
            )
            
            # Image x and y flipped from dataset 
            self.data.sel_plot_opts.xlim = (self.data.exrds.y.min().item(), self.data.exrds.y.max().item())
            self.data.sel_plot_opts.ylim = (self.data.exrds.x.min().item(), self.data.exrds.x.max().item())
            
    @param.depends("sample_selection.selection_expr", watch=True)
    def update_bounds(self):
        if self.sample_selection.selection_expr is None:
            self.data.x_min, self.data.x_max, self.data.y_min, self.data.y_max = (-1, -1, -1, -1)
        else:
            sel_expr = str(self.sample_selection.selection_expr)
            self.data.x_min = max(0, round(float(re.search(r"'x'\)>=(\-*\d+\.*\d*)", sel_expr).group(1))))
            self.data.x_max = min(self.data.all_data.shape[1], round(float(re.search(r"'x'\)<=(\-*\d+\.*\d*)", sel_expr).group(1))))
            self.data.y_min = max(0, round(float(re.search(r"'y'\)>=(\-*\d+\.*\d*)", sel_expr).group(1))))
            self.data.y_max = min(self.data.all_data.shape[0], round(float(re.search(r"'y'\)<=(\-*\d+\.*\d*)", sel_expr).group(1))))
    
    @param.depends("sample_selection.selection_expr", "data.all_data", "data.do_normalisation", watch=True)
    def update_selection_data(self):
        if self.sample_selection.selection_expr:
            if self.data.do_normalisation:
                sel_data = self.data.all_norm_data[self.data.y_min:self.data.y_max, self.data.x_min:self.data.x_max, :]
            else:
                sel_data = self.data.all_data[self.data.y_min:self.data.y_max, self.data.x_min:self.data.x_max, :]
            
            # Replace the old data with the new selection
            self.data.exrds = xr.Dataset(
                data_vars = dict(absorbance=(["x", "y", "band"], sel_data)),
                coords = dict(x=np.arange(sel_data.shape[0]), y=np.arange(sel_data.shape[1]), band=self.data.wavenumbers),
            ).stack(idx=("x", "y"))
            self.data.eds = hv.Dataset(self.data.exrds, ["band", "idx"])
            self.data.sel_data = sel_data
        else:
            self.dp.normalise()
            self.data.sel_data = None
        
    def view(self):
        selection_plot = self.create_selection_plot()
        selected_plot = self.show_selection()
        return pn.Row(selection_plot, selected_plot,)
    

class EmbeddingsProducer(param.Parameterized):

    do_pca = param.Boolean()
    do_tsne = param.Boolean()
    do_umap = param.Boolean()

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
    
    
    def __init__(self, datacontainer):
        self.data = datacontainer
        super().__init__()
        
    
    @param.depends("data.calculate_button.clicks", watch=True)
    def update_calc_button_clicks(self):
        self.data.previous_calc_btn_clicks = self.data.calculate_button.clicks
    
    
    @param.depends("data.previous_calc_btn_clicks", watch=True)
    def calculate_embeddings(self):
        print("Calculating embeddings")
        
        # Skip calculation if there is no data yet
        if len(self.data.exrds) != 0:
            # Flatten the data into a 2D matrix
            # If no region is selected, take all data
            flattened_data = self.data.exrds.absorbance.data.reshape(-1, self.data.exrds.absorbance.shape[-1]).T

            # PCA
            if self.do_pca :
                # Only recalculate if there is no current data or the absorbance data has changed shape
                if self.data.pca_data is None or self.data.pca_data["scores"].shape[0] != self.data.exrds.absorbance.idx.shape[0]:
                    pca = PCA(n_components=self.data.no_pca_comps, **self.pca_params)

                    pca_data = dict(
                        scores = pca.fit_transform(flattened_data),
                        loadings = pca.components_.T * np.sqrt(pca.explained_variance_),
                        expl_var = pca.explained_variance_ratio_,
                    )

    #                 columns = [f"PC {i+1}" for i in range(pca_data["scores"].shape[1])]
                    self.data.exrds = self.data.exrds.assign({
                        f"PC {i+1}": ("idx", pca_data["scores"][:, i])
                        for i in range(pca_data["scores"].shape[1])
                    })
                    self.data.eds = hv.Dataset(self.data.exrds, ["band", "idx"])

                    self.data.pca_data = pca_data
            else:
                self.data.pca_data = None

            # t-SNE
            if self.do_tsne:
                # Only recalculate if there is no current data or the absorbance data has changed shape
                if self.data.tsne_data is None or self.data.tsne_data.shape[0] != self.data.exrds.absorbance.idx.shape[0]:
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
            if self.do_umap:
                # Only recalculate if there is no current data or the absorbance data has changed shape
                if self.data.umap_data is None or self.data.umap_data.shape[0] != self.data.exrds.absorbance.idx.shape[0]:
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
    
    def sidebar(self):
        do_pca = pn.widgets.Checkbox.from_param(self.param.do_pca, name="PCA")
        pca_comp = pn.widgets.IntInput.from_param(self.data.param.no_pca_comps, name="No. of PCA components")
        pca_params = pn.widgets.LiteralInput.from_param(self.param.pca_params, name="Other PCA parameters (dict)")
        do_tsne = pn.widgets.Checkbox.from_param(self.param.do_tsne, name="t-SNE")
        tsne_dist = pn.widgets.Select.from_param(self.param.tsne_distance_metric, name="t-SNE distance metric")
        tsne_params = pn.widgets.LiteralInput.from_param(self.param.tsne_params, name="Other t-SNE parameters (dict)")
        do_umap = pn.widgets.Checkbox.from_param(self.param.do_umap, name="UMAP")
        umap_dist = pn.widgets.Select.from_param(self.param.umap_distance_metric, name="UMAP distance metric")
        umap_params = pn.widgets.LiteralInput.from_param(self.param.umap_params, name="Other UMAP parameters (dict)")
        cb = self.data.calculate_button
        
        return pn.Column(
            pn.Row(
                pn.Column(do_pca, pca_comp, pca_params),
                pn.Column(do_tsne, tsne_dist, tsne_params),
            ),
            pn.Row(
                pn.Column(do_umap, umap_dist, umap_params,),
                None
            ),
            cb,
            visible=False,
        )


class ExplorationDashboard(param.Parameterized):
    
    pcx = param.Integer(1)
    pcy = param.Integer(2)
    
    pca_sel_pipe = hv.streams.Pipe([])
    pca_nonsel_pipe = hv.streams.Pipe([])
    tsne_sel_pipe = hv.streams.Pipe([])
    tsne_nonsel_pipe = hv.streams.Pipe([])
    umap_sel_pipe = hv.streams.Pipe([])
    umap_nonsel_pipe = hv.streams.Pipe([])
    sample_sel_pipe = hv.streams.Pipe([])
    sample_nonsel_pipe = hv.streams.Pipe([])
    
    pca_opts = Opts(xlabel="PC 1", ylabel="PC 2")
    tsne_opts = Opts()
    umap_opts = Opts()
    sample_opts = Opts()
    
    plot_options = dict(framewise=True, responsive=True)
    
    ls1 = hv.link_selections.instance()
    ls2 = hv.link_selections.instance()
    ls3 = hv.link_selections.instance()
    ls4 = hv.link_selections.instance()
    
    def __init__(self, datacontainer, dataprocessor):
        self.data = datacontainer
        self.dp = dataprocessor
        self.param.pcx.bounds = (1, self.data.no_pca_comps)
        self.param.pcy.bounds = (1, self.data.no_pca_comps)
        
        super().__init__()
    
    clear_selection_button = pn.widgets.Button(name="Clear selection")

    @param.depends("clear_selection_button.clicks", "data.sel_data", watch=True)
    def clear_selection(self):
        self.data.current_sel = None
    
    def get_pca_plot(self, data):
        if self.pca_opts.xlabel in data:
            return hv.Scatter(data.set_coords(self.pca_opts.xlabel), kdims=[self.pca_opts.xlabel], vdims=[self.pca_opts.ylabel], )
        else:
            return hv.Scatter([], kdims=[self.pca_opts.xlabel], vdims=[self.pca_opts.ylabel], )
    
    def plot_pca(self):
        return self.ls1(
            dynspread(rasterize(hv.DynamicMap(self.get_pca_plot, streams=[self.pca_sel_pipe]))).opts(cmap="kbc_r") *
            dynspread(rasterize(hv.DynamicMap(self.get_pca_plot, streams=[self.pca_nonsel_pipe]))).opts(cmap="reds", alpha=0.5)
        ).opts(
                hv.opts.Scatter(**self.plot_options, title="PCA"),
                hv.opts.Image(**self.plot_options, title="PCA"),
        ).apply.opts(xlabel=self.pca_opts.param.xlabel, ylabel=self.pca_opts.param.ylabel, 
                     xlim=self.pca_opts.param.xlim, ylim=self.pca_opts.param.ylim
        )
     
    def plot_tsne(self):
        return self.ls2(
            dynspread(rasterize(hv.DynamicMap(partial(hv.Scatter, kdims="t-SNE component 1", vdims="t-SNE component 2"), streams=[self.tsne_sel_pipe]))).opts(cmap="kbc_r") *
            dynspread(rasterize(hv.DynamicMap(partial(hv.Scatter, kdims="t-SNE component 1", vdims="t-SNE component 2"), streams=[self.tsne_nonsel_pipe]))).opts(cmap="reds", alpha=0.5)
        ).opts(
            hv.opts.Image(**self.plot_options, title="t-SNE embedding", ),
            hv.opts.RGB(**self.plot_options, title="t-SNE embedding", ),
        ).apply.opts(xlim=self.tsne_opts.param.xlim, ylim=self.tsne_opts.param.ylim)
    
    def plot_umap(self):
        return self.ls3(
            dynspread(rasterize(hv.DynamicMap(partial(hv.Scatter, kdims=["UMAP component 1"], vdims=["UMAP component 2"]), streams=[self.umap_sel_pipe]))).opts(cmap="kbc_r") *
            dynspread(rasterize(hv.DynamicMap(partial(hv.Scatter, kdims=["UMAP component 1"], vdims=["UMAP component 2"]), streams=[self.umap_nonsel_pipe]))).opts(cmap="reds", alpha=0.5)
        ).opts(
            hv.opts.Image(**self.plot_options, title="UMAP embedding", ),
            hv.opts.RGB(**self.plot_options, title="UMAP embedding", ),
        ).apply.opts(xlim=self.umap_opts.param.xlim, ylim=self.umap_opts.param.ylim)
    
    def plot_sample(self):
        return self.ls4(
            hv.DynamicMap(partial(hv.Image, kdims=["y", "x"], vdims=["absorbance"]), streams=[self.sample_sel_pipe]).opts(cmap="kbc_r") *
            hv.DynamicMap(partial(hv.Image, kdims=["y", "x"], vdims=["absorbance"]), streams=[self.sample_nonsel_pipe]).opts(cmap="reds", alpha=0.2)
        ).opts(
            hv.opts.Image(**self.plot_options, clipping_colors={"NaN": "transparent"}),
        ).apply.opts(
            title = self.sample_opts.param.title, aspect=self.sample_opts.param.aspect,
            xlim = self.sample_opts.param.xlim, ylim = self.sample_opts.param.ylim, 
        )
    
    @param.depends("ls1.selection_expr", watch=True)
    def update_pca_selection(self):
        if self.ls1.selection_expr is not None:
            selection_coords = self.data.eds.select(self.ls1.selection_expr).data.idx.data.astype("i,i")
            self.update_selection_indices(selection_coords)
            
            self.ls1.selection_expr = None
    
    @param.depends("ls2.selection_expr", watch=True)
    def update_tsne_selection(self):
        if self.ls2.selection_expr is not None:
            selection_coords = self.data.eds.select(self.ls2.selection_expr).data.idx.data.astype("i,i")
            self.update_selection_indices(selection_coords)
            
            self.ls2.selection_expr = None
            
    @param.depends("ls3.selection_expr", watch=True)
    def update_umap_selection(self):
        if self.ls3.selection_expr is not None:
            selection_coords = self.data.eds.select(self.ls3.selection_expr).data.idx.data.astype("i,i")
            self.update_selection_indices(selection_coords)
            
            self.ls3.selection_expr = None
            
    @param.depends("ls4.selection_expr", watch=True)
    def update_sample_selection(self):
        if self.ls4.selection_expr is not None:
            data = (self.data.exrds.idx.data.astype("i,i"), self.data.exrds.x.data, self.data.exrds.y.data)
            ds = hv.Dataset(data, ["idx", "x", "y"])
            selection_coords = ds.select(self.ls4.selection_expr).data["idx"].to_numpy().astype("i,i")
            self.update_selection_indices(selection_coords)
            
            self.ls4.selection_expr = None
    
    def update_selection_indices(self, selection_coords):
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
    
    @param.depends("data.pca_data", "data.tsne_data", "data.umap_data", "data.current_sel", "data.sel_data", watch=True)
    def update_embeddings_pipes(self):
        data = self.data.exrds.drop(["absorbance", "band"])
        
        sel = data.unstack()
        if self.data.current_sel is not None:
            # Split selected and non-selected data points and send them to the appropriate pipes
            sel = data[dict(idx=self.data.current_sel)].unstack()
            nonsel = data[dict(idx=~self.data.current_sel)].unstack()
        
        if "PC 1" not in self.data.exrds:
            self.pca_sel_pipe.send([])
            self.pca_nonsel_pipe.send([])
        else:
            pc_names = [f"PC {i+1}" for i in range(self.data.pca_data["expl_var"].shape[0])]

            self.pca_sel_pipe.send(sel[pc_names])
            if self.data.current_sel is None:
                self.pca_nonsel_pipe.send([])
            else:
                self.pca_nonsel_pipe.send(nonsel[pc_names])
                
            self.pca_opts.xlim = (self.data.exrds[self.pca_opts.xlabel].min().item(), self.data.exrds[self.pca_opts.xlabel].max().item())
            self.pca_opts.ylim = (self.data.exrds[self.pca_opts.ylabel].min().item(), self.data.exrds[self.pca_opts.ylabel].max().item())
            
        if "t-SNE component 1" not in self.data.exrds:
            self.tsne_sel_pipe.send([])
            self.tsne_nonsel_pipe.send([])
        else:
            tsne_names = [f"t-SNE component {i+1}" for i in range(self.data.tsne_data.shape[1])]
        
            self.tsne_sel_pipe.send(sel[tsne_names].set_coords("t-SNE component 1"))
            if self.data.current_sel is None:
                self.tsne_nonsel_pipe.send([])
            else:
                self.tsne_nonsel_pipe.send(nonsel[tsne_names].set_coords("t-SNE component 1"))
        
            self.tsne_opts.xlim = (self.data.tsne_data[:, 0].min(), self.data.tsne_data[:, 0].max())
            self.tsne_opts.ylim = (self.data.tsne_data[:, 1].min(), self.data.tsne_data[:, 1].max())
        
        if "UMAP component 1" not in self.data.exrds:
            self.umap_sel_pipe.send([])
            self.umap_nonsel_pipe.send([])
        else:
            umap_names = [f"UMAP component {i+1}" for i in range(self.data.umap_data.shape[1])]
            
            self.umap_sel_pipe.send(sel[umap_names].set_coords("UMAP component 1"))
            if self.data.current_sel is None:
                self.umap_nonsel_pipe.send([])
            else:
                self.umap_nonsel_pipe.send(nonsel[umap_names].set_coords("UMAP component 1"))

            self.umap_opts.xlim = (self.data.umap_data[:, 0].min(), self.data.umap_data[:, 0].max())
            self.umap_opts.ylim = (self.data.umap_data[:, 1].min(), self.data.umap_data[:, 1].max())     
            
    @param.depends("data.sel_data", "data.band", "data.current_sel", watch=True)
    def update_sample_pipe(self):
        if "absorbance" in self.data.exrds:
            data = self.dp.calculate_quantile(self.data.exrds.sel(band=self.data.band).absorbance, self.data.quantile)
            
            sel = data.unstack()
            nonsel = []
            if self.data.current_sel is not None:
                # Split selected and non-selected data points and send them to the appropriate pipes
                sel = data.where(self.data.current_sel).unstack()
                nonsel = data.where(~self.data.current_sel).unstack()
            
            self.sample_opts.title = f"Absorbance at {self.data.band} cm⁻¹"
            self.sample_opts.aspect = "equal"
            # Flip the coordinates
            self.sample_opts.xlim = (data.y.min().item(), data.y.max().item())
            self.sample_opts.ylim = (data.x.min().item(), data.x.max().item())

            self.sample_sel_pipe.send(sel)
            self.sample_nonsel_pipe.send(nonsel)

        else:
            self.sample_sel_pipe.send([])
            self.sample_nonsel_pipe.send([])
    
    def sidebar(self):
        pcx = pn.widgets.IntInput.from_param(self.param.pcx, name="Plot PC on x-axis")
        pcy = pn.widgets.IntInput.from_param(self.param.pcy, name="Plot PC on y-axis")
        return pn.Row(pcx, pcy)
        
    def view(self):
        return pn.Column(
            self.clear_selection_button,
            hv.Layout([
                self.plot_tsne(),
                self.plot_umap(),
                self.plot_pca(),
            ]).opts(shared_axes=False),
            self.plot_sample(),
        )


class DashboardView(param.Parameterized):
    
    dc = DataContainer()

    dp = DataProcessor(dc)
    filename_select = dp.filename_select()
    dp_sidebar = dp.sidebar()

    rs = RegionSelection(dc, dp)
    rs_view = rs.view()

    ep = EmbeddingsProducer(dc)
    ep_sidebar = ep.sidebar()

    ed = ExplorationDashboard(dc, dp)
    ed_sidebar = ed.sidebar()
    ed_view = ed.view()
    
    @param.depends("dc.filename")
    def reveal_dp_sidebar(self):
        if self.dc.filename:
            self.dp_sidebar.visible = True

            
    material = pn.template.MaterialTemplate(title="Spectral visualisation dashboard", sidebar_width=440)
    
    material.sidebar.append(filename_select)
    material.sidebar.append(dp.sidebar)
    material.main.append(rs_view)
    material.sidebar.append(ep_sidebar)
    material.sidebar.append(ed_sidebar)
    material.main.append(ed_view)
    
    def view(self):
        return pn.serve(
            self.material.servable()
        )

dv = DashboardView()
dv.view()