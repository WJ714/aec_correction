import warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
from statsmodels.nonparametric import smoothers_lowess as sml
from scipy import interpolate
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def linear_extp(x,y):
    """
    Description: Fits a linear regression model to the data.

    Parameters:
    -----------
    x (numpy array): The input data for the model.
    y (numpy array): The target data for the model.

    Returns:
    -----------
    LinearRegression model: The fitted model.
    """
    return LinearRegression().fit(x.reshape((-1, 1)), y)

def compute_r_p(x, y, min_length=3):
    """
    Description: Computes the Pearson correlation coefficient and the p-value for testing non-correlation.

    Parameters:
    -----------
    x (numpy array): 1D array.
    y (numpy array): 1D array.
    min_length (int): Minimum length of arrays x and y. Default is 3.

    Returns:
    -----------
    correlation (float): Pearson correlation coefficient.
    p_value (float): Two-tailed p-value.
    """

    try:
        if len(x) < min_length or len(y) < min_length:
            raise ValueError("Arrays are too short")
        
        correlation, p_value = pearsonr(x, y)
        return correlation,p_value
    except ValueError:
        return np.nan,np.nan
    
def fcor_estimator(LE, NETRAD, H, G,  maskQC, frac_in_lowess=2/3, n_qntl_bins=60,
                    ErrorIn='Error_In_LeH'):
    """
    Description: Estimates the correction factor based on the given parameters.

    Parameters:
    -----------
    LE (xarray DataArray): Latent heat flux.
    NETRAD (xarray DataArray): Net radiation.
    H (xarray DataArray): Sensible heat flux.
    G (xarray DataArray): Ground heat flux.
    maskQC (xarray DataArray): Quality control mask.
    frac_in_lowess (float): Fraction of the data used when estimating each y-value. Default is 2/3.
    n_qntl_bins (int): Number of quantile bins. Default is 60.
    ErrorIn (str): Error in 'LeH' or 'RnG'. Default is 'Error_In_LeH'.

    Returns:
    -----------
    xr.Dataset: A dataset with the correction factor, exog and endog.
    """

    _endog = LE + H
    _exog  = NETRAD - G
    _m0 = np.isfinite(_endog) & np.isfinite(_exog) & (maskQC==1)

    if ErrorIn=='Error_In_RnG':
        _estimate = sml.lowess(_exog[_m0].values, _endog[_m0].values, return_sorted=True, frac=frac_in_lowess)
        x = _estimate[:,1]
        y = _estimate[:,0]
    if ErrorIn=='Error_In_LeH':
        _estimate = sml.lowess(_endog[_m0].values, _exog[_m0].values, return_sorted=True, frac=frac_in_lowess)
        x = _estimate[:,0]
        y = _estimate[:,1]

    n_slope_bins = _m0.sum().values.tolist()//n_qntl_bins
    A = np.vstack([x, np.ones(len(x))]).T

    if n_slope_bins<3:
        n_slope_bins = 3
    qntls = np.quantile(x, np.linspace(0, 1, n_slope_bins))

    out_y = np.zeros(qntls.size-1)*np.nan
    out_x = np.zeros(qntls.size-1)*np.nan

    for j in range(qntls.size-1):
        start = np.argmin(np.abs(x-qntls[j]))
        end   = np.argmin(np.abs(x-qntls[j+1]))
        a1, _, _, _ = np.linalg.lstsq(A[start:end], y[start:end], rcond=None)
        out_x[j]  = x[start] + ((x[end]-x[start])/2)
        r_val, p_val = pearsonr(x[start:end], y[start:end])
        out_y[j]  = a1[0]

    fcor_y = 1/out_y
    f = interpolate.interp1d(out_x, fcor_y, bounds_error=False)

    _exog.name='exog'
    _endog.name='endog'
    da = _exog.to_dataframe()
    da['fcor'] = f(da['exog'])

    _df = da.dropna(axis=0)
    df_sorted = _df.sort_values(by='exog')
    left_tail_x = df_sorted['exog'][:21].values
    left_tail_y = df_sorted['fcor'][:21].values
    left_tail_xs = (da['exog'][da['exog']<out_x[0]]).values
    if len(left_tail_xs) >0:
        da['fcor'][da['exog']<out_x[0]] = linear_extp(left_tail_x,left_tail_y).predict(left_tail_xs.reshape((-1, 1)))

    right_tail_x = df_sorted['exog'][-21:].values
    right_tail_y = df_sorted['fcor'][-21:].values
    right_tail_xs = (da['exog'][da['exog']>out_x[-1]]).values
    if len(right_tail_xs) >0:
        da['fcor'][da['exog']>out_x[-1]] = linear_extp(right_tail_x,right_tail_y).predict(right_tail_xs.reshape((-1, 1)))

    Fcor = xr.DataArray(
        da['fcor'],
        coords=_exog.coords,
        dims=("time"),
        attrs={'Long_name':f'Correction facor estimated based on {ErrorIn}'},
        name="FCor"
    )
    return xr.merge([Fcor, _exog, _endog])
