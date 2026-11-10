# Import packages 

import os
#from astroquery.sdss import SDSS
from astropy.table import Table
import astropy.units as u
from io import BytesIO
import requests
import pandas as pd
import numpy as np
from scipy import optimize
from scipy import stats
from scipy.stats import norm
import pylab as pl
import glob
from datetime import datetime as dt
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import re
import shutil
import tempfile
from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import PercentileInterval
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from regions import Regions, PixelRegion, SkyRegion
from casatasks import importfits, imstat, imhead, imfit
from casatasks import importfits, imstat, imhead, imfit
from casatools import image as iatool
ia = iatool()

powerlaw2 = lambda freq, tt, alpha: tt*freq**(-alpha)



###
#   Functions to convert coordinates
###
def deg_to_sexagesimal(ra_deg, dec_deg):
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    ra_str = coord.ra.to_string(unit=u.hour, sep=' ', precision=1)
    dec_str = coord.dec.to_string(unit=u.degree, sep=' ', precision=1, alwayssign=True)
    return ra_str, dec_str

def sexagesimal_to_deg(ra_str, dec_str):
    coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg), frame='icrs')
    return coord.ra.deg, coord.dec.deg


###
#   Funtion to extract a cutout image from Lotts
###
def view_cutout(source, ra, dec, size=5):

    if dec is not None:
        ra_str, dec_str = deg_to_sexagesimal(ra, dec)
        pos = f"{ra_str} {dec_str}"
    else:
        pos = ra

    url = "https://lofar-surveys.org/dr3-cutout.fits"
    params = {"pos": pos, "size": size}

    r = requests.get(url, params=params)
    if r.status_code != 200:
        print("Request failed:", r.status_code)
        return

    hdul = fits.open(BytesIO(r.content))
    hdul.writeto(f"{source}_LoTSS_DR3.fits", overwrite=True) # Save FITS file

    data = hdul[0].data
    header = hdul[0].header

    while data.ndim > 2:
        data = data[0]

    # ---- convert to mJy/beam ----
    data = data * 1e3
    wcs = WCS(header)

    # better contrast scaling
    vmin, vmax = PercentileInterval(98).get_limits(data)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(projection=wcs)

    im = ax.imshow(data, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)

    ax.set_title(f"{source} (LoTSS DR3)")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")

    # ---- 32 arcsec diameter circle ----
    radius_arcsec = 16.0
    radius_deg = radius_arcsec / 3600.0

    pixscale = proj_plane_pixel_scales(wcs)[0] * u.deg
    radius_pix = (radius_deg * u.deg / pixscale).decompose().value

    ny, nx = data.shape
    center = (nx / 2, ny / 2)

    circ = Circle(center, radius_pix, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(circ)

    # ---- Beam in lower-left corner ----
    if all(k in header for k in ['BMAJ', 'BMIN']):
        bmaj = header['BMAJ']  # deg
        bmin = header['BMIN']  # deg
        bpa = header.get('BPA', 0.0)  # deg

        pixscale = proj_plane_pixel_scales(wcs)[0]  # deg/pixel

        beam_width = bmin / pixscale
        beam_height = bmaj / pixscale

        ny, nx = data.shape

        beam_x = 0.1 * nx
        beam_y = 0.1 * ny

        beam = Ellipse((beam_x, beam_y), width=beam_width, height=beam_height, angle=-bpa, edgecolor='white', facecolor='white', linewidth=1)
        ax.add_patch(beam)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("mJy/beam")

    plt.show()
    hdul.close()


###
#   Function to connect to SDSS database and query data from the MAnGA survey
###
def get_SDSS_data(query):
    res = SDSS.query_sql(query)
    t = Table(res)


###
#   Function to reorganize simulations file in the timesteps and frequencies we want to further analysis
###
def reorganize_by_timestep(master_folder, output_folder, remove_freqs=None):

    master_folder = Path(master_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    remove_freqs = set(remove_freqs or [])

    name_re = re.compile(r'(?P<prefix>.+?)_\d+_combined_nu=(?P<freq>\d+(?:\.\d+)?)_los=.*?_(?P<timestep>\d+(?:\.\d+)?)Myr')

    # Loop over every frequency directory
    for freq_dir in master_folder.iterdir():

        if not freq_dir.is_dir():
            continue

        # Loop over FITS files in that directory
        for file in freq_dir.glob("*.fits*"):

            m = name_re.match(file.stem.replace(".fits", ""))
            if m is None:
                continue

            prefix = m.group("prefix")
            freq = float(m.group("freq"))
            timestep = float(m.group("timestep"))

            
            if freq in remove_freqs:
                continue

            folder = output_folder / f"{prefix}_{round(timestep)}Myr"
            folder.mkdir(exist_ok=True)

            shutil.copy2(file, folder / file.name)


###
#   Function to calculate the radio luminosity and the jet power 
###
def luminosity(z, S_jy, alpha=0.8, kcorr=True):
    cosmo = FlatLambdaCDM(H0=67.8, Om0=0.3)

    DL = cosmo.luminosity_distance(z).to(u.m).value
    # Radio Luminosity (W/Hz/sr )
    L = DL**2 * S_jy*1e-26                              

    if kcorr:
        L *= (1 + z)**(alpha - 1)

    # Jet power (erg/s)
    Q_jet = 5e44 * (L / 1e26)**0.64

    return Q_jet, L


###
#   Function to calculate spectral index with uncertainty using a MC simulation
###
def spec_index_MC(source_name, radio_maps_path, region_path, freq1, freq2, method, freq_tol=0.01, method_flux='imstat'):
    ###
    #   Open and save the radio maps according to the method and frequencies used
    ###

    # Open all .fits files in the directory
    full_path = os.path.abspath(radio_maps_path)
    all_files  = glob.glob(os.path.join(radio_maps_path, "*"))

    # Make sure freq_lo <= freq_hi regardless of the order passed in
    freq1_hz = freq1 * 1e6
    freq2_hz = freq2 * 1e6
    freq_lo, freq_hi = sorted([freq1_hz, freq2_hz])

    selected_files = []   # list of (file, frequency) tuples

    for file in all_files:
        dt_clean = dt.now().isoformat().replace('-','_').replace(':','_').replace('.','_')
        tmp_orig = 'tmp_orig_image' + dt_clean

        try:
            importfits(fitsimage=file, imagename=tmp_orig, overwrite=True)
            imhead_val = imhead(imagename=tmp_orig, mode='list')

            # Read the reference frequency (Hz) from the frequency axis
            nu = None
            for i in range(1, 5):
                if str(imhead_val.get(f'ctype{i}', '')).lower() == 'frequency':
                    nu = float(imhead_val[f'crval{i}'])
                    break
            if nu is None:
                nu = float(imhead_val['restfreq'][0]) 

            # Only matches the two frequency selected
            if method == 'two_freq':
                matches_f1 = abs(nu - freq1_hz) / freq1_hz < freq_tol
                matches_f2 = abs(nu - freq2_hz) / freq2_hz < freq_tol
                if matches_f1 or matches_f2:
                    selected_files.append((file, nu))

            # Matches all the frequency maps between frequencies
            elif method == 'multi_freq':
                if freq_lo * (1 - freq_tol) <= nu <= freq_hi * (1 + freq_tol):
                    selected_files.append((file, nu))

            else:
                raise ValueError(f"Unknown method '{method}'. Use 'two_freq' or 'multi_freq'.")

        except Exception as e:
            warnings.warn(f"Failed to process '{os.path.basename(file)}': {e}", UserWarning)

        finally:
            if os.path.exists(tmp_orig):
                shutil.rmtree(tmp_orig)

    # Sort by frequency
    selected_files.sort(key=lambda x: x[1])

    print(f"Selected {len(selected_files)} file(s) for method '{method}':")
    for f, nu in selected_files:
        print(f"  {os.path.basename(f)}  ({nu/1e9:.3f} GHz)")


    ###
    #  Convert the DS9 .reg file to CRTF once
    ###
    regs = Regions.read(region_path, format='ds9')

    # Use the WCS of the first selected map to put everything in sky coordinates
    with fits.open(selected_files[0][0]) as hdul:
        wcs = WCS(hdul[0].header).celestial

    sky_regs = []
    for r in regs:
        if isinstance(r, PixelRegion):
            r = r.to_sky(wcs)
        if isinstance(r, SkyRegion):
            sky_regs.append(r)
        else:
            warnings.warn(f"Skipping unsupported region: {r}", UserWarning)

    tmp_dir = tempfile.mkdtemp(prefix='crtf_')
    region_crtf = os.path.join(tmp_dir, 'region.crtf')
    Regions(sky_regs).write(region_crtf, format='crtf', overwrite=True)


    ###
    #  Measuring the flux either by imfit or imstat 
    ###
    nu, fluxX, fluxXerr = [], [], []

    for final_map, _ in selected_files:
        dt_clean = dt.now().isoformat().replace('-','_').replace(':','_').replace('.','_')
        tmp_rebin = 'tmp_rebin_image' + dt_clean

        try:
            importfits(fitsimage=final_map, imagename=tmp_rebin, overwrite=True)

            ### Imfit - Most of my sources it does not work ###
            if method_flux == 'imfit':
                # Fit the image inside the CRTF region
                fit_results = imfit(imagename=tmp_rebin, region=region_crtf)
                comp = fit_results['results']['component0']

                nu.append(fit_results['results']['component0']['spectrum']['frequency']['m0']['value'])
                fluxX.append(fit_results['results']['component0']['flux']['value'][0])
                fluxXerr.append(fit_results['results']['component0']['flux']['error'][0])

            ### Imstat - We are already using the same region to run BRATS on all the pixesl since the have good S/N
            elif method_flux == 'imstat':
                fractional_error = 0.10 # Set from my previous paper, check Sequeira-Murillo et al 2026
                ia.open(tmp_rebin)
                beamarea = ia.beamarea()['pixels']
                imsum = ia.summary()
                ia.close()

                # Integrated flux (sum / beam area) inside the region
                stats = imstat(imagename=tmp_rebin, region=region_crtf)
                flux  = stats['flux'][0]
                npts  = stats['npts'][0]

                # Noise from the whole image, with source pixels clipped out (Jy/beam)
                rms = imstat(imagename=tmp_rebin, algorithm='chauvenet')['rms'][0]

                stat_err = rms * np.sqrt(max(npts / beamarea, 1.0))
                flux_err = np.sqrt(stat_err**2 + (fractional_error * flux)**2)

                nu.append(imsum['refval'][-1] if (imsum['refval'][-1] != 1.0) else imsum['refval'][-2])
                fluxX.append(flux)
                fluxXerr.append(flux_err)

        except Exception as e:
            warnings.warn(f"imfit failed for '{os.path.basename(final_map)}': {e}", UserWarning)

        finally:
            if os.path.exists(tmp_rebin):
                shutil.rmtree(tmp_rebin)


    ###
    #  Calculation of the Spectral Index
    ###
    # We normalised frequency, the spectral index is unaffected
    nu       = np.array(nu)
    fluxX    = np.array(fluxX)
    fluxXerr = np.array(fluxXerr)

    x  = nu / nu.min()              
    p0 = [fluxX[0], 0.8]

    (p, p_conv) = optimize.curve_fit(powerlaw2, x, fluxX, p0=p0, sigma=fluxXerr,  absolute_sigma=True)
    
    # MC bootstrap
    rng = np.random.default_rng()
    allpnew = []
    for i in range(1000):
        newfluxX = fluxX + fluxXerr * rng.standard_normal(len(fluxX))
        try:
            (pnew, pnew_conv) = optimize.curve_fit(powerlaw2, nu, newfluxX, sigma=fluxXerr, absolute_sigma=True)
            allpnew.append(pnew[1])
        except:
            continue

    allpnew = np.array(allpnew)
    alpha = p[1]
    alpha_err = np.std(allpnew)
    print(f"alpha = {alpha:.3f} ± {alpha_err:.3f}  ({len(allpnew)}/1000 fits converged)")

    # Histogram
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(allpnew, bins=40, density=True)

    x1 = np.linspace(allpnew.min(), allpnew.max(), 1000)
    ax.plot(x1, norm.pdf(x1, loc=np.mean(allpnew), scale=alpha_err), 'r-', lw=3)

    ax.set_xlim(allpnew.min(), allpnew.max())
    ax.set_xlabel(r'$\alpha$')
    ax.set_title(rf'$\sigma$ = {alpha_err:.3f}')
    ax.text(.1, .85, source_name, transform=ax.transAxes)

    fig.tight_layout()
    plt.show()
    
    return nu, fluxX, fluxXerr, alpha, alpha_err


''' 
source_name = 'L45'
radio_map_path = '/Users/fsequeira15/Library/CloudStorage/OneDrive-Personal/UW-Madison/Brats_Ages/Brats_Folder/FitsMaps/Jennifer_sims/smooth_files/brats_results/L45_M5_b1m/L45_M5_b1m_16Myr/noise'
region_path = '/Users/fsequeira15/Library/CloudStorage/OneDrive-Personal/UW-Madison/Brats_Ages/Brats_Folder/FitsMaps/Jennifer_sims/smooth_files/brats_results/L45_M5_b1m/L45_M5_b1m_16Myr/regions/L45_M5_b1m_outer_path.reg'

nu, flux, fluxerr, alpha, error = spec_index_MC(source_name, radio_maps_path=radio_map_path, region_path=region_path, freq1=395, freq2=1400, method='multi_freq')
'''


ra  = 69.2679
dec = 29.6703

#"3C123": ("69.2679", "29.6703", 0.218), 
#"3C320": ("232.8558", "35.5608", 0.342), 
#"3C338": ("247.1592", "39.5511", 0.030), 
#"3C438": ("328.9675", "38.0078", 0.290), 
#"3C444": ("333.6063", "-17.0272", 0.153)} 

#view_cutout(source="3C123", ra=ra, dec=dec, size=5)


'''
###
#   Example on how to run the functions above
###

manga_ids = ["1-548033","1-378360","1-378910","1-71481","1-548196", "1-604860","1-121737","1-378770","1-548622","1-72322","1-47705","1-48760","1-217041","1-394382",
             "1-255890","1-279314","1-419533","1-419288","1-260541","1-569263","1-207943","1-314484","1-373327","1-314374","1-236745","1-91582","1-569162","1-210614",
             "1-134884","1-547290","1-114171",]

# Get information of the sources from MAnGA survey
for mangaid in manga_ids:
    query = f"SELECT mangaid, objra, objdec FROM MangaDrpAll WHERE mangaid = '{mangaid}'"
    t = get_SDSS_data(query)

# Convert coordinates
ra, dec = 112.0034403, 41.91876835
x1, y1,  = deg_to_sexagesimal(ra, dec)
pos_val = str(x1)+' '+str(y1)
print(pos_val)


#master_folder = '/Users/fsequeira15/Library/CloudStorage/OneDrive-Personal/UW-Madison/Brats_Ages/Brats_Folder/FitsMaps/Yihao_sims/original_files/L446_rc100_beta07/combined_lobe/fits_obs'
#output_folder = '/Users/fsequeira15/Library/CloudStorage/OneDrive-Personal/UW-Madison/Brats_Ages/Brats_Folder/FitsMaps/Yihao_sims/org_sims'
#remove_freqs = [100, 215.4, 350, 464.2, 610, 2154.4, 5000,8000]

#reorganize_by_timestep(master_folder, output_folder, remove_freqs)
'''


'''
###
#   Example on how to modify the FITS header to remove the DATE-OBS keyword
###
from astropy.io import fits

fits_file = "/Users/fsequeira15/Library/CloudStorage/OneDrive-Personal/UW-Madison/Brats_Ages/Brats_Folder/FitsMaps/Andrea_maps_original_resolution/Original_maps_revised/1-378360/1-378360_first_.fits"

with fits.open(fits_file, mode='update') as hdul:
    header = hdul[0].header
    if 'DATE-OBS' in header:
        del header['DATE-OBS']
        print("[fits] Deleted DATE-OBS keyword")
    else:
        print("[fits] DATE-OBS not present, nothing to delete")
    # mode='update' + context manager exit will flush changes to disk automatically
'''