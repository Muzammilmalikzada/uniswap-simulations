import numpy as np

# taken from https://stackoverflow.com/questions/53386933/how-to-solve-fit-a-geometric-brownian-motion-process-in-python

# dt = 9.513e-6 ~= 0.00001 for 5m
# dt = 1.142e-4 ~= 0.0001 for 1h
dt5m = 5 / (365 * 24 * 60)
dt1h = 1 / (365 * 24)
dts = {
    "5m": dt5m,
    "1h": dt1h,
}


def binance_data(coin, granularity, start=None, end=None):
    import csv
    import glob
    import sys

    if start and end:
        assert start < end

    print(f"Grabbing Binance data for {coin}-{granularity} between {start}-{end}")
    filenames = glob.glob(f"binance/prices/*-{coin}-{granularity}-*.csv")
    assert len(filenames) > 0, f"No matches for {coin}-{granularity}"
    assert len(filenames) == 1, f"Too many matches for {coin}-{granularity}"

    reader = csv.reader(open(filenames[0], "r"))
    prices = list()
    times = list()
    for l in reader:
        # skip if out of range
        ts = int(l[0])
        if (start and ts < start) or (end and ts > end):
            continue
        times.append(ts) 
        prices.append(float(l[1]))

    return prices, times


def simple_estimate_mu(series, dt):
    n = len(series)
    T = dt * n
    return (series[-1] - series[0]) / T


# Use all the increments combined (maximum likelihood estimator)
# Note this is for a linear drift-diffusion process, i.e. the log of GBM
def incremental_estimate_mu(series, dt):
    n = len(series)
    T = dt * n
    ts = np.linspace(dt, T, n)
    total = (1.0 / dt) * (ts**2).sum()
    return (1.0 / total) * (1.0 / dt) * ( ts * series ).sum()


# This just estimates the sigma by its definition as the infinitesimal variance (simple Monte Carlo)
# Note this is for a linear drift-diffusion process, i.e. the log of GBM
# One can do better than this of course (MLE?)
def estimate_sigma(series, dt):
    # n = len(series)
    # return np.sqrt( ( np.diff(series)**2 ).sum() / (n * dt) )

    series = np.array(series)
    n = len(series)
    ratio = series[1:]/series[:-1]
    return np.std(ratio) / np.sqrt(dt)
    

# Since log-GBM is a linear Ito drift-diffusion process (scaled Wiener process with drift), we
# take the log of the realizations, compute mu and sigma, and then translate the mu and sigma
# to that of the GBM (instead of the log-GBM). (For sigma, nothing is required in this simple case).
def gbm_drift(log_mu, log_sigma):
    return log_mu + 0.5 * log_sigma**2


# window is dict of Arrow date range, e.g. {month: 2}
# step is string, e.g. "week"
def estimate_parameters_for_coins(coin_granularities, window, step):
    import arrow

    for coin, gran, invert in coin_granularities:
        print(f"Estimating GBM parameters for {coin}-{gran} in a {window} window with step {step}:")
        prices, times = binance_data(coin, gran)
        if invert:
            prices = list(map(lambda x: 1/x, prices))
        # take log of prices
        # prices = np.log(prices)
        start = arrow.get(times[0])

        neg_window = dict()
        for k in window:
            neg_window[k] = -window[k]
        end = arrow.get(times[-1]).shift(**neg_window)

        mu1s = list()
        mu2s = list()
        sigmas = list()
        for r in arrow.Arrow.range(step, start, end):
            t1 = r.timestamp() * 1000
            t2 = r.shift(**window).timestamp() * 1000
            idx1 = np.argmin(abs(np.array(times) - t1))
            idx2 = np.argmin(abs(np.array(times) - t2))

            assert idx2 > idx1
            mu1 = simple_estimate_mu(prices[idx1:idx2], dts[gran])
            sigma = estimate_sigma(prices[idx1:idx2], dts[gran])
            sigmas.append(sigma)
            mu1s.append(gbm_drift(mu1, sigma))
        mu1 = simple_estimate_mu(prices, dts[gran])
        s = estimate_sigma(prices, dts[gran])
        mu2 = gbm_drift(incremental_estimate_mu(prices, dts[gran]), s)
        print(f"Simple mu estimate: {mu1s}\navg:{np.average(mu1s)}")
        print(f"Windowed sigma estimate: {sigmas}")
        print(f"Full window simple mu estimate: {mu1}")
        print(f"Incremental mu estimate: {mu2}")
        print(f"Full window sigma estimate: {s}")

cgs = [
    ("ETHUSDT", "5m", False),
    ("ETHUSDT", "1h", False),
    ("WBTCETH", "5m", False),
    ("WBTCETH", "1h", False),
    ("LINKETH", "5m", True),
    ("LINKETH", "1h", True),
    ("USDCUSDT", "5m", False),
    ("USDCUSDT", "1h", False),
]
window = dict(months=3)
step = "week"

estimate_parameters_for_coins(cgs, window, step)