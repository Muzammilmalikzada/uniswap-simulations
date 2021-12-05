import numpy as np


# Return a single realization of GBM
def gbm(mu, sigma, s0, dt, n):
    s = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), n).T
    )
    s = np.concatenate(([1], s))
    s = s0 * s.cumprod(axis=0)
    return s    


# Uniswap v2 LP fee simulation
# Given a price movement and initial deposit amounts, simulate fees collected assuming
# arbitrage only. 
def simulate_v2_fees(s, x, y, f):
    k = x * y
    initial_value = x * s[0] + y
    hold_value = x * s[-1] + y
    lp_prev_price = s[0]
    arb_count = [0,0]
    lp_fees_collected = [0,0]

    for p in s[1:]:
        # if out of range, arb and collect fees
        if p < (1-f)*lp_prev_price or lp_prev_price/(1-f) < p:
            if p < (1-f)*lp_prev_price:
                arb_count[0] += 1
            else:
                arb_count[1] += 1
            # determine amount to arb
            new_x = np.sqrt(k/p)
            new_y = k/new_x
            # collect fees
            # note that fees are collected before the trade
            if new_x > x:
                lp_fees_collected[0] += f/(1-f) * (new_x -x)
            elif new_y > y:
                lp_fees_collected[1] += f/(1-f) * (new_y - y)
            x = new_x
            y = new_y
            lp_prev_price = p

    total_fees_collected = lp_fees_collected[0] * s[-1] + lp_fees_collected[1]
    portfolio_value = x * s[-1] + y
    il = (portfolio_value + total_fees_collected - hold_value)/initial_value
    return total_fees_collected, portfolio_value, hold_value, il, arb_count[0], arb_count[1]


# Uniswap V2 LP fee simulation assuming fees are reallocated to the pool
def simulate_v2_fees_reallocation(s, x, y, f):
    k = x * y
    hold_value = x * s[-1] + y
    lp_prev_price = s[0]
    arb_count = [0,0]
    lp_fees_collected = [0,0]

    for p in s[1:]:
        # if out of range, arb and collect fees
        if p < (1-f)*lp_prev_price or lp_prev_price/(1-f) < p:
            if p < (1-f)*lp_prev_price:
                arb_count[0] += 1
            else:
                arb_count[1] += 1
            # determine amount to arb
            new_x = np.sqrt(k/p)
            new_y = k/new_x
            # collect fees
            # note that fees are collected before the trade
            if new_x > x:
                swap_amount = 1/(1-f) * (new_x - x)
                lp_fees_collected[0] += f * swap_amount
                x += swap_amount
                y = new_y
                # update k as we include fees
                k = x * y
            elif new_y > y:
                swap_amount = 1/(1-f) * (new_y - y)
                lp_fees_collected[1] += f * swap_amount
                x = new_x
                y += swap_amount
                # update k as we include fees
                k = x * y
            lp_prev_price = p

    total_fees_collected = lp_fees_collected[0] * s[-1] + lp_fees_collected[1]
    portfolio_value = x * s[-1] + y
    il = (portfolio_value + total_fees_collected - hold_value)/hold_value
    return total_fees_collected, portfolio_value, hold_value, il, arb_count[0], arb_count[1]


def simulate_over_time_ranges(trials, output_filename):
    # GBM parameters
    mus = [0]
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    s0 = 1
    ns = [3000]
    dt = 0.0001

    # Uniswap v2 parameters
    x = 0.5
    y = 0.5
    fees = np.arange(0.001, 1.001, 0.001)

    f = open(output_filename, "w")
    f.write(f"mu,sigma,s0,n,dt,x,y,fee,mean,p5,p50,p95,mean_il,p5_il,p95_il,mean_pv,p5_pv,p95_pv,mean_hv,p5_hv,p95_hv,mean_lower,p5_lower,p95_lower,mean_upper,p5_upper,p95_upper\n")

    for mu in mus:
        for sigma in sigmas:
            for fee in fees:
                for n in ns:
                    print(f"simulating with params mu={mu} sigma={sigma} fee={fee} n={n}")
                    collected_fees = list()
                    collected_pv = list()
                    collected_hv = list()
                    collected_il = list()
                    lowers = list()
                    uppers = list()
                    for _ in range(trials):
                        s = gbm(mu, sigma, s0, dt, n)
                        fs, pv, hv, il, lower, upper = simulate_v2_fees_reallocation(s, x, y, fee)
                        collected_il.append(il)
                        collected_fees.append(fs)
                        collected_pv.append(pv)
                        collected_hv.append(hv)
                        lowers.append(lower)
                        uppers.append(upper)
                    mean = np.mean(collected_fees)
                    p5 = np.quantile(collected_fees, 0.05, interpolation='nearest')
                    p50 = np.quantile(collected_fees, 0.5, interpolation='nearest')
                    p95 = np.quantile(collected_fees, 0.95, interpolation='nearest')

                    mean_il = np.mean(collected_il)
                    p5_il = np.quantile(collected_il, 0.05, interpolation='nearest')
                    p95_il = np.quantile(collected_il, 0.95, interpolation='nearest')

                    mean_pv = np.mean(collected_pv)
                    p5_pv = np.quantile(collected_pv, 0.05, interpolation='nearest')
                    p95_pv = np.quantile(collected_pv, 0.95, interpolation='nearest')

                    mean_hv = np.mean(collected_hv)
                    p5_hv = np.quantile(collected_hv, 0.05, interpolation='nearest')
                    p95_hv = np.quantile(collected_hv, 0.95, interpolation='nearest')

                    mean_lowers = np.mean(lowers)
                    p5_lowers = np.quantile(lowers, 0.05, interpolation='nearest')
                    p95_lowers = np.quantile(lowers, 0.95, interpolation='nearest')
                    
                    mean_uppers = np.mean(uppers)
                    p5_uppers = np.quantile(uppers, 0.05, interpolation='nearest')
                    p95_uppers = np.quantile(uppers, 0.95, interpolation='nearest')

                    f.write(f"{mu},{sigma},{s0},{n},{dt},{x},{y},{fee},{mean},{p5},{p50},{p95},{mean_il},{p5_il},{p95_il},{mean_pv},{p5_pv},{p95_pv},{mean_hv},{p5_hv},{p95_hv},{mean_lowers},{p5_lowers},{p95_lowers},{mean_uppers},{p5_uppers},{p95_uppers}\n")


simulate_over_time_ranges(500, "data/v2_simulations_il.csv")
