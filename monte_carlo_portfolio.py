#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
monte_carlo_portfolio.py - Monte Carlo Risk Simulator for Stock Portfolios.

Simulates correlated Geometric Brownian Motion (GBM) for multi-asset stock portfolios,
reporting horizon-specific 95% VaR, 95% CVaR (Expected Shortfall), and probability of loss.
"""

import argparse
from dataclasses import dataclass
import os
import sys
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADING_DAYS = 252


# ==============================================================================
# DATA FETCHING
# ==============================================================================
def load_log_returns(tickers: list[str], lookback: str = "3y") -> pd.DataFrame:
    """Daily log returns from auto-adjusted Yahoo Finance closes."""
    import yfinance as yf

    print(f"Downloading historical data for: {', '.join(tickers)}...")
    raw = yf.download(list(tickers), period=lookback, auto_adjust=True, progress=False)

    if raw.empty:
        raise SystemExit("No data returned from Yahoo Finance. Check your internet connection or tickers.")

    data = raw["Close"] if "Close" in raw.columns else raw
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])

    missing = [t for t in tickers if t not in data.columns or data[t].isna().all()]
    if missing:
        raise SystemExit(
            f"No data returned for: {', '.join(missing)}. "
            "Check ticker symbols (Yahoo format, e.g., APN.L)."
        )

    data = data[list(tickers)]  # preserve exact order
    returns = np.log(data / data.shift(1)).dropna()
    if len(returns) < 60:
        raise SystemExit(f"Only {len(returns)} overlapping daily returns - too few to estimate risk.")
    return returns


def demo_log_returns(tickers: list[str], n_days: int = 3 * TRADING_DAYS, seed: int = 0) -> pd.DataFrame:
    """Synthetic correlated returns so the script runs offline if needed."""
    rng = np.random.default_rng(seed)
    k = len(tickers)
    vols = np.linspace(0.20, 0.35, k) / np.sqrt(TRADING_DAYS)
    corr = np.full((k, k), 0.4)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    data = rng.multivariate_normal(np.full(k, 0.0003), cov, size=n_days)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame(data, index=idx, columns=list(tickers))


# ==============================================================================
# SIMULATION ENGINE & VALIDATION
# ==============================================================================
def normalise_weights(weights: list[float] | np.ndarray | None, n: int,
                      initial_capital: float | None = None) -> tuple[np.ndarray, float]:
    """
    Equal weight by default. Positive numbers are scaled to sum to 1.
    Returns (normalized_weights, resolved_initial_capital).
    """
    if weights is None:
        w = np.full(n, 1.0 / n)
        cap = initial_capital if initial_capital is not None else 10_000.0
        return w, cap

    w = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError("Weights must contain only finite numbers (no NaN or Inf).")
    if len(w) != n:
        raise ValueError(f"Got {len(w)} weights for {n} tickers.")
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("Weights must be non-negative and not all zero.")
    
    w_sum = float(w.sum())
    is_unit_sum = np.isclose(w_sum, 1.0, atol=1e-5)

    if initial_capital is not None:
        cap = initial_capital
    else:
        if is_unit_sum:
            cap = 10_000.0
        else:
            raise ValueError(
                f"Provided weights sum to {w_sum:.4f} (!= 1.0) with no explicit capital supplied. "
                "To avoid ambiguity between cash amounts and un-normalized fractions, "
                "please specify initial_capital (or CLI --capital) explicitly."
            )

    return w / w_sum, cap


def _repair_cov_psd(cov_daily: np.ndarray) -> tuple[np.ndarray, bool]:
    """Ensure covariance matrix is positive semi-definite via scale-relative eigenvalue clipping."""
    try:
        np.linalg.cholesky(cov_daily)
        return cov_daily, False
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov_daily)
        max_val = np.max(vals)
        rel_floor = max(1e-10 * max_val, 1e-14)
        vals = np.maximum(vals, rel_floor)
        cov_repaired = vecs @ np.diag(vals) @ vecs.T
        return cov_repaired, True


@dataclass
class SimulationResult:
    tickers: list[str]
    weights: np.ndarray
    initial_capital: float
    annual_return: float
    years: float
    currency: str
    vols: pd.Series
    corr: pd.DataFrame
    paths: np.ndarray          # Shape: (n_sims, n_days + 1)
    psd_repaired: bool = False

    @property
    def final_values(self) -> np.ndarray:
        return self.paths[:, -1]

    @property
    def var_95(self) -> float:
        """Absolute 95% Value at Risk at the end of the specified horizon (V0 - P5)."""
        return self.initial_capital - np.percentile(self.final_values, 5)

    @property
    def cvar_95(self) -> float:
        """Absolute 95% Conditional VaR / Expected Shortfall at the end of the specified horizon."""
        tail = self.final_values[self.final_values <= np.percentile(self.final_values, 5)]
        return self.initial_capital - tail.mean()

    def asset_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Weight": self.weights,
            "Assumed Ann. Return (Simple)": self.annual_return,
            "Ann. Volatility": self.vols.values,
        }, index=self.tickers)

    def summary(self) -> pd.Series:
        fv, c, y = self.final_values, self.currency, self.years
        h_label = f"{y:g}y" if y != 1 else "1y"
        return pd.Series({
            "Portfolio Strategy": "Buy-and-Hold Total Return (Embedded Reinvestment)",
            "Simulation Horizon": f"{y:g} Year(s) ({int(y * TRADING_DAYS)} trading days)",
            "Starting Portfolio Value": f"{c}{self.initial_capital:,.2f}",
            "Expected Value (Mean at Horizon)": f"{c}{fv.mean():,.2f}",
            "Median Final Value at Horizon": f"{c}{np.median(fv):,.2f}",
            "5th Percentile Outcome at Horizon": f"{c}{np.percentile(fv, 5):,.2f}",
            f"95% Value at Risk ({h_label} Horizon VaR)": f"{c}{self.var_95:,.2f}",
            f"95% Conditional VaR ({h_label} Horizon CVaR)": f"{c}{self.cvar_95:,.2f}",
            f"Probability of Loss at Horizon ({h_label})": f"{(fv < self.initial_capital).mean():.1%}",
        })


def simulate_portfolio(log_returns: pd.DataFrame, weights: list[float] | None = None,
                       initial_capital: float | None = None, annual_return: float = 0.09,
                       years: float = 1.0, n_sims: int = 10_000, seed: int = 42,
                       currency: str = "$") -> SimulationResult:
    """Correlated GBM Monte Carlo simulation of a buy-and-hold portfolio."""
    if log_returns is None or not isinstance(log_returns, pd.DataFrame) or log_returns.empty or log_returns.shape[1] == 0:
        raise ValueError("Return DataFrame is empty or contains no asset columns.")
    if not np.isfinite(log_returns.to_numpy()).all():
        raise ValueError("Return DataFrame contains missing, NaN, or infinite (Inf) values.")
    if len(log_returns) < 30:
        raise ValueError(f"Insufficient return observations ({len(log_returns)}). Minimum 30 observations required.")

    tickers = list(log_returns.columns)
    w, cap = normalise_weights(weights, len(tickers), initial_capital)

    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("Initial capital must be a strictly positive finite number.")
    if not np.isfinite(annual_return) or annual_return <= -1.0:
        raise ValueError("Annual return drift must be a finite number strictly greater than -1.0 (-100%).")
    if not np.isfinite(years) or years <= 0:
        raise ValueError("Years horizon must be a strictly positive finite number.")

    n_days = int(years * TRADING_DAYS)
    if n_days < 1:
        raise ValueError(
            f"Simulation horizon (years={years:g}) is shorter than 1 trading day "
            f"({1/TRADING_DAYS:.5f} years / ~{1/TRADING_DAYS*365:.1f} days). "
            "Please specify a horizon of at least 1 trading day."
        )

    if n_sims <= 0:
        raise ValueError("Number of simulations must be a strictly positive integer.")

    vols = log_returns.std() * np.sqrt(TRADING_DAYS)
    if (vols <= 0).any() or not np.all(np.isfinite(vols)):
        zero_assets = list(log_returns.columns[vols <= 0])
        raise ValueError(f"Assets with zero or invalid volatility detected: {zero_assets}. All assets must have positive volatility.")

    corr = log_returns.corr()
    dt = 1.0 / TRADING_DAYS

    cov_daily = np.outer(vols, vols) * corr.values * dt
    cov_daily, psd_repaired = _repair_cov_psd(cov_daily)

    L = np.linalg.cholesky(cov_daily)

    mu_log = np.log1p(annual_return)
    mu_daily = (mu_log - 0.5 * vols.values ** 2) * dt

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_sims, n_days, len(tickers)))
    daily_log = mu_daily + Z @ L.T

    growth = np.exp(np.cumsum(daily_log, axis=1))
    growth = np.pad(growth, ((0, 0), (1, 0), (0, 0)), constant_values=1.0)
    paths = cap * (growth * w).sum(axis=2)

    return SimulationResult(tickers, w, cap, annual_return, years,
                            currency, vols, corr, paths, psd_repaired)


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================
def plot_fan_chart(res: SimulationResult, n_sample_paths: int = 25) -> plt.Figure:
    t = np.linspace(0, res.years, res.paths.shape[1])
    p5, p25, p50, p75, p95 = np.percentile(res.paths, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(t, res.paths[:n_sample_paths].T, color="grey", alpha=0.15, lw=0.7)
    ax.fill_between(t, p5, p95, alpha=0.20, label="5th-95th Percentile")
    ax.fill_between(t, p25, p75, alpha=0.35, label="25th-75th Percentile")
    ax.plot(t, p50, lw=2, label="Median Path")
    ax.axhline(res.initial_capital, color="black", ls="--", lw=1, label="Starting Capital")
    ax.set_xlabel("Years")
    ax.set_ylabel(f"Portfolio Value ({res.currency})")
    ax.set_title(f"Simulated Portfolio Trajectories over {res.years:g} Year(s) ({res.paths.shape[0]:,} Paths)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_final_distribution(res: SimulationResult) -> plt.Figure:
    fv, c, y = res.final_values, res.currency, res.years
    p5 = np.percentile(fv, 5)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(fv, bins=80, alpha=0.75, color="#1f77b4", edgecolor="none")
    ax.axvline(res.initial_capital, color="black", ls="--", label="Starting Capital")
    ax.axvline(fv.mean(), color="green", lw=1.5, label=f"Mean: {c}{fv.mean():,.0f}")
    ax.axvline(p5, color="crimson", lw=1.5, label=f"5th Percentile: {c}{p5:,.0f}")
    ax.axvspan(fv.min(), p5, color="crimson", alpha=0.12)
    ax.text(0.98, 0.95,
            f"Horizon: {y:g} Year(s)\n"
            f"VaR 95%: {c}{res.var_95:,.0f}\n"
            f"CVaR 95%: {c}{res.cvar_95:,.0f}\n"
            f"P(Loss): {(fv < res.initial_capital).mean():.1%}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_xlabel(f"Final Portfolio Value at {y:g}y Horizon ({c})")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Final Portfolio Outcomes at {y:g}-Year Horizon")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(res: SimulationResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(1.2 * len(res.tickers) + 3, 1.0 * len(res.tickers) + 2.5))
    im = ax.imshow(res.corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(res.tickers)))
    ax.set_yticks(range(len(res.tickers)))
    ax.set_xticklabels(res.tickers, rotation=45, ha="right")
    ax.set_yticklabels(res.tickers)
    for i in range(len(res.tickers)):
        for j in range(len(res.tickers)):
            ax.text(j, i, f"{res.corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Historical Correlation Matrix (Daily Log Returns)")
    fig.tight_layout()
    return fig


def make_plots(res: SimulationResult, save_dir: str | None = None):
    figs = {
        "fan_chart": plot_fan_chart(res),
        "final_distribution": plot_final_distribution(res),
        "correlation_heatmap": plot_correlation_heatmap(res),
    }
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            path = os.path.join(save_dir, f"{name}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Saved figure to: {path}")
    else:
        plt.show()
    return figs


# ==============================================================================
# UNIT TESTS
# ==============================================================================
class TestMonteCarlo(unittest.TestCase):
    def test_weight_normalization_equal(self):
        w, cap = normalise_weights(None, 3)
        np.testing.assert_allclose(w, [1/3, 1/3, 1/3])
        self.assertEqual(cap, 10000.0)

    def test_weight_normalization_cash_with_explicit_capital(self):
        w, cap = normalise_weights([1190, 1100, 630], 3, initial_capital=2920.0)
        np.testing.assert_allclose(w, [1190/2920, 1100/2920, 630/2920])
        self.assertEqual(cap, 2920.0)

    def test_non_unit_weight_without_capital_raises(self):
        with self.assertRaises(ValueError):
            normalise_weights([0.6, 0.6], 2)
        with self.assertRaises(ValueError):
            normalise_weights([0.2, 0.3, 0.4], 3)

    def test_expected_return_compounding_behavior(self):
        df_single = demo_log_returns(["KO"])
        res = simulate_portfolio(df_single, initial_capital=10000, annual_return=0.10, years=2, n_sims=10000)
        self.assertEqual(res.initial_capital, 10000)
        expected_target = 10000 * ((1 + 0.10) ** 2)
        self.assertAlmostEqual(res.final_values.mean(), expected_target, delta=150.0)

    def test_horizon_less_than_one_day_rejection(self):
        df = demo_log_returns(["KO", "AMZN"])
        with self.assertRaises(ValueError):
            simulate_portfolio(df, years=0.001)

    def test_invalid_weights_nan_or_negative(self):
        with self.assertRaises(ValueError):
            normalise_weights([np.nan, 1.0, 2.0], 3)
        with self.assertRaises(ValueError):
            normalise_weights([-1.0, 2.0, 3.0], 3)

    def test_infinite_and_nan_return_rejection(self):
        df = demo_log_returns(["KO"])
        df.iloc[0, 0] = np.inf
        with self.assertRaises(ValueError):
            simulate_portfolio(df)

    def test_invalid_return_drift(self):
        df = demo_log_returns(["KO"])
        with self.assertRaises(ValueError):
            simulate_portfolio(df, annual_return=-1.0)

    def test_scale_relative_cov_repair(self):
        cov_singular = np.ones((3, 3)) * 0.0004
        cov_rep, repaired = _repair_cov_psd(cov_singular)
        self.assertTrue(repaired)
        L = np.linalg.cholesky(cov_rep)
        self.assertEqual(L.shape, (3, 3))


def run_unit_tests():
    print("\nRunning Monte Carlo Portfolio Unit Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMonteCarlo)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


# ==============================================================================
# MAIN PROGRAM LOGIC
# ==============================================================================
def main():
    p = argparse.ArgumentParser(description="Monte Carlo Risk Simulation for Stock Portfolios.")
    p.add_argument("tickers", nargs="*", help="Yahoo Finance tickers, e.g., KO AMZN APN.L")
    p.add_argument("--weights", type=float, nargs="+", help="Cash amounts or fractions per ticker")
    p.add_argument("--capital", type=float, default=None, help="Starting portfolio value (default: 10000 if weights sum to 1.0)")
    p.add_argument("--currency", default="$", help="Symbol used in report")
    p.add_argument("--lookback", default="3y", help="Historical period (1y, 3y, 5y)")
    p.add_argument("--years", type=float, default=1.0, help="Simulation horizon in years")
    p.add_argument("--sims", type=int, default=10_000, help="Number of simulated paths")
    p.add_argument("--return", dest="annual_return", type=float, default=0.09, help="Assumed simple annual return drift (default 0.09 = 9%%)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo", action="store_true", help="Use synthetic returns (no internet)")
    p.add_argument("--plot", action="store_true", help="Show interactive charts")
    p.add_argument("--save-plots", metavar="DIR", help="Save charts as PNGs in DIR")
    p.add_argument("--test", action="store_true", help="Run built-in unit tests")
    a = p.parse_args()

    # --- 1. RUN UNIT TESTS IF REQUESTED ---
    if a.test:
        run_unit_tests()
        return

    # --- 2. DEMO MODE DEFAULT TICKERS ---
    if a.demo and not a.tickers:
        a.tickers = ["AAA", "BBB", "CCC"]

    # --- 3. EASY IDE RUN CONFIGURATION (When pressing 'Run' in Spyder / VS Code) ---
    if not a.tickers and not a.demo:
        a.tickers = ["KO", "AMZN", "APN.L"]     # CHANGE YOUR TICKERS HERE
        portfolio_values = [1190, 1100, 630]     # CHANGE YOUR POSITION AMOUNTS HERE
        
        a.capital = sum(portfolio_values) if portfolio_values else 2920.0
        a.weights = portfolio_values
        a.lookback = "3y"
        a.years =3.0                             #CHANGE FOR DIFFERENT YEARS.
        a.sims = 10_000
        a.annual_return = 0.09
        a.plot = True                            # AUTOMATICALLY OPENS ALL 3 GRAPHS!

    # --- 4. INPUT VALIDATION ---
    if a.weights and len(a.weights) != len(a.tickers):
        p.error(f"Provided {len(a.weights)} weights/values for {len(a.tickers)} tickers.")
    if a.sims <= 0:
        p.error("--sims must be a strictly positive integer.")
    if a.years <= 0 or not np.isfinite(a.years):
        p.error("--years must be a strictly positive finite number.")
    if int(a.years * TRADING_DAYS) < 1:
        p.error(f"--years ({a.years:g}) is shorter than 1 trading day (~{1/TRADING_DAYS:.5f} years).")
    if not np.isfinite(a.annual_return) or a.annual_return <= -1.0:
        p.error("--return must be a finite number strictly greater than -1.0 (-100%).")

    # --- 5. EXECUTE SIMULATION ---
    try:
        log_returns = (demo_log_returns(a.tickers) if a.demo
                       else load_log_returns(a.tickers, a.lookback))
        res = simulate_portfolio(log_returns, a.weights, a.capital, a.annual_return,
                                 a.years, a.sims, a.seed, a.currency)
    except Exception as e:
        print(f"\n[ERROR] Simulation failed: {e}")
        sys.exit(1)

    # --- 6. PRINT RESULTS & MODELING CAVEATS ---
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    pd.set_option("display.width", 200)

    if res.psd_repaired:
        print(
            "\n[NOTICE] Historical covariance matrix was non-positive definite. "
            "Repaired using scale-relative eigenvalue clipping (note: this slightly adjusts "
            "the covariance matrix and simulated asset volatilities relative to raw historical values)."
        )

    print("\n--- 1. COMPANY PARAMETERS ---")
    print(res.asset_table().to_string(formatters={
        "Weight": "{:.1%}".format,
        "Assumed Ann. Return (Simple)": "{:.2%}".format,
        "Ann. Volatility": "{:.2%}".format}))
    print(f"\n--- 2. {len(a.tickers)}x{len(a.tickers)} CORRELATION MATRIX ---")
    print(res.corr.round(2))
    print(f"\n--- 3. MONTE CARLO RISK REPORT ({a.years:g}y Horizon, {a.sims:,} Paths) ---")
    print(res.summary().to_string())

    print("\n--- 4. MODELING STRATEGY & HORIZON CAVEATS ---")
    print(
        "• Strategy (Buy-and-Hold Total Return): Historical parameters use dividend- and "
        "split-adjusted total returns (yfinance auto_adjust=True). Simulated asset paths model "
        "buy-and-hold growth with reinvested dividend returns continuously embedded in asset growth "
        "(no separate cash dividend payouts or portfolio rebalancing)."
    )
    print(
        f"• Horizon-Specific Risk Metrics: VaR, CVaR, and P(Loss) represent dollar shortfalls "
        f"relative to starting capital at the end of the {a.years:g}-year horizon (not 1-year annualized VaR)."
    )
    print(
        f"• Expected Return Drift: Treated --return {a.annual_return:.2%} as a simple expected annual return. "
        f"Drift is set via mu_log = ln(1 + r) = {np.log1p(a.annual_return):.4f}, so expected portfolio "
        f"value compounds at exactly (1 + r)^T."
    )
    print(
        "• Long-Horizon Extrapolation Warning: Assuming constant historical volatility and correlations "
        "over multi-year horizons is a strong simplification. Structural market shifts, regime changes, "
        "and crisis correlations are understated. Treat outcomes as scenario estimates, not exact forecasts."
    )
    print(
        "• FX & Currency: Asset log returns are combined as if denominated directly in the portfolio "
        "reporting currency. Foreign exchange rate fluctuations and multi-currency conversions are not modeled."
    )

    if a.plot or a.save_plots:
        make_plots(res, save_dir=a.save_plots)


if __name__ == "__main__":
    main()