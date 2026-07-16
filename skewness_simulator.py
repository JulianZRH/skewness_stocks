"""Stock return skewness simulator.

GUI tool that simulates buy-and-hold stock returns via Monte Carlo and
shows how the compounding of random returns induces positive skewness
into the distribution of long-horizon outcomes -- the effect described in
Bessembinder, "One Hundred Years in the U.S. Stock Markets" (2026):
the mean buy-and-hold return across stocks is huge while the median is
negative, and the main driver of the skewness is short-horizon volatility
(Farago & Hjalmarsson, 2023).

Model: i.i.d. normal log-returns per step (geometric Brownian motion).
The per-step drift is calibrated so that the *expected arithmetic* annual
return matches the user input, i.e.

    E[terminal wealth] = (1 + R)^T        (exactly)
    median[terminal]   = (1 + R)^T * exp(-sigma^2 * T / 2)

so the gap between mean and median ("volatility drag") emerges naturally.
"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

# ----------------------------------------------------------------------
# Simulation core (no GUI dependencies -- importable for testing)
# ----------------------------------------------------------------------


def step_params(exp_return, volatility, steps_per_year):
    """Per-step mean and stdev of log returns.

    Calibrated so the expected gross return per year is exactly
    (1 + exp_return).
    """
    s = volatility / np.sqrt(steps_per_year)
    m = np.log1p(exp_return) / steps_per_year - 0.5 * s ** 2
    return m, s


def simulate_terminal(exp_return, volatility, years, steps_per_year,
                      n_paths, rng):
    """Terminal wealth multiples (1.0 = break-even) for n_paths paths."""
    m, s = step_params(exp_return, volatility, steps_per_year)
    n_steps = int(round(years * steps_per_year))
    total_log = rng.normal(m * n_steps, s * np.sqrt(n_steps), size=n_paths)
    return np.exp(total_log)


def simulate_paths(exp_return, volatility, years, steps_per_year,
                   n_paths, rng):
    """Full wealth paths, shape (n_paths, n_steps + 1), starting at 1.0."""
    m, s = step_params(exp_return, volatility, steps_per_year)
    n_steps = int(round(years * steps_per_year))
    increments = rng.normal(m, s, size=(n_paths, n_steps))
    log_paths = np.cumsum(increments, axis=1)
    paths = np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return paths


def skewness(x):
    """Sample skewness (Fisher-Pearson coefficient g1)."""
    x = np.asarray(x, dtype=float)
    d = x - x.mean()
    s2 = np.mean(d ** 2)
    if s2 == 0:
        return 0.0
    return np.mean(d ** 3) / s2 ** 1.5


def wealth_concentration(terminal, rf_multiple):
    """Smallest share of paths accounting for 100% of net wealth
    creation in excess of the risk-free outcome (Bessembinder's SWC
    concentration: the paths outside the top group collectively net to
    zero excess wealth).  Returns None if no net wealth is created."""
    excess = np.sort(terminal - rf_multiple)[::-1]
    total = excess.sum()
    if total <= 0:
        return None
    cum = np.cumsum(excess)
    return (np.argmax(cum >= total) + 1) / excess.size


def summary_stats(terminal, years, risk_free):
    """Dictionary of headline statistics for terminal wealth multiples."""
    rf_multiple = (1.0 + risk_free) ** years
    mean = terminal.mean()
    median = np.median(terminal)
    conc_all = wealth_concentration(terminal, rf_multiple)
    return {
        "conc_all": conc_all,
        "mean": mean,
        "median": median,
        "mean_ann": mean ** (1.0 / years) - 1.0,
        "median_ann": median ** (1.0 / years) - 1.0,
        "skew": skewness(terminal),
        "log_skew": skewness(np.log(terminal)),
        "pct_positive": np.mean(terminal > 1.0),
        "pct_beat_rf": np.mean(terminal > rf_multiple),
        "pct_beat_mean": np.mean(terminal > mean),
        "p05": np.quantile(terminal, 0.05),
        "p95": np.quantile(terminal, 0.95),
        "rf_multiple": rf_multiple,
    }


def volatility_sweep(exp_return, years, steps_per_year, n_paths,
                     risk_free, vols, seed):
    """Run the simulation for each volatility and collect key stats."""
    out = {"vol": [], "median_ann": [], "pct_positive": [],
           "pct_beat_mean": [], "skew": []}
    for i, vol in enumerate(vols):
        rng = np.random.default_rng(None if seed is None else seed + i)
        terminal = simulate_terminal(exp_return, vol, years,
                                     steps_per_year, n_paths, rng)
        st = summary_stats(terminal, years, risk_free)
        out["vol"].append(vol)
        out["median_ann"].append(st["median_ann"])
        out["pct_positive"].append(st["pct_positive"])
        out["pct_beat_mean"].append(st["pct_beat_mean"])
        out["skew"].append(st["skew"])
    return {k: np.asarray(v) for k, v in out.items()}


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------

DEFAULTS = {
    "exp_return": 8.0,      # % per year, arithmetic expectation
    "volatility": 30.0,     # % per year, typical single stock
    "years": 30.0,
    "n_paths": 10000,
    "steps_per_year": 12,   # monthly, like the CRSP data in the paper
    "risk_free": 3.0,       # % per year (T-bills did 3.3% over 1926-2025)
    "seed": "42",           # empty string = random seed
}

SWEEP_VOLS = np.arange(0.05, 0.65, 0.05)


class SimulatorApp:
    def __init__(self, root):
        self.root = root
        root.title("Stock Return Skewness Simulator")
        root.geometry("1150x720")

        self._build_inputs()
        self._build_plots()
        self.run_simulation()

    # ------------------------------------------------------------------
    def _build_inputs(self):
        panel = ttk.Frame(self.root, padding=10)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(panel, text="Inputs", font=("", 11, "bold")).pack(
            anchor="w", pady=(0, 6))

        self.vars = {}
        fields = [
            ("exp_return", "Expected return (% p.a.)"),
            ("volatility", "Volatility (% p.a.)"),
            ("years", "Horizon (years)"),
            ("n_paths", "Number of paths"),
            ("steps_per_year", "Steps per year"),
            ("risk_free", "Risk-free rate (% p.a.)"),
            ("seed", "Random seed (blank = random)"),
        ]
        for key, label in fields:
            row = ttk.Frame(panel)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=26).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(DEFAULTS[key]))
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT)
            self.vars[key] = var

        ttk.Button(panel, text="Run simulation",
                   command=self.run_simulation).pack(fill=tk.X, pady=(12, 2))
        ttk.Button(panel, text="Run volatility sweep",
                   command=self.run_sweep).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Reset defaults",
                   command=self.reset_defaults).pack(fill=tk.X, pady=2)

        ttk.Label(panel, text="Results", font=("", 11, "bold")).pack(
            anchor="w", pady=(14, 4))
        self.stats_text = tk.Text(panel, width=38, height=32,
                                  font=("Consolas", 9), state=tk.DISABLED,
                                  relief=tk.FLAT, background="#f0f0f0")
        self.stats_text.pack(fill=tk.BOTH, expand=True)

    def _build_plots(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figures = {}
        self.canvases = {}
        self.hist_scale = tk.StringVar(value="log")
        for name, title in [("hist", "Distribution"),
                            ("paths", "Sample paths"),
                            ("sweep", "Volatility sweep")]:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=title)
            if name == "hist":
                bar = ttk.Frame(frame)
                bar.pack(side=tk.TOP, anchor="w", padx=8, pady=4)
                ttk.Label(bar, text="x-axis:").pack(side=tk.LEFT)
                for text, value in [("Log scale", "log"),
                                    ("Linear scale", "linear")]:
                    ttk.Radiobutton(bar, text=text, value=value,
                                    variable=self.hist_scale,
                                    command=self.replot_hist).pack(
                        side=tk.LEFT, padx=6)
            fig = Figure(figsize=(7, 5.5), dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            NavigationToolbar2Tk(canvas, frame)
            self.figures[name] = fig
            self.canvases[name] = canvas
        self.last_run = None

    # ------------------------------------------------------------------
    def read_inputs(self):
        try:
            exp_return = float(self.vars["exp_return"].get()) / 100.0
            volatility = float(self.vars["volatility"].get()) / 100.0
            years = float(self.vars["years"].get())
            n_paths = int(float(self.vars["n_paths"].get()))
            steps = int(float(self.vars["steps_per_year"].get()))
            risk_free = float(self.vars["risk_free"].get()) / 100.0
            seed_str = self.vars["seed"].get().strip()
            seed = int(seed_str) if seed_str else None
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Please enter numeric values.")
            return None
        if exp_return <= -1.0 or volatility < 0 or years <= 0 \
                or n_paths < 100 or steps < 1:
            messagebox.showerror(
                "Invalid input",
                "Return must be > -100%, volatility >= 0, horizon > 0,\n"
                "at least 100 paths and 1 step per year.")
            return None
        return dict(exp_return=exp_return, volatility=volatility,
                    years=years, n_paths=n_paths, steps_per_year=steps,
                    risk_free=risk_free, seed=seed)

    def reset_defaults(self):
        for key, var in self.vars.items():
            var.set(str(DEFAULTS[key]))

    # ------------------------------------------------------------------
    def run_simulation(self):
        p = self.read_inputs()
        if p is None:
            return
        rng = np.random.default_rng(p["seed"])
        terminal = simulate_terminal(p["exp_return"], p["volatility"],
                                     p["years"], p["steps_per_year"],
                                     p["n_paths"], rng)
        rng_paths = np.random.default_rng(p["seed"])
        n_show = min(100, p["n_paths"])
        paths = simulate_paths(p["exp_return"], p["volatility"], p["years"],
                               p["steps_per_year"], n_show, rng_paths)

        stats = summary_stats(terminal, p["years"], p["risk_free"])
        self.last_run = (terminal, stats, p)
        self.show_stats(stats, p)
        plot_histogram(self.figures["hist"], terminal, stats, p,
                       scale=self.hist_scale.get())
        plot_paths(self.figures["paths"], paths, p, stats)
        self.canvases["hist"].draw()
        self.canvases["paths"].draw()

    def replot_hist(self):
        if self.last_run is None:
            return
        terminal, stats, p = self.last_run
        plot_histogram(self.figures["hist"], terminal, stats, p,
                       scale=self.hist_scale.get())
        self.canvases["hist"].draw()

    def run_sweep(self):
        p = self.read_inputs()
        if p is None:
            return
        sweep = volatility_sweep(p["exp_return"], p["years"],
                                 p["steps_per_year"], p["n_paths"],
                                 p["risk_free"], SWEEP_VOLS, p["seed"])
        plot_sweep(self.figures["sweep"], sweep, p)
        self.canvases["sweep"].draw()
        self.notebook.select(2)

    def show_stats(self, st, p):
        lines = [
            f"Horizon: {p['years']:.0f}y   Paths: {p['n_paths']:,}",
            f"E[return]: {p['exp_return']:.1%}   "
            f"Vol: {p['volatility']:.1%}",
            "",
            "Terminal wealth (1 = break-even)",
            f"  Mean            {st['mean']:>10.2f}x",
            f"  Median          {st['median']:>10.2f}x",
            f"  5% quantile     {st['p05']:>10.2f}x",
            f"  95% quantile    {st['p95']:>10.2f}x",
            "",
            "Lifetime (buy & hold) return",
            f"  mean          {st['mean'] - 1:>+12,.1%}",
            f"  median        {st['median'] - 1:>+12,.1%}",
            "",
            "Annualized (geometric)",
            f"  of mean         {st['mean_ann']:>10.2%}",
            f"  of median       {st['median_ann']:>10.2%}",
            "",
            "Skewness",
            f"  terminal wealth {st['skew']:>10.2f}",
            f"  log wealth      {st['log_skew']:>10.2f}",
            "",
            "Share of paths that ...",
            f"  end positive    {st['pct_positive']:>10.1%}",
            f"  beat risk-free  {st['pct_beat_rf']:>10.1%}",
            f"  beat the mean   {st['pct_beat_mean']:>10.1%}",
            "",
            "Wealth creation (vs risk-free)",
        ]
        if st["conc_all"] is None:
            lines.append("  no net wealth created")
        else:
            lines += [
                f"  all from top    {st['conc_all']:>10.1%}",
                f"  (bottom {1 - st['conc_all']:.1%} of paths",
                "   collectively net to zero)",
            ]
        lines += [
            "",
            f"Risk-free ends at {st['rf_multiple']:.2f}x",
        ]
        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", "\n".join(lines))
        self.stats_text.configure(state=tk.DISABLED)


# ----------------------------------------------------------------------
# Plotting (figure-level, reusable outside the GUI)
# ----------------------------------------------------------------------


def plot_histogram(fig, terminal, st, p, scale="log"):
    fig.clear()
    ax = fig.add_subplot(111)
    markers = [(st["mean"], "#c0392b", f"mean {st['mean']:.1f}x"),
               (st["median"], "#27ae60", f"median {st['median']:.1f}x"),
               (1.0, "#7f8c8d", "break-even 1x")]
    if scale == "log":
        ax.hist(np.log10(terminal), bins=80, color="#4878a8",
                edgecolor="white", linewidth=0.3)
        for value, color, label in markers:
            ax.axvline(np.log10(value), color=color, linestyle="--",
                       linewidth=1.4, label=label)
        ticks = ax.get_xticks()
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{10 ** v:g}x"))
        ax.set_xticks(ticks)
        ax.set_xlabel("Terminal wealth per $1 invested (log scale)")
    else:
        hi = np.quantile(terminal, 0.99)
        clipped = terminal[terminal <= hi]
        n_over = terminal.size - clipped.size
        ax.hist(clipped, bins=80, color="#4878a8",
                edgecolor="white", linewidth=0.3)
        for value, color, label in markers:
            if value > hi:
                ax.plot([], [], color=color, linestyle="--",
                        linewidth=1.4, label=label + " (off scale)")
            else:
                ax.axvline(value, color=color, linestyle="--",
                           linewidth=1.4, label=label)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.set_xlabel(
            f"Terminal wealth per $1 invested (linear scale, "
            f"top 1% = {n_over} paths beyond axis)")
    ax.set_ylabel("Number of paths")
    ax.set_title(
        f"Terminal wealth after {p['years']:.0f} years  "
        f"(E[r]={p['exp_return']:.0%}, vol={p['volatility']:.0%})\n"
        f"Only {st['pct_beat_mean']:.0%} of paths beat the mean outcome")
    ax.legend()
    fig.tight_layout()


def plot_paths(fig, paths, p, st):
    fig.clear()
    ax = fig.add_subplot(111)
    t = np.linspace(0, p["years"], paths.shape[1])
    ax.plot(t, paths.T, color="#4878a8", alpha=0.25, linewidth=0.7)
    ax.plot(t, np.median(paths, axis=0), color="#27ae60", linewidth=2,
            label="median path")
    ax.plot(t, (1 + p["exp_return"]) ** t, color="#c0392b", linewidth=2,
            linestyle="--", label="expected value")
    ax.set_yscale("log")
    ax.set_xlabel("Years")
    ax.set_ylabel("Wealth per $1 invested (log scale)")
    ax.set_title(f"{paths.shape[0]} sample paths")
    ax.legend()
    fig.tight_layout()


def plot_sweep(fig, sweep, p):
    fig.clear()
    ax1 = fig.add_subplot(211)
    ax1.plot(sweep["vol"], sweep["median_ann"], "o-", color="#27ae60",
             label="median annualized return")
    ax1.axhline(p["exp_return"], color="#c0392b", linestyle="--",
                label=f"expected return {p['exp_return']:.0%}")
    ax1.axhline(0, color="#7f8c8d", linewidth=0.8)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylabel("Annualized return")
    ax1.set_title(
        f"Effect of volatility over {p['years']:.0f} years "
        f"(E[r]={p['exp_return']:.0%} in all runs)")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(sweep["vol"], sweep["pct_positive"], "o-", color="#4878a8",
             label="paths ending positive")
    ax2.plot(sweep["vol"], sweep["pct_beat_mean"], "s-", color="#e67e22",
             label="paths beating the mean")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_xlabel("Annual volatility")
    ax2.set_ylabel("Share of paths")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)
    fig.tight_layout()


def main():
    root = tk.Tk()
    SimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
