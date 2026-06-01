"""
Fourier series approximation of a hat (triangle) function on [-π, π].
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    L: float = np.pi          # half-period: function defined on [-L, L]
    ub: float = np.pi / 2     # hat half-width  (must be ≤ L)
    dx: float = 0.001         # spatial resolution
    n_modes: int = 10         # number of Fourier modes to show


CFG = Config()


# ---------------------------------------------------------------------------
# Domain & target function
# ---------------------------------------------------------------------------

def make_domain(cfg: Config) -> np.ndarray:
    """Uniform grid on (-L, L] with step dx."""
    return np.arange(-cfg.L + cfg.dx, cfg.L + cfg.dx, cfg.dx)


def hat_function(x: np.ndarray, ub: float) -> np.ndarray:
    """Vectorised hat (triangle) function centred at 0."""
    return np.where(np.abs(x) < ub, np.pi / 2 - np.abs(x), 0.0)


# ---------------------------------------------------------------------------
# Fourier coefficients  (full Fourier series on [-L, L])
# ---------------------------------------------------------------------------

def fourier_coefficients(
    f: np.ndarray, x: np.ndarray, n_modes: int, L: float, dx: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Fourier coefficients via numerical quadrature.

    Returns
    -------
    a0 : float
        Zeroth cosine coefficient (A0 / L).
    A  : ndarray, shape (n_modes,)
        Cosine coefficients A_k / L.
    B  : ndarray, shape (n_modes,)
        Sine coefficients B_k / L.
    """
    k = np.arange(1, n_modes + 1)          # shape (n_modes,)
    # x: (n,)  →  broadcast with k: (n_modes,) via (n,1) x (n_modes,)
    cos_mat = np.cos(np.pi * k[None, :] * x[:, None] / L)   # (n, n_modes)
    sin_mat = np.sin(np.pi * k[None, :] * x[:, None] / L)

    a0 = (np.sum(f) * dx) / L
    A  = (f[:, None] * cos_mat).sum(axis=0) * dx / L        # (n_modes,)
    B  = (f[:, None] * sin_mat).sum(axis=0) * dx / L

    return a0, A, B


def partial_sums(
    x: np.ndarray,
    a0: float,
    A: np.ndarray,
    B: np.ndarray,
    L: float,
) -> list[np.ndarray]:
    """
    Return a list of partial sums S_1, S_2, …, S_K where S_k includes
    modes 1 … k on top of the DC term a0/2.
    """
    k_all = np.arange(1, len(A) + 1)
    accumulator = np.full_like(x, a0 / 2)
    sums = []
    for k in k_all:
        accumulator = (
            accumulator
            + A[k - 1] * np.cos(k * np.pi * x / L)
            + B[k - 1] * np.sin(k * np.pi * x / L)
        )
        sums.append(accumulator.copy())
    return sums


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def errors(f: np.ndarray, fa: np.ndarray) -> tuple[float, float]:
    """Return (L∞, L²) errors between f and its approximation fa."""
    diff = np.abs(f - fa)
    linf = diff.max()
    l2   = np.sqrt(np.mean(diff ** 2))
    return linf, l2


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_original(x: np.ndarray, f: np.ndarray, filename: str = "hat_function.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, f, lw=2.5, color="steelblue", label="Hat function")
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$f(x)$", fontsize=13)
    ax.set_title("Hat Function on $[-\\pi,\\, \\pi]$", fontsize=14)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.1, np.pi / 2 + 0.2)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v/np.pi:.1g}π" if v != 0 else "0")
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved → {filename}")
    plt.show()


def plot_modes(
    x: np.ndarray,
    f: np.ndarray,
    sums: list[np.ndarray],
    filename: str = "modes_function.png",
) -> None:
    cmap   = plt.get_cmap("plasma")
    n      = len(sums)
    colors = [cmap(i / (n - 1)) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: partial sums overlaid ---
    ax = axes[0]
    ax.plot(x, f, "--", lw=2.5, color="black", label="Exact", zorder=10)
    for k, (fa, col) in enumerate(zip(sums, colors), start=1):
        ax.plot(x, fa, lw=1.5, color=col, alpha=0.85, label=f"$S_{{{k}}}$")

    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$f(x)$", fontsize=13)
    ax.set_title("Fourier Partial Sums", fontsize=14)
    ax.set_xlim(x[0], x[-1])
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.35)

    # --- Right: L∞ and L² error vs number of modes ---
    ax2 = axes[1]
    mode_idx = np.arange(1, n + 1)
    linf_vals, l2_vals = zip(*(errors(f, fa) for fa in sums))

    ax2.semilogy(mode_idx, linf_vals, "o-", lw=2, label=r"$L^\infty$ error")
    ax2.semilogy(mode_idx, l2_vals,   "s-", lw=2, label=r"$L^2$ error")
    ax2.set_xlabel("Number of modes", fontsize=13)
    ax2.set_ylabel("Error", fontsize=13)
    ax2.set_title("Convergence of Fourier Series", fontsize=14)
    ax2.set_xticks(mode_idx)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved → {filename}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = CFG

    x = make_domain(cfg)
    f = hat_function(x, cfg.ub)

    # Coefficients & partial sums
    a0, A, B = fourier_coefficients(f, x, cfg.n_modes, cfg.L, cfg.dx)
    sums = partial_sums(x, a0, A, B, cfg.L)

    # Print error table
    print(f"\n{'Mode':>5}  {'L∞ error':>12}  {'L² error':>12}")
    print("-" * 35)
    for k, fa in enumerate(sums, start=1):
        li, l2 = errors(f, fa)
        print(f"{k:>5}  {li:>12.4e}  {l2:>12.4e}")

    # Plots
    plot_original(x, f)
    plot_modes(x, f, sums)


if __name__ == "__main__":
    main()
