"""
Neural network approximation of the Runge function  1 / (1 + 25x²)
on [-1, 1].
"""


from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All hyper-parameters in one place."""
    seed: int = 1234
    a: float = -1.0
    b: float = 1.0

    # Network
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64, 64])
    activation: str = "tanh"          # "tanh" | "leaky_relu" | "gelu"

    # Optimisation
    lr: float = 1e-3
    n_iter: int = 8_000
    patience: int = 500               # early-stopping patience (0 = disabled)
    grad_clip: float = 1.0            # max gradient norm (0 = disabled)

    # Scheduler
    use_scheduler: bool = True
    eta_min: float = 1e-5             # CosineAnnealing floor

    # Evaluation / plotting
    n_plot: int = 500
    log_every: int = 500


CFG = Config()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def exact_sol(x: torch.Tensor) -> torch.Tensor:
    """Runge function  f(x) = 1 / (1 + 25x²)."""
    return 1.0 / (1.0 + 25.0 * x ** 2)


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> torch.Tensor:
    """Return n Chebyshev nodes of the first kind on [a, b], shape (n, 1)."""
    if n <= 0:
        raise ValueError("Number of points must be positive.")

    k = torch.arange(n, dtype=torch.float32)
    x = 0.5 * (a + b) + 0.5 * (b - a) * torch.cos((2 * k + 1) * np.pi / (2 * n))
    return x.reshape(-1, 1)


def get_training_points(choice: int, n: int, cfg: Config) -> torch.Tensor:
    """Return training points of shape (n, 1)."""
    if n <= 0:
        raise ValueError("Number of training points must be positive.")

    a, b = cfg.a, cfg.b

    if choice == 1:
        print("\nChoosing random training points\n")
        return a + (b - a) * torch.rand(n, 1)
    elif choice == 2:
        print("\nChoosing equidistant training points\n")
        return torch.linspace(a, b, n).reshape(-1, 1)
    elif choice == 3:
        print("\nChoosing Chebyshev nodes\n")
        return chebyshev_nodes(n, a, b)
    else:
        raise ValueError("Choice must be 1, 2, or 3.")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def build_activation(name: str) -> nn.Module:
    activations = {
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
    }

    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations)}.")

    return activations[name]


def initialise_layer(linear: nn.Linear, activation: str) -> None:
    if activation == "tanh":
        nn.init.xavier_normal_(linear.weight)
    else:
        nn.init.kaiming_normal_(linear.weight, nonlinearity="leaky_relu")

    nn.init.zeros_(linear.bias)


def build_network(hidden_sizes: list[int], activation: str) -> nn.Sequential:
    """
    Build a fully-connected network with configurable depth/width.
    Kaiming initialisation for hidden layers, Xavier for the output layer.
    """
    if not hidden_sizes:
        raise ValueError("hidden_sizes must contain at least one hidden layer.")

    layers: list[nn.Module] = []
    in_dim = 1

    for h in hidden_sizes:
        if h <= 0:
            raise ValueError("Hidden-layer sizes must be positive.")

        linear = nn.Linear(in_dim, h)
        initialise_layer(linear, activation)
        layers += [linear, build_activation(activation)]
        in_dim = h

    out = nn.Linear(in_dim, 1)
    nn.init.xavier_normal_(out.weight)
    nn.init.zeros_(out.bias)
    layers.append(out)

    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    net: nn.Sequential,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    cfg: Config,
) -> list[float]:
    """Train the network and return the loss history."""
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.n_iter, eta_min=cfg.eta_min
        )
        if cfg.use_scheduler
        else None
    )

    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    patience_counter = 0
    loss_history: list[float] = []

    for it in range(cfg.n_iter + 1):
        net.train()

        y_pred = net(x_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if it % cfg.log_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Iter {it:6d} | Loss = {loss_val:.4e} | LR = {lr_now:.2e}")

        # Early stopping
        if cfg.patience > 0:
            if loss_val < best_loss - 1e-10:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                print(f"\nEarly stopping at iteration {it} (patience={cfg.patience}).")
                break

    return loss_history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    net: nn.Sequential,
    x_plot: torch.Tensor,
    y_exact: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Return predictions and L∞ / L² errors on the dense evaluation grid."""
    net.eval()

    with torch.no_grad():
        y_pred = net(x_plot)

    err = (y_pred - y_exact).abs()
    linf = err.max().item()
    l2 = torch.sqrt(torch.mean((y_pred - y_exact) ** 2)).item()

    return y_pred, linf, l2


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_approximation(
    x_plot: torch.Tensor,
    y_exact: torch.Tensor,
    y_pred: torch.Tensor,
    x_train: torch.Tensor,
    linf: float,
    l2: float,
    cfg: Config,
    filename: str = "NN_approximation.png",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: function approximation ---
    ax = axes[0]
    xp = x_plot.detach().numpy().ravel()

    ax.plot(xp, y_exact.detach().numpy().ravel(), "--", lw=2, label="Exact")
    ax.plot(xp, y_pred.detach().numpy().ravel(), "-", lw=2, label="NN prediction")
    ax.scatter(
        x_train.detach().numpy().ravel(),
        np.full(len(x_train), -0.03),
        marker="|", s=80, color="C2", zorder=5, label="Training pts",
    )

    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$f(x)$", fontsize=14)
    ax.set_xlim(cfg.a - 0.1, cfg.b + 0.1)
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("Runge Function Approximation", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)

    # --- Right: pointwise error ---
    ax2 = axes[1]
    err = (y_pred - y_exact).abs().detach().numpy().ravel()

    ax2.semilogy(xp, err, color="C3", lw=2)
    ax2.set_xlabel(r"$x$", fontsize=14)
    ax2.set_ylabel("Absolute error", fontsize=14)
    ax2.set_title(
        f"Pointwise Error   L∞={linf:.2e},  L²={l2:.2e}", fontsize=13
    )
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved → {filename}")
    plt.show()


def plot_loss(
    loss_history: list[float],
    filename: str = "loss_history.png",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(loss_history, lw=1.5, color="C0")
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("MSE Loss", fontsize=13)
    ax.set_title("Training Loss History", fontsize=14)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved → {filename}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(CFG.seed)

    # --- User input ---
    print(
        "Select distribution for training points:\n"
        "  1. Random\n"
        "  2. Equidistant\n"
        "  3. Chebyshev nodes"
    )

    try:
        d = int(input("Choice: "))
        N = int(input("Number of training points: "))
    except (ValueError, EOFError):
        print("Non-interactive mode detected – using defaults: Chebyshev, N=30.")
        d, N = 3, 30

    # --- Data ---
    x_train = get_training_points(d, N, CFG)
    y_train = exact_sol(x_train)

    x_plot = torch.linspace(CFG.a, CFG.b, CFG.n_plot).reshape(-1, 1)
    y_exact = exact_sol(x_plot)

    # --- Model ---
    net = build_network(CFG.hidden_sizes, CFG.activation)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(
        f"\nNetwork: {CFG.hidden_sizes}  |  "
        f"activation: {CFG.activation}  |  "
        f"params: {n_params:,}\n"
    )

    # --- Train ---
    loss_history = train(net, x_train, y_train, CFG)

    # --- Evaluate ---
    y_pred, linf, l2 = evaluate(net, x_plot, y_exact)

    print(f"\nFinal errors on [{CFG.a}, {CFG.b}]:  L∞ = {linf:.4e}   L² = {l2:.4e}")

    # --- Plot ---
    plot_approximation(x_plot, y_exact, y_pred, x_train, linf, l2, CFG)
    plot_loss(loss_history)


if __name__ == "__main__":
    main()


   
   
   
