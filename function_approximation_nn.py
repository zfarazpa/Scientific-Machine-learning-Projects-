import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ConstantLR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(1234)
np.random.seed(1234)

# ── Target functions ──────────────────────────────────────────────────────────
def fun_x(x):
    """Noisy piecewise function used for training labels."""
    f = (5.0 + np.sin(x) + np.sin(2*x) + np.sin(3*x) + np.sin(4*x)
         if x < 0.0 else np.cos(10.0 * x))
    return f + np.random.normal(0, 0.1)

def efun_x(x):
    """Exact (noise-free) version used for evaluation."""
    return (5.0 + np.sin(x) + np.sin(2*x) + np.sin(3*x) + np.sin(4*x)
            if x < 0.0 else np.cos(10.0 * x))

# ── Data ──────────────────────────────────────────────────────────────────────
a, b, N = -np.pi, np.pi, 700
x_all = np.linspace(a, b, N)

x_train, x_test = train_test_split(x_all, test_size=0.3, random_state=42)

# Noisy labels for training, exact labels for evaluation
y_train_noisy = np.array([fun_x(xi) for xi in x_train])
y_train_exact = np.array([efun_x(xi) for xi in x_train])   # ← fix: use efun_x
y_test_vals   = np.array([fun_x(xi) for xi in x_test])

def to_tensor(arr):
    return torch.from_numpy(arr.reshape(-1, 1)).float()

x_tr  = to_tensor(x_train)
y_tr  = to_tensor(y_train_noisy)
y_tr_e = to_tensor(y_train_exact)
x_te  = to_tensor(x_test)
y_te  = to_tensor(y_test_vals)

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(hidden=100):
    return nn.Sequential(
        nn.Linear(1, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )

net = build_model()

# ── Optimizer + LR Scheduler       Am I using the best optimizer? Why Adam???? It is wellknow :)))──────────────────────────────────────────────────
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
# ConstantLR keeps the LR fixed (factor=1.0) — easy to swap for CosineAnnealingLR etc.
scheduler = ConstantLR(optimizer, factor=1.0, total_iters=30_000)

loss_fn = nn.MSELoss()

# ── Training loop ─────────────────────────────────────────────────────────────
N_ITER   = 30_000
LOG_FREQ = 500

train_losses, test_losses = [], []
train_accs,   test_accs   = [], []

print("Adam Optimization")
for it in range(N_ITER):
    net.train()
    y_pred = net(x_tr)
    loss   = loss_fn(y_pred, y_tr)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()          # ← scheduler updated every iteration

    # ── Logging every LOG_FREQ steps ──────────────────────────────────────────
    if (it + 1) % LOG_FREQ == 0 or it == 0:
        net.eval()
        with torch.no_grad():
            # Training metrics (vs exact signal)
            y_tr_pred  = net(x_tr)
            loss_exact = loss_fn(y_tr_pred, y_tr_e).item()
            tr_acc     = (torch.linalg.norm(y_tr_e - y_tr_pred)
                          / torch.linalg.norm(y_tr_pred)).item()

            # Test metrics
            y_te_pred  = net(x_te)
            loss_test  = loss_fn(y_te_pred, y_te).item()
            te_acc     = (torch.linalg.norm(y_te - y_te_pred)
                          / torch.linalg.norm(y_te_pred)).item()

        train_losses.append(loss_exact)
        test_losses.append(loss_test)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        print(f"Epoch {it+1:>6}/{N_ITER} | "
              f"Train Loss: {loss_exact:.4e} | Test Loss: {loss_test:.4e} | "
              f"Train Err: {tr_acc:.4f} | Test Err: {te_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

y_const_lr = net(x_tr).detach().numpy()

# ── Plotting ──────────────────────────────────────────────────────────────────
epochs_logged = [i * LOG_FREQ for i in range(len(train_losses))]
epochs_logged[0] = 1  # first entry is epoch 1

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].semilogy(epochs_logged, train_losses, label="Train (exact)")
axes[0].semilogy(epochs_logged, test_losses,  label="Test")
axes[0].set(title="MSE Loss", xlabel="Epoch", ylabel="Loss")
axes[0].legend()

axes[1].plot(epochs_logged, train_accs, label="Train")
axes[1].plot(epochs_logged, test_accs,  label="Test")
axes[1].set(title="Relative Error", xlabel="Epoch", ylabel="‖y−ŷ‖/‖ŷ‖")
axes[1].legend()

plt.tight_layout()
plt.show()
