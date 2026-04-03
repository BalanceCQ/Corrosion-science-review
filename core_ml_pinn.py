import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import shap
from pathlib import Path
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from input_define import time_series_prediction_input_without_time, pinn_partial_dependence_input


# -----------------------
# 1. Network Architecture
# -----------------------
class CorrosionNet(nn.Module):
    def __init__(self, env_dim=4, comp_dim=7, hidden_dim=36):
        super().__init__()
        # Separate branches for environmental and compositional features
        self.env_fc = nn.Sequential(nn.Linear(env_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.comp_fc = nn.Sequential(nn.Linear(comp_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        # Fusion layer to combine both branches
        self.fusion_fc = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.out_fc = nn.Linear(hidden_dim, 2)

        # Learnable uncertainty weights for multi-task balancing
        self.log_vars = nn.ParameterDict(
            {'A': nn.Parameter(torch.zeros(1)), 'n': nn.Parameter(torch.zeros(1)), 'y': nn.Parameter(torch.zeros(1))})

    def forward(self, env_in, comp_in):
        return self.out_fc(self.fusion_fc(torch.cat([self.env_fc(env_in), self.comp_fc(comp_in)], dim=1)))


# -----------------------
# 2. Training Pipeline
# -----------------------
def train(model, env_dim, train_loader, val_loader, epochs=10000, lr=0.001, device="cuda", patience=200, env_w=0.6,
          comp_w=0.5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    prev_loss, no_improve = float('inf'), 0
    history = []

    def calc_phys_loss(env_in, comp_in, pred_A, pred_n):
        # Environmental physical loss
        gA_env = torch.autograd.grad(pred_A.sum(), env_in, create_graph=True)[0]
        gn_env = torch.autograd.grad(pred_n.sum(), env_in, create_graph=True)[0]
        env_loss = torch.relu(-gA_env).mean() + torch.relu(-gn_env).mean()

        # Compositional physical loss (Constraints: Cu[3], Cr[4], Ni[5], P[6] -> Negative; C[0], S[1], Mn[2] -> Positive)
        gA_comp = torch.autograd.grad(pred_A.sum(), comp_in, create_graph=True)[0]
        gn_comp = torch.autograd.grad(pred_n.sum(), comp_in, create_graph=True)[0]
        neg_idx, pos_idx = [3, 4, 5, 6], [0, 1, 2]

        comp_loss = (torch.relu(gA_comp[:, neg_idx]).mean() + torch.relu(gn_comp[:, neg_idx]).mean() +
                     torch.relu(-gA_comp[:, pos_idx]).mean() + torch.relu(-gn_comp[:, pos_idx]).mean())
        return env_loss, comp_loss

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            env_in, comp_in = x[:, :env_dim].requires_grad_(True), x[:, env_dim:].requires_grad_(True)

            preds = model(env_in, comp_in)
            pA, pn = preds[:, 0:1], preds[:, 1:2]
            tA, tn, tT, ty = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

            # Homoscedastic uncertainty optimization
            lA, ln, ly = mse(pA, tA), mse(pn, tn), mse(pA * (tT ** pn), ty)
            lvA, lvn, lvy = model.log_vars['A'], model.log_vars['n'], model.log_vars['y']
            data_loss = (torch.exp(-lvA) * lA + lvA) + (torch.exp(-lvn) * ln + lvn) + (torch.exp(-lvy) * ly + lvy)

            env_phys, comp_phys = calc_phys_loss(env_in, comp_in, pA, pn)
            loss = data_loss + env_w * env_phys + comp_w * comp_phys

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation logic
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p = model(x[:, :env_dim], x[:, env_dim:])
                lA, ln, ly = mse(p[:, 0:1], y[:, 0:1]), mse(p[:, 1:2], y[:, 1:2]), mse(
                    p[:, 0:1] * (y[:, 2:3] ** p[:, 1:2]), y[:, 3:4])
                lvA, lvn, lvy = model.log_vars['A'], model.log_vars['n'], model.log_vars['y']
                val_loss += ((torch.exp(-lvA) * lA + lvA) + (torch.exp(-lvn) * ln + lvn) + (
                            torch.exp(-lvy) * ly + lvy)).item()

        val_loss /= len(val_loader)
        history.append({'epoch': epoch, 'train_loss': train_loss / len(train_loader), 'val_loss': val_loss})

        # Early Stopping
        if val_loss < prev_loss:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Breaking loop.")
                break
        prev_loss = val_loss

    # Save final optimized model exactly at the stopping point
    save_dir = Path('./Result/Model')
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / 'MF-PINN_model_final.pth')
    return model, pd.DataFrame(history)


# -----------------------
# 3. Evaluation & Utils
# -----------------------
def evaluate(model, env_dim, loader, device="cuda"):
    model.eval()
    pA, pn, py, tA, tn, ty = [], [], [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            preds, targets = model(x[:, :env_dim].to(device), x[:, env_dim:].to(device)).cpu().numpy(), y.numpy()
            pA.extend(preds[:, 0]);
            pn.extend(preds[:, 1]);
            py.extend(preds[:, 0] * (targets[:, 2] ** preds[:, 1]))
            tA.extend(targets[:, 0]);
            tn.extend(targets[:, 1]);
            ty.extend(targets[:, 3])

    def calc_metrics(t, p):
        return {'R2': r2_score(t, p), 'RMSE': root_mean_squared_error(t, p), 'MAE': mean_absolute_error(t, p),
                'MdAPE': np.median(np.abs((t - p) / t))}

    return {'A': calc_metrics(np.array(tA), np.array(pA)), 'n': calc_metrics(np.array(tn), np.array(pn)),
            'y': calc_metrics(np.array(ty), np.array(py))}


def save_predictions(model, env_dim, loader, meta_df, save_path, device="cuda"):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, y in loader:
            out = model(x[:, :env_dim].to(device), x[:, env_dim:].to(device)).cpu().numpy()
            preds.append(np.hstack([out, y.numpy()]))

    preds = np.vstack(preds)
    df = meta_df.copy().reset_index(drop=True)
    df.assign(A_Pred=preds[:, 0], n_Pred=preds[:, 1], y_Pred=preds[:, 0] * (preds[:, 4] ** preds[:, 1]),
              A_True=preds[:, 2], n_True=preds[:, 3], t_True=preds[:, 4], y_True=preds[:, 5]).to_excel(save_path,
                                                                                                       index=False)


# -----------------------
# 4. Analytics (PDP & SHAP Data Exporter Only)
# -----------------------
def run_pdp_and_shap(model, data, scaler, env_feats, comp_feats, device="cuda"):
    env_dim = len(env_feats)

    # 1. Partial Dependence Analysis (Export to Excel)
    print("Initiating PDP Analysis...")
    for feat in env_feats + comp_feats:
        _, _, x_mean, rng = pinn_partial_dependence_input(feat)
        x_tensor = torch.tensor(x_mean, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(x_tensor[:, :env_dim], x_tensor[:, env_dim:]).cpu().numpy()

        save_dir = Path(f'./Result/PDP_Analysis/{"Env_Factors" if feat in env_feats else "Comp_Factors"}')
        save_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'feature_range': rng, 'A': preds[:, 0], 'n': preds[:, 1]}).to_excel(
            save_dir / f'PDP_{feat}.xlsx', index=False)

    # 2. SHAP Analysis (Values Export Only)
    print("Initiating SHAP Analysis...")
    X_scaled = scaler.transform(data[env_feats + comp_feats].values) if scaler else data[env_feats + comp_feats].values

    def predict_fn(x):
        x_t = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad(): return model(x_t[:, :env_dim], x_t[:, env_dim:]).cpu().numpy()

    explainer = shap.KernelExplainer(predict_fn, X_scaled)
    shap_vals = explainer.shap_values(X_scaled, nsamples="auto")
    shap_A, shap_n = (shap_vals[0], shap_vals[1]) if isinstance(shap_vals, list) else (
    shap_vals[:, :, 0], shap_vals[:, :, 1])

    shap_dir = Path('./Result/SHAP_Analysis')
    shap_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Feature': env_feats + comp_feats, 'Mean_Abs_SHAP_A': np.abs(shap_A).mean(axis=0),
                  'Mean_Abs_SHAP_n': np.abs(shap_n).mean(axis=0)}).to_excel(
        shap_dir / "SHAP_Feature_Importance.xlsx", index=False)


# -----------------------
# 5. Main Execution
# -----------------------
if __name__ == '__main__':
    torch.manual_seed(0);
    np.random.seed(0);
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Loading
    df = pd.read_excel(r'.\Dataset\PINN-Data.xlsx', sheet_name='Sheet1')
    df = df[df["y"] != 0].reset_index(drop=True).fillna(0)
    
    # Feature Definitions
    env_feats = ['T', 'RH', 'SO2', 'Cl']
    comp_feats = ['C', 'S', 'Mn', 'Cu', 'Cr', 'Ni', 'P']
    target_feats = ['A1', 'n1', 't', 'y1']
    # Split & Scale
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=2)
    train_idx, test_idx = next(gss.split(df, groups=df['Station'] + "_" + df['Steel']))

    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(df[env_feats + comp_feats].iloc[train_idx]), dtype=torch.float32)
    X_test = torch.tensor(scaler.transform(df[env_feats + comp_feats].iloc[test_idx]), dtype=torch.float32)
    y_train, y_test = torch.tensor(df[target_feats].iloc[train_idx].values, dtype=torch.float32), torch.tensor(
        df[target_feats].iloc[test_idx].values, dtype=torch.float32)

    loader_args = dict(batch_size=64, pin_memory=True)
    train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, **loader_args)
    train_eval_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=False, **loader_args)
    test_loader = DataLoader(TensorDataset(X_test, y_test), shuffle=False, **loader_args)

    # Train
    model = CorrosionNet(env_dim=len(env_feats), comp_dim=len(comp_feats)).to(device)
    model, history_df = train(model, len(env_feats), train_loader, test_loader, device=device)

    Path('./Result/Train_Test').mkdir(parents=True, exist_ok=True)
    history_df.to_excel(Path('./Result/Train_Test/Training_History.xlsx'), index=False)

    # Evaluate Metrics & Flatten Dict
    train_res, test_res = evaluate(model, len(env_feats), train_eval_loader, device), evaluate(model, len(env_feats),
                                                                                               test_loader, device)
    flat_metrics = {f"{phase}_{tgt}_{met}": val for phase, res in zip(['Train', 'Test'], [train_res, test_res]) for
                    tgt, mets in res.items() for met, val in mets.items()}
    pd.DataFrame([flat_metrics]).to_excel(Path('./Result/Train_Test/Performance_Metrics.xlsx'), index=False)

    # Save Predictions
    save_predictions(model, len(env_feats), train_eval_loader, df[['Station', 'Steel']].iloc[train_idx],
                     Path('./Result/Train_Test/Train_Predictions.xlsx'), device)
    save_predictions(model, len(env_feats), test_loader, df[['Station', 'Steel']].iloc[test_idx],
                     Path('./Result/Train_Test/Test_Predictions.xlsx'), device)

    # Analytics
    run_pdp_and_shap(model, df, scaler, env_feats, comp_feats, device)
    print("Pipeline execution completed successfully.")
