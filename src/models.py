import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

def train_unthinkable_blend(target_name, X_train, y_train_series, X_test_full, groups):
    print(f"\n--- Training 15-Fold Unthinkable Blend for {target_name} ---")
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test_full))
    
    sgkf = StratifiedGroupKFold(n_splits=15, shuffle=True, random_state=42)
    fold_aucs, fold_lls = [], []
    SEEDS = [42, 2024, 888] 
    
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_train, y_train_series, groups=groups)):
        X_trn, y_trn = X_train.iloc[trn_idx], y_train_series.iloc[trn_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train_series.iloc[val_idx]
        
        fold_val_preds = np.zeros(len(X_val))
        fold_test_preds = np.zeros(len(X_test_full))
        
        for seed in SEEDS:
            # 1. LIGHTGBM
            lgb_base = lgb.LGBMClassifier(
                n_estimators=1500, learning_rate=0.01, max_depth=4,
                min_child_samples=10, scale_pos_weight=15.0, 
                subsample=0.85, colsample_bytree=0.4, 
                random_state=seed + fold, n_jobs=-1, verbose=-1
            )
            lgb_base.fit(X_trn, y_trn)
            val_lgb = lgb_base.predict_proba(X_val)[:, 1]
            test_lgb = lgb_base.predict_proba(X_test_full)[:, 1]
            
            iso_lgb = IsotonicRegression(out_of_bounds='clip').fit(val_lgb, y_val)
            cal_val_lgb = np.clip(iso_lgb.predict(val_lgb), 1e-15, 1)
            cal_test_lgb = np.clip(iso_lgb.predict(test_lgb), 1e-15, 1)
            
            # 2. CATBOOST
            cb_base = CatBoostClassifier(
                iterations=1500, learning_rate=0.01, depth=5,
                scale_pos_weight=15.0, colsample_bylevel=0.4, 
                verbose=0, random_seed=seed + fold
            )
            cb_base.fit(X_trn, y_trn)
            val_cb = cb_base.predict_proba(X_val)[:, 1]
            test_cb = cb_base.predict_proba(X_test_full)[:, 1]
            
            iso_cb = IsotonicRegression(out_of_bounds='clip').fit(val_cb, y_val)
            cal_val_cb = np.clip(iso_cb.predict(val_cb), 1e-15, 1)
            cal_test_cb = np.clip(iso_cb.predict(test_cb), 1e-15, 1)

            # THE ARITHMETIC BLEND (40% LGB, 60% CB)
            arithmetic_val = (cal_val_lgb * 0.40) + (cal_val_cb * 0.60)
            arithmetic_test = (cal_test_lgb * 0.40) + (cal_test_cb * 0.60)
            
            # THE GEOMETRIC BLEND (40% LGB, 60% CB)
            geometric_val = (cal_val_lgb ** 0.40) * (cal_val_cb ** 0.60)
            geometric_test = (cal_test_lgb ** 0.40) * (cal_test_cb ** 0.60)
            
            # THE 70/30 HYBRID MASTER BLEND
            seed_val_blend = (geometric_val * 0.70) + (arithmetic_val * 0.30)
            seed_test_blend = (geometric_test * 0.70) + (arithmetic_test * 0.30)
            
            fold_val_preds += seed_val_blend / len(SEEDS)
            fold_test_preds += seed_test_blend / len(SEEDS)
            
        fold_val_preds = np.clip(fold_val_preds, 1e-15, 1 - 1e-15)
        oof_preds[val_idx] = fold_val_preds
        test_preds += fold_test_preds / 15 
        
        auc = roc_auc_score(y_val, fold_val_preds)
        ll = log_loss(y_val, fold_val_preds)
        fold_aucs.append(auc)
        fold_lls.append(ll)
        print(f"Fold {fold+1:02d} | AUC: {auc:.5f} | LogLoss: {ll:.5f}")
        
    print(f">> Final BLEND {target_name} | MEAN AUC: {np.mean(fold_aucs):.5f} | MEAN LogLoss: {np.mean(fold_lls):.5f}")
    return test_preds