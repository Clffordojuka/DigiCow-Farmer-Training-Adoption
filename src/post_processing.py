import numpy as np
import pandas as pd

def process_and_save_submission(test_ids, pred_07, pred_90, pred_120, output_path):
    print("\nEnforcing Monotonicity and Applying the 0.01% Dampener...")
    
    # 1. Enforce Monotonicity (120 >= 90 >= 07)
    final_90 = np.maximum(pred_90, pred_07)
    final_120 = np.maximum(pred_120, final_90)

    sub = test_ids.copy()
    sub['Target_07_AUC'] = pred_07
    sub['Target_07_LogLoss'] = pred_07
    sub['Target_90_AUC'] = final_90
    sub['Target_90_LogLoss'] = final_90
    sub['Target_120_AUC'] = final_120
    sub['Target_120_LogLoss'] = final_120

    # 2. THE UNTHINKABLE DAMPENER
    for col in sub.columns:
        if col != 'ID':
            sub[col] = (sub[col] * 0.999) + 0.0005

    sub.to_csv(output_path, index=False)
    print(f"\nBOOM! Pipeline complete. Submission saved to {output_path}.")