import argparse
import pandas as pd
import numpy as np
import librosa
import os
import joblib

def extract_features(audio_path, pause_start, sr=16000, window_sec=1.5):
    duration = pause_start
    offset = max(0, duration - window_sec)
    load_duration = min(duration, window_sec)
    
    try:
        y, _ = librosa.load(audio_path, sr=sr, offset=offset, duration=load_duration)
    except:
        return np.zeros(6)
        
    if len(y) == 0:
        return np.zeros(6)
        
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    f0 = f0[f0 > 0]
    
    if len(f0) > 0:
        f0_mean = np.mean(f0)
        f0_std = np.std(f0)
        if len(f0) >= 10:
            f0_slope = np.mean(f0[-5:]) - np.mean(f0[:5])
        else:
            f0_slope = 0
    else:
        f0_mean, f0_std, f0_slope = 0, 0, 0
        
    return np.array([rms_mean, rms_std, f0_mean, f0_std, f0_slope, duration])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing audio and labels.csv")
    parser.add_argument("--out", required=True, help="Output CSV file for predictions")
    args = parser.parse_args()
    
    labels_path = os.path.join(args.data_dir, "labels.csv")
    df = pd.read_csv(labels_path)
    
    # Load the pretrained model (path resolved relative to this script,
    # not the caller's working directory, so this works no matter where
    # predict.py is invoked from)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    try:
        clf = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: model.pkl not found at {model_path}. Please run train.py first.")
        return
        
    predictions = []
    
    for _, row in df.iterrows():
        audio_path = os.path.join(args.data_dir, row['audio_file'])
        features = extract_features(audio_path, row['pause_start'])
        
        # Predict probability of class 1 ('eot')
        p_eot = clf.predict_proba(features.reshape(1, -1))[0][1]
        
        predictions.append({
            'turn_id': row['turn_id'],
            'pause_index': row['pause_index'],
            'p_eot': p_eot
        })
        
    out_df = pd.DataFrame(predictions)
    out_df.to_csv(args.out, index=False)
    print(f"Predictions saved to {args.out}")

if __name__ == "__main__":
    main()