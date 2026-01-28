import csv
import datetime as dt
import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import firebase_admin
from firebase_admin import credentials, db
import serial  # pyserial


# ---------------------------
# Serial wrapper for Arduino
# ---------------------------
class SerialPort:
    """
    Minimal wrapper around pyserial to behave similarly to your original SocketSerial:
    - readline() waits until a newline '\n' is received
    - close() closes the serial port
    """

    def __init__(self, port='COM3', baudrate=115200, timeout=0.1):
        # For Windows COM10+ use r'\\.\COM10' if needed
        self.port = port
        try:
            self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Could not open serial port {port}: {e}")
        print(f"Opened serial {port} @ {baudrate} baud")
        self.buffer = b''

    def readline(self):
        # Block until newline present in buffer, but sleep briefly to be CPU-friendly
        while b'\n' not in self.buffer:
            try:
                avail = self.ser.in_waiting
                if avail:
                    data = self.ser.read(avail)
                else:
                    # read a single byte (blocks up to timeout)
                    data = self.ser.read(1)
                if not data:
                    time.sleep(0.01)
                    continue
                self.buffer += data
            except serial.SerialException as e:
                raise ConnectionError(f"Serial read error: {e}")

        i = self.buffer.index(b'\n')
        line = self.buffer[:i + 1]
        self.buffer = self.buffer[i + 1:]
        return line.decode(errors='ignore')

    def close(self):
        try:
            if hasattr(self, 'ser') and self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass


# ---------------------------
# Helpers and detectors
# ---------------------------
def parse_serial_line(line):
    """
    Expects a CSV line of floats terminated by newline.
    Original code used: return parts[0], parts[2], parts[3]
    That implies the Arduino produces at least 4 comma-separated floats,
    where parts[0] is ultrasonic distance, parts[2] is flow1, parts[3] is flow2.

    If your Arduino prints a different order (e.g. distance,flow1,flow2),
    change the indices here.
    """
    try:
        parts = list(map(float, line.strip().split(",")))
        if len(parts) >= 4:
            return parts[0], parts[2], parts[3]
        else:
            return None
    except Exception:
        return None


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def create_csv_if_not_exists(path, headers):
    ensure_dir(path)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)


def append_leakage_anomaly(path, minute, flow1, flow2, leak_error):
    ensure_dir(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([minute.isoformat(), f"{flow1:.6f}", f"{flow2:.6f}", f"{leak_error:.6f}"])


def append_ultrasonic_anomaly(path, minute, distance, predicted, train_rmse, error, weekday, hhmm, method):
    ensure_dir(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            minute.isoformat(),
            f"{distance:.6f}",
            f"{predicted:.6f}",
            f"{train_rmse:.6f}",
            f"{error:.6f}",
            weekday,
            hhmm,
            method
        ])


class WeeklyLSTM:
    def __init__(self, seq_len=2, models_dir=None):
        self.seq_len = seq_len
        self.timeslot_data = {}
        self.models = {}
        self.models_dir = models_dir or "models_weekly"
        os.makedirs(self.models_dir, exist_ok=True)

    def model_paths(self, weekday, hhmm):
        safe_hhmm = hhmm.replace(":", "")
        model_path = os.path.join(self.models_dir, f"model_w{weekday}_{safe_hhmm}.h5")
        scaler_X_path = os.path.join(self.models_dir, f"scalerX_w{weekday}_{safe_hhmm}.pkl")
        scaler_y_path = os.path.join(self.models_dir, f"scalery_w{weekday}_{safe_hhmm}.pkl")
        meta_path = os.path.join(self.models_dir, f"meta_w{weekday}_{safe_hhmm}.pkl")
        return model_path, scaler_X_path, scaler_y_path, meta_path

    def add_data_point(self, weekday, hhmm, value):
        key = (weekday, hhmm)
        self.timeslot_data.setdefault(key, []).append(float(value))
        if len(self.timeslot_data[key]) > 52:
            self.timeslot_data[key] = self.timeslot_data[key][-52:]

    def can_train_lstm(self, weekday, hhmm):
        key = (weekday, hhmm)
        return key in self.timeslot_data and len(self.timeslot_data[key]) >= (self.seq_len + 1)

    def prepare_weekly_sequences(self, values):
        X, y = [], []
        for i in range(len(values) - self.seq_len):
            X.append(values[i:i + self.seq_len])
            y.append(values[i + self.seq_len])
        if not X:
            return None, None
        return np.array(X), np.array(y)

    def train_for_timeslot(self, weekday, hhmm, epochs=40):
        key = (weekday, hhmm)
        if not self.can_train_lstm(weekday, hhmm):
            return None
        values = self.timeslot_data[key]
        X, y = self.prepare_weekly_sequences(values)
        if X is None:
            return None

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_flat = X.reshape(-1, 1)
        X_scaled_flat = scaler_X.fit_transform(X_flat)
        X_scaled = X_scaled_flat.reshape(X.shape[0], X.shape[1], 1)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        model = Sequential([
            LSTM(16, input_shape=(self.seq_len, 1)),
            Dense(8, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=1, verbose=0)

        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        train_rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

        self.models[key] = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'rmse': train_rmse
        }

        model_path, scaler_X_path, scaler_y_path, meta_path = self.model_paths(weekday, hhmm)
        try:
            model.save(model_path)
            with open(scaler_X_path, "wb") as f:
                pickle.dump(scaler_X, f)
            with open(scaler_y_path, "wb") as f:
                pickle.dump(scaler_y, f)
            with open(meta_path, "wb") as f:
                pickle.dump({'rmse': train_rmse, 'seq_len': self.seq_len}, f)
        except Exception as e:
            print("Warning: could not save model/scalers:", e)

        return self.models[key]

    def load_model_from_disk(self, weekday, hhmm):
        model_path, scaler_X_path, scaler_y_path, meta_path = self.model_paths(weekday, hhmm)
        key = (weekday, hhmm)
        if not os.path.exists(model_path) or not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
            return False
        try:
            model = load_model(model_path)
            with open(scaler_X_path, "rb") as f:
                scaler_X = pickle.load(f)
            with open(scaler_y_path, "rb") as f:
                scaler_y = pickle.load(f)
            rmse = None
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f)
                        rmse = meta.get('rmse', None)
                except Exception:
                    rmse = None
            self.models[key] = {'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y, 'rmse': rmse}
            return True
        except Exception as e:
            print("Warning: failed to load model for", key, e)
            return False

    def predict_for_timeslot(self, weekday, hhmm):
        key = (weekday, hhmm)
        if key not in self.models:
            return None, None
        if key not in self.timeslot_data:
            return None, None
        values = self.timeslot_data[key]
        if len(values) < self.seq_len:
            return None, None

        recent = np.array(values[-self.seq_len:])
        scaler_X = self.models[key]['scaler_X']
        model = self.models[key]['model']
        scaler_y = self.models[key]['scaler_y']

        recent_scaled = scaler_X.transform(recent.reshape(-1, 1)).reshape(1, self.seq_len, 1)
        y_pred_scaled = model.predict(recent_scaled)
        pred = float(scaler_y.inverse_transform(y_pred_scaled)[0][0])
        rmse = self.models[key].get('rmse', None)
        return pred, rmse


class StatisticalDetector:
    def __init__(self):
        self.timeslot_stats = {}

    def update_stats(self, weekday, hhmm, value):
        key = (weekday, hhmm)
        s = self.timeslot_stats.setdefault(key, {'values': [], 'mean': 0.0, 'std': 0.0, 'count': 0})
        s['values'].append(float(value))
        if len(s['values']) > 52:
            s['values'] = s['values'][-52:]
        s['count'] = len(s['values'])
        s['mean'] = float(np.mean(s['values']))
        s['std'] = float(np.std(s['values'])) if s['count'] > 1 else 0.0

    def check_anomaly(self, weekday, hhmm, value, threshold_z=3.0):
        key = (weekday, hhmm)
        if key not in self.timeslot_stats:
            return False, None, None, None
        s = self.timeslot_stats[key]
        if s['count'] < 3:
            return False, s['mean'], s['std'], None
        if s['std'] == 0:
            error = abs(value - s['mean'])
            return error > 5.0, s['mean'], s['std'], None
        z = abs(value - s['mean']) / s['std']
        return z > threshold_z, s['mean'], s['std'], z


def init_firebase(json_key_path, firebase_url):
    try:
        cred = credentials.Certificate(json_key_path)
        firebase_admin.initialize_app(cred, {'databaseURL': firebase_url})
    except Exception as e:
        print("Warning: Firebase init failed:", e)


def push_to_firebase(node, data_dict):
    try:
        ref = db.reference(node)
        ref.push(data_dict)
    except Exception as e:
        print("Warning: Firebase push failed:", e)


# ---------------------------
# Main
# ---------------------------
def main(base_path,
         leakage_threshold,
         lstm_threshold_factor,
         firebase_key,
         firebase_url,
         serial_port='COM3',
         baudrate=115200,
         seq_len=2):
    minute_csv = base_path + "_minutes.csv"
    flow_anomaly_csv = base_path + "_flow_anomalies.csv"
    ultrasonic_anomaly_csv = base_path + "_ultrasonic_anomalies.csv"
    models_dir = base_path + "_models"

    # Add a new "warning" column to the minute CSV to record per-minute warnings
    create_csv_if_not_exists(minute_csv, [
        "minute_start_iso", "weekday", "hhmm",
        "avg_flow_lpm1", "avg_flow_lpm2", "avg_ultrasonic_cm",
        "reading_count", "warning"
    ])
    create_csv_if_not_exists(flow_anomaly_csv, ["minute_start_iso", "flow1", "flow2", "leakage_error"])
    create_csv_if_not_exists(ultrasonic_anomaly_csv, [
        "minute_start_iso", "ultrasonic_distance", "predicted",
        "train_rmse", "error", "weekday", "hhmm", "method"
    ])

    weekly_lstm = WeeklyLSTM(seq_len=seq_len, models_dir=models_dir)
    stats_detector = StatisticalDetector()

    timeslot_row_counts = {}

    # Preload historical minute CSV (if present) to seed timeslot_data and stats
    try:
        df = pd.read_csv(minute_csv, usecols=lambda c: c in [
            'minute_start_iso', 'weekday', 'hhmm', 'avg_ultrasonic_cm', 'reading_count'
        ], on_bad_lines='skip')
        for _, row in df.iterrows():
            try:
                minute_dt = dt.datetime.fromisoformat(row['minute_start_iso'])
                wk = minute_dt.weekday()
                hhmm = row['hhmm'] if pd.notnull(
                    row.get('hhmm', None)) else f"{minute_dt.hour:02d}:{minute_dt.minute:02d}"
            except Exception:
                wk_name = row.get('weekday', None)
                if wk_name and pd.notnull(wk_name):
                    try:
                        wk = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(
                            str(wk_name))
                    except Exception:
                        continue
                else:
                    continue
                hhmm = row.get('hhmm', None)
                if not hhmm or pd.isnull(hhmm):
                    continue

            if 'avg_ultrasonic_cm' not in row or pd.isnull(row['avg_ultrasonic_cm']):
                continue
            val = float(row['avg_ultrasonic_cm'])

            key = (wk, hhmm)
            timeslot_row_counts[key] = timeslot_row_counts.get(key, 0) + 1
            weekly_lstm.add_data_point(wk, hhmm, val)
            stats_detector.update_stats(wk, hhmm, val)

    except pd.errors.EmptyDataError:
        pass
    except Exception as e:
        print("Warning: could not preload historical CSV (handled gracefully):", e)

    # Try loading models or train short ones at startup if enough rows
    required_rows = seq_len + 1
    for key, count in list(timeslot_row_counts.items()):
        wk, hhmm = key
        loaded = weekly_lstm.load_model_from_disk(wk, hhmm)
        if loaded:
            print(f"Loaded existing model for {wk} {hhmm}")
        else:
            if count >= required_rows and weekly_lstm.can_train_lstm(wk, hhmm):
                print(f"Training model at startup for timeslot {wk} {hhmm} (rows={count}) ...")
                res = weekly_lstm.train_for_timeslot(wk, hhmm, epochs=40)
                if res is not None:
                    print(f"  Trained model for {wk} {hhmm}, rmse={res['rmse']:.3f}")

    init_firebase(firebase_key, firebase_url)

    # Open serial port (Arduino)
    ser = None
    try:
        ser = SerialPort(port=serial_port, baudrate=baudrate, timeout=0.1)
    except Exception as e:
        print("Fatal: could not open serial port:", e)
        return

    buf_f1, buf_f2, buf_us = [], [], []
    current_minute = None

    print("Running. Ctrl+C to stop.")
    try:
        while True:
            now = dt.datetime.now()
            minute = now.replace(second=0, microsecond=0)
            if current_minute and minute != current_minute:
                avg_f1 = float(np.mean(buf_f1)) if buf_f1 else 0.0
                avg_f2 = float(np.mean(buf_f2)) if buf_f2 else 0.0
                avg_us = float(np.mean(buf_us)) if buf_us else 0.0
                reading_count = len(buf_us)

                wk = current_minute.weekday()
                hhmm = f"{current_minute.hour:02d}:{current_minute.minute:02d}"
                key = (wk, hhmm)

                # Determine flow anomaly
                flow_anom = False
                leak_error = abs(avg_f1 - avg_f2)
                if leak_error > leakage_threshold:
                    flow_anom = True
                    append_leakage_anomaly(flow_anomaly_csv, current_minute, avg_f1, avg_f2, leak_error)
                    push_to_firebase("flow_anomalies", {
                        "minute_start_iso": current_minute.isoformat(),
                        "flow1": avg_f1,
                        "flow2": avg_f2,
                        "leakage_error": leak_error
                    })

                # Update detectors with ultrasonic reading
                weekly_lstm.add_data_point(wk, hhmm, avg_us)
                stats_detector.update_stats(wk, hhmm, avg_us)

                model_ready = ((timeslot_row_counts.get(key, 0) >= required_rows) and (
                            (key in weekly_lstm.models) or weekly_lstm.can_train_lstm(wk, hhmm)))

                pred, rmse = (None, None)
                method = None
                us_anom = False
                error = None

                # Try LSTM-based detection first
                if key in weekly_lstm.models:
                    pred, rmse = weekly_lstm.predict_for_timeslot(wk, hhmm)
                    method = "lstm"
                    if pred is not None:
                        rmse_val = rmse if rmse is not None else 1e-6
                        threshold = lstm_threshold_factor * rmse_val
                        error = abs(avg_us - pred)
                        if error > threshold:
                            us_anom = True
                else:
                    if timeslot_row_counts.get(key, 0) >= required_rows:
                        loaded = weekly_lstm.load_model_from_disk(wk, hhmm)
                        if loaded:
                            print("Loaded model from disk for", key)
                            pred, rmse = weekly_lstm.predict_for_timeslot(wk, hhmm)
                            method = "lstm"
                            if pred is not None:
                                rmse_val = rmse if rmse is not None else 1e-6
                                threshold = lstm_threshold_factor * rmse_val
                                error = abs(avg_us - pred)
                                if error > threshold:
                                    us_anom = True
                        else:
                            if weekly_lstm.can_train_lstm(wk, hhmm):
                                res = weekly_lstm.train_for_timeslot(wk, hhmm, epochs=40)
                                if res is not None:
                                    print(f"Trained model for {wk} {hhmm} at runtime, rmse={res['rmse']:.3f}")
                                    pred, rmse = weekly_lstm.predict_for_timeslot(wk, hhmm)
                                    method = "lstm"
                                    if pred is not None:
                                        rmse_val = rmse if rmse is not None else 1e-6
                                        threshold = lstm_threshold_factor * rmse_val
                                        error = abs(avg_us - pred)
                                        if error > threshold:
                                            us_anom = True

                # If LSTM didn't flag anomaly or isn't used, fallback to statistical
                if not us_anom and (method is None or pred is None):
                    is_anom, mean_v, std_v, z = stats_detector.check_anomaly(wk, hhmm, avg_us, threshold_z=3.0)
                    if mean_v is not None:
                        method = "statistical"
                        pred = mean_v
                        if is_anom:
                            us_anom = True
                            error = abs(avg_us - mean_v)

                # If ultrasonic anomaly detected, append and push
                if us_anom:
                    append_ultrasonic_anomaly(
                        ultrasonic_anomaly_csv,
                        current_minute,
                        avg_us,
                        pred if pred is not None else 0.0,
                        rmse if rmse is not None else 0.0,
                        error if error is not None else 0.0,
                        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][wk],
                        hhmm,
                        method if method else "unknown"
                    )
                    push_to_firebase("ultrasonic_anomalies", {
                        "minute_start_iso": current_minute.isoformat(),
                        "ultrasonic_distance": avg_us,
                        "predicted": pred if pred is not None else None,
                        "train_rmse": rmse if rmse is not None else None,
                        "error": error if error is not None else None,
                        "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][wk],
                        "hhmm": hhmm,
                        "method": method
                    })

                # Decide warning text for minute CSV and possible combined alert
                warning_text = ""
                if flow_anom and us_anom:
                    warning_text = "leakage_detected"
                    # send a combined alert to Firebase
                    push_to_firebase("alerts", {
                        "minute_start_iso": current_minute.isoformat(),
                        "type": "leakage_detected",
                        "flow_leak_error": leak_error,
                        "ultrasonic_error": error if error is not None else None,
                        "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][wk],
                        "hhmm": hhmm
                    })
                elif flow_anom:
                    warning_text = "flow_warning"
                elif us_anom:
                    warning_text = "ultrasonic_warning"

                # Write minute CSV (including the warning_text)
                with open(minute_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        current_minute.isoformat(),
                        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][wk],
                        hhmm,
                        avg_f1,
                        avg_f2,
                        avg_us,
                        reading_count,
                        warning_text
                    ])

                # Also include warning in the waterflow_minutes Firebase node
                push_to_firebase("waterflow_minutes", {
                    "minute_start_iso": current_minute.isoformat(),
                    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][wk],
                    "hhmm": hhmm,
                    "avg_flow_lpm1": avg_f1,
                    "avg_flow_lpm2": avg_f2,
                    "avg_ultrasonic_cm": avg_us,
                    "reading_count": reading_count,
                    "warning": warning_text
                })

                # Update bookkeeping and buffers
                timeslot_row_counts[key] = timeslot_row_counts.get(key, 0) + 1

                buf_f1.clear()
                buf_f2.clear()
                buf_us.clear()

            current_minute = minute

            try:
                line = ser.readline()
            except ConnectionError as e:
                print("Arduino disconnected, stopping main loop. Error:", e)
                break
            except Exception as e:
                print("Warning: read error:", e)
                time.sleep(0.1)
                continue

            parsed = parse_serial_line(line)
            if parsed:
                d, f1, f2 = parsed
                buf_us.append(d)
                buf_f1.append(f1)
                buf_f2.append(f2)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if ser:
            ser.close()


if __name__ == "__main__":
    main(
        base_path="wf_data/waterflow",
        leakage_threshold=2.0,
        lstm_threshold_factor=3.0,
        firebase_key=r"C:\Users\John Tafalla\PycharmProjects\multisensor\.venv\wf_data\firebase_key.json",
        firebase_url="https://hello-f1194-default-rtdb.firebaseio.com/",
        serial_port='COM3',
        baudrate=115200,
        seq_len=2
    )
