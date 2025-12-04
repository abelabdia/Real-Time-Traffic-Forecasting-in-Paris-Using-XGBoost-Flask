from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
import json
import os
import pytz

app = Flask(__name__)

# Cache file for processed data
CACHE_FILE = 'temp_traffic_cache.csv'
CACHE_TIMEOUT = 3600  # 1 hour in seconds

# Timezone - Paris timezone
TZ = pytz.timezone('Europe/Paris')

# Load all models (1h, 2h, 3h)
model_1727_1h = joblib.load('saved_models/xgb_model_1727.pkl')
model_6670_1h = joblib.load('saved_models/xgb_model_6670.pkl')

model_1727_2h = joblib.load('saved_models/xgb_model_1727_h2.pkl')
model_6670_2h = joblib.load('saved_models/xgb_model_6670_h2.pkl')

model_1727_3h = joblib.load('saved_models/xgb_model_1727_h3.pkl')
model_6670_3h = joblib.load('saved_models/xgb_model_6670_h3.pkl')

API_URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents/records"

def get_traffic_data():
    """Fetch traffic data from Paris API or use cache"""
    # MODIFICATION: Ajouter un indicateur de source
    data_source = {
        'from_cache': False,
        'cache_age_seconds': None,
        'message': ''
    }
    
    # Try to use cache first
    if os.path.exists(CACHE_FILE):
        cache_time = os.path.getmtime(CACHE_FILE)
        current_time = datetime.now().timestamp()
        cache_age = current_time - cache_time
        
        if cache_age < CACHE_TIMEOUT:
            try:
                cache_data = pd.read_csv(CACHE_FILE)
                cached_results = cache_data.to_dict('records')
                print(f"[CACHE] Using cached data from {datetime.fromtimestamp(cache_time)}")
                
                data_source['from_cache'] = True
                data_source['cache_age_seconds'] = int(cache_age)
                data_source['message'] = f"Données du cache (il y a {int(cache_age/60)} min)"
                
                return cached_results, None, data_source
            except Exception as e:
                print(f"[CACHE] Error reading cache: {e}")
    
    # Fetch from API
    try:
        full_url = (
            f"{API_URL}?"
            "select=iu_ac%2C%20t_1h%2C%20k&"
            "where=iu_ac%20%3D%201727%20or%20iu_ac%20%3D%206670&"
            "order_by=t_1h%20DESC&"
            "limit=96&"
            "refine=libelle%3A%22Aubervilliers%22&"
            "refine=libelle%3A%22Av_Pte_Aubervilliers%22&"
            "refine=libelle_nd_amont%3A%22Pte_Aubervilliers-Hermite-Mail%22&"
            "refine=libelle_nd_amont%3A%22Aubervilliers-face_191%22"
        )
        
        print(f"[API] Fetching traffic data...")
        
        response = requests.get(full_url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'results' not in data or len(data['results']) == 0:
            return None, "Aucune donnée disponible", data_source
        
        print(f"[API] Retrieved {len(data['results'])} records")
        
        arcs_found = set([r['iu_ac'] for r in data['results']])
        print(f"[API] Arcs trouvés: {arcs_found}")
        
        # Save to cache
        try:
            cache_df = pd.DataFrame(data['results'])
            cache_df.to_csv(CACHE_FILE, index=False)
            print(f"[CACHE] Data saved to cache at {datetime.now()}")
        except Exception as e:
            print(f"[CACHE] Error saving to cache: {e}")
        
        data_source['from_cache'] = False
        data_source['message'] = "✅ Données récupérées depuis l'API de Paris"
        
        return data['results'], None, data_source
    except requests.exceptions.RequestException as e:
        print(f"[API] Error: {str(e)}")
        
        # Fallback to cache even if expired
        if os.path.exists(CACHE_FILE):
            try:
                cache_data = pd.read_csv(CACHE_FILE)
                cached_results = cache_data.to_dict('records')
                cache_time = os.path.getmtime(CACHE_FILE)
                cache_age = datetime.now().timestamp() - cache_time
                
                print(f"[CACHE] Using expired cache (from {datetime.fromtimestamp(cache_time)})")
                
                data_source['from_cache'] = True
                data_source['cache_age_seconds'] = int(cache_age)
                data_source['message'] = f"⚠️ API indisponible, cache expiré (il y a {int(cache_age/60)} min)"
                
                return cached_results, "⚠️ API indisponible, utilisation du cache", data_source
            except Exception as cache_error:
                return None, f"Erreur API et cache: {str(e)}", data_source
        
        return None, f"Erreur API: {str(e)}", data_source

def get_current_state(api_data, arc_id):
    """Get the current state (most recent data) for an arc - for NOW button"""
    # Filter data for specific arc
    arc_data = [d for d in api_data if int(d['iu_ac']) == arc_id and d['k'] is not None]
    
    if not arc_data:
        return None, f"Aucune donnée disponible pour arc {arc_id}"
    
    # Sort by time and get most recent
    arc_data = sorted(arc_data, key=lambda x: x['t_1h'], reverse=True)
    most_recent = arc_data[0]
    
    # Utiliser l'heure actuelle du système au lieu de t_1h
    now_paris = datetime.now(TZ)
    current_time_formatted = now_paris.strftime('%Y-%m-%d %H:%M')
    
    return {
        'arc_id': arc_id,
        'prediction': round(float(most_recent['k']), 2),
        'next_hour': current_time_formatted,  # Heure actuelle du système
        'horizon': 'now'
    }, None

def prepare_features(api_data, arc_id, horizon='1h'):
    """Prepare features from API data for prediction based on horizon"""
    # Filter data for specific arc
    arc_data = [d for d in api_data if int(d['iu_ac']) == arc_id and d['k'] is not None]
    arc_data = sorted(arc_data, key=lambda x: x['t_1h'])
    
    # Minimum data points needed depends on horizon
    min_points = {'1h': 6, '2h': 6, '3h': 6}[horizon]
    
    if len(arc_data) < min_points:
        return None, f"Pas assez de données historiques pour arc {arc_id} ({len(arc_data)} points, besoin de {min_points})"
    
    # Create DataFrame
    df = pd.DataFrame(arc_data)
    df['t_1h'] = pd.to_datetime(df['t_1h'])
    df = df.sort_values('t_1h').reset_index(drop=True)
    df['taux_arc'] = df['k'].astype(float)
    
    print(f"[DEBUG] Arc {arc_id} ({horizon}): {len(df)} records, last values: {df['taux_arc'].tail(3).tolist()}")
    
    # Get CURRENT time in Paris timezone
    now_paris = datetime.now(TZ)
    current_time = now_paris.replace(minute=0, second=0, microsecond=0)
    
    # Calculate next hour based on horizon
    hours_ahead = {'1h': 1, '2h': 2, '3h': 3}[horizon]
    next_hour = current_time + timedelta(hours=hours_ahead)
    
    # Extract time features for target hour
    hour = next_hour.hour
    dayofweek = next_hour.weekday()
    month = next_hour.month
    
    # Cyclic encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * dayofweek / 7)
    day_cos = np.cos(2 * np.pi * dayofweek / 7)
    
    # Get values for THIS arc
    values = df['taux_arc'].values
    
    # Prepare lags based on horizon
    if horizon == '1h':
        lags_needed = 3
    elif horizon == '2h':
        lags_needed = 5
    else:  # 3h
        lags_needed = 6
    
    lag_this = []
    for i in range(1, lags_needed + 1):
        if len(values) >= i:
            lag_this.append(float(values[-i]))
        else:
            lag_this.append(float(values[-1]))
    
    # Rolling features
    roll3 = float(np.mean(values[-3:]))
    roll6 = float(np.mean(values[-6:])) if len(values) >= 6 else roll3
    
    print(f"[DEBUG] Arc {arc_id}: lags={lag_this[:3]}, roll3={roll3:.2f}, roll6={roll6:.2f}")
    
    # Get OTHER arc data
    other_arc = 6670 if arc_id == 1727 else 1727
    other_arc_data = [d for d in api_data if int(d['iu_ac']) == other_arc and d['k'] is not None]
    other_arc_data = sorted(other_arc_data, key=lambda x: x['t_1h'])
    
    if other_arc_data:
        other_df = pd.DataFrame(other_arc_data)
        other_values = other_df['k'].astype(float).values
        
        lag_other = []
        for i in range(1, lags_needed + 1):
            if len(other_values) >= i:
                lag_other.append(float(other_values[-i]))
            else:
                lag_other.append(float(other_values[-1] if len(other_values) > 0 else lag_this[0]))
        
        roll3_other = float(np.mean(other_values[-3:])) if len(other_values) >= 3 else roll3
        roll6_other = float(np.mean(other_values[-6:])) if len(other_values) >= 6 else roll6
    else:
        lag_other = lag_this.copy()
        roll3_other = roll3
        roll6_other = roll6
    
    # Build feature dict based on horizon and arc
    feature_dict = {
        'hour': hour,
        'dayofweek': dayofweek,
        'month': month,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos,
    }
    
    # Add lags based on arc and horizon
    if arc_id == 1727:
        for i in range(lags_needed):
            feature_dict[f'lag_1727_{i+1}'] = lag_this[i]
            feature_dict[f'lag_6670_{i+1}'] = lag_other[i]
    else:  # 6670
        for i in range(lags_needed):
            feature_dict[f'lag_1727_{i+1}'] = lag_other[i]
            feature_dict[f'lag_6670_{i+1}'] = lag_this[i]
    
    # Add rolling features
    if arc_id == 1727:
        feature_dict['roll3_1727'] = roll3
        feature_dict['roll6_1727'] = roll6
        feature_dict['roll3_6670'] = roll3_other
        feature_dict['roll6_6670'] = roll6_other
    else:
        feature_dict['roll3_1727'] = roll3_other
        feature_dict['roll6_1727'] = roll6_other
        feature_dict['roll3_6670'] = roll3
        feature_dict['roll6_6670'] = roll6
    
    feature_dict['next_hour'] = next_hour.strftime('%Y-%m-%d %H:%M')
    feature_dict['arc_id'] = arc_id
    
    return feature_dict, None

def predict_traffic(features, arc_id, horizon='1h'):
    """Make prediction for specific arc and horizon"""
    # Select the right model
    if horizon == '1h':
        model = model_1727_1h if arc_id == 1727 else model_6670_1h
        lags_needed = 3
    elif horizon == '2h':
        model = model_1727_2h if arc_id == 1727 else model_6670_2h
        lags_needed = 5
    else:  # 3h
        model = model_1727_3h if arc_id == 1727 else model_6670_3h
        lags_needed = 6
    
    # Build feature array in correct order
    feature_list = [
        features['hour'],
        features['dayofweek'],
        features['month'],
        features['hour_sin'],
        features['hour_cos'],
        features['day_sin'],
        features['day_cos'],
    ]
    
    # Add lags (1727 then 6670)
    for i in range(1, lags_needed + 1):
        feature_list.append(features[f'lag_1727_{i}'])
    for i in range(1, lags_needed + 1):
        feature_list.append(features[f'lag_6670_{i}'])
    
    # Add rolling features
    feature_list.extend([
        features['roll3_1727'],
        features['roll6_1727'],
        features['roll3_6670'],
        features['roll6_6670']
    ])
    
    feature_array = np.array([feature_list])
    
    print(f"[DEBUG] Predicting {horizon} for arc {arc_id}: {feature_array.shape[1]} features")
    
    try:
        prediction = model.predict(feature_array)[0]
        return float(prediction), None
    except Exception as e:
        return None, f"Erreur prédiction: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Main prediction endpoint"""
    try:
        data = request.json
        horizon = data.get('horizon', '1h')  # 'now', '1h', '2h', or '3h'
        
        print(f"[INFO] Prediction request for horizon: {horizon}")
        
        # Get API data
        api_data, error, data_source = get_traffic_data()
        if error:
            return jsonify({'success': False, 'error': error}), 400
        
        results = {}
        
        # NOUVEAU: Si horizon est "now", retourner l'état actuel
        if horizon == 'now':
            for arc_id in [1727, 6670]:
                current_state, error = get_current_state(api_data, arc_id)
                if error:
                    print(f"[ERROR] Arc {arc_id}: {error}")
                    return jsonify({'success': False, 'error': f"Arc {arc_id}: {error}"}), 400
                
                results[f'arc_{arc_id}'] = current_state
        else:
            # Comportement normal pour 1h, 2h, 3h
            for arc_id in [1727, 6670]:
                # Prepare features
                features, error = prepare_features(api_data, arc_id, horizon)
                if error:
                    print(f"[ERROR] Arc {arc_id}: {error}")
                    return jsonify({'success': False, 'error': f"Arc {arc_id}: {error}"}), 400
                
                # Make prediction
                prediction, error = predict_traffic(features, arc_id, horizon)
                if error:
                    print(f"[ERROR] Prediction failed: {error}")
                    return jsonify({'success': False, 'error': error}), 400
                
                results[f'arc_{arc_id}'] = {
                    'arc_id': arc_id,
                    'prediction': round(prediction, 2),
                    'next_hour': features['next_hour'],
                    'horizon': horizon
                }
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'predictions': results,
            'data_source': data_source
        })
    except Exception as e:
        print(f"[ERROR] Unexpected error in /api/predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f"Erreur serveur: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """Check API connectivity and data availability"""
    api_data, error, data_source = get_traffic_data()
    
    if error:
        return jsonify({'connected': False, 'error': error}), 400
    
    # Count by arc
    arc_counts = {}
    for record in api_data:
        arc = int(record.get('iu_ac'))
        arc_counts[arc] = arc_counts.get(arc, 0) + 1
    
    cache_info = {}
    if os.path.exists(CACHE_FILE):
        cache_time = os.path.getmtime(CACHE_FILE)
        cache_info = {
            'cache_exists': True,
            'cache_age_seconds': int(datetime.now().timestamp() - cache_time),
            'cache_file': CACHE_FILE
        }
    
    return jsonify({
        'connected': True,
        'records_fetched': len(api_data),
        'arc_counts': arc_counts,
        'timestamp': datetime.now().isoformat(),
        'cache': cache_info,
        'data_source': data_source
    })

@app.route('/api/debug', methods=['GET'])
def api_debug():
    """Debug endpoint to see raw API data"""
    api_data, error, data_source = get_traffic_data()
    
    if error:
        return jsonify({'error': error}), 400
    
    # Count records per arc
    arc_1727 = [d for d in api_data if int(d['iu_ac']) == 1727]
    arc_6670 = [d for d in api_data if int(d['iu_ac']) == 6670]
    
    return jsonify({
        'total_records': len(api_data),
        'arc_1727_count': len(arc_1727),
        'arc_6670_count': len(arc_6670),
        'sample_record': api_data[0] if api_data else None,
        'iu_ac_types': list(set([type(d['iu_ac']).__name__ for d in api_data])),
        'iu_ac_values': list(set([d['iu_ac'] for d in api_data])),
        'models_loaded': {
            '1h': True,
            '2h': True,
            '3h': True
        },
        'data_source': data_source
    })

@app.route('/api/raw-data', methods=['GET'])
def api_raw_data():
    """Voir les données brutes de l'API avec détails"""
    api_data, error, data_source = get_traffic_data()
    
    if error:
        return jsonify({'error': error}), 400
    
    # Grouper par arc
    data_by_arc = {}
    for record in api_data:
        arc = int(record.get('iu_ac'))
        if arc not in data_by_arc:
            data_by_arc[arc] = []
        data_by_arc[arc].append({
            't_1h': record.get('t_1h'),
            'k': record.get('k'),
            'iu_ac': arc
        })
    
    return jsonify({
        'total_records': len(api_data),
        'arcs_found': list(data_by_arc.keys()),
        'data_by_arc': data_by_arc,
        'arc_1727_count': len(data_by_arc.get(1727, [])),
        'arc_6670_count': len(data_by_arc.get(6670, [])),
        'data_source': data_source
    })

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Vider le cache et forcer un nouveau chargement"""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            return jsonify({
                'success': True,
                'message': 'Cache supprimé avec succès'
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Aucun cache à supprimer'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
    