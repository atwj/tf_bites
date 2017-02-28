#ml
from firebase import firebase
import pandas as pd
import numpy as np
import json
import time
from datetime import timedelta
from sklearn.neighbors import DistanceMetric, BallTree
from sklearn.cluster import DBSCAN
firebase_db = firebase.FirebaseApplication('your-firebase-endpoint',
                                           authentication=None)

# Constants
EARTH_RAD = 6371.0
EPS_CARTESIAN = 0.2 / EARTH_RAD
MINPTS_CARTESIAN = 10
EPS_RAD = 30 / (24 * 60.0) * (2 * np.pi)
MINTPTS_RAD = 10
RADIUS_DEFAULT = 0.5 / EARTH_RAD # 500m

def invert(x):
    return abs(int(x))

def getBitesDF():
    # Retrieve records from FB
    result = firebase_db.get('/bites/', None)

    # Extract timestamp
    df_timestamp = pd.DataFrame(result.keys(),columns=['timestamp'])

    # Unpack nested dict
    df_results = pd.DataFrame.from_dict(result,orient='index').stack().reset_index()

    # Create new DataFrame
    bites_df = pd.DataFrame(df_results[0].tolist())
    bites_df['timestamp'] = pd.to_datetime(df_timestamp['timestamp'].apply(invert), unit='s')
    bites_df = bites_df.set_index('timestamp',drop=True)

    return bites_df

def getHotSpots(lat_, long_):
    result = retrieveAllBiteReccords()

# To return value in mins / total mins in one day
def to_mins(x):
    x = pd.Timestamp(x)
    year = x.year
    month =  x.month
    day = x.day
    smth = (x.value - pd.Timestamp(str(year)+'-'+str(month)+'-'+str(day)).value) / (60 * (10**9))
    return smth

# Helper method to convert values to radian
def convert_to_radian(x):
    return ((x / (24*60.0)) * 2 * np.pi)

def toHHMM(mins):
    return time.strftime('%H:%M:%S', time.gmtime(mins*60))

def extract_clusters(labels, data):
    cluster_dict_ = {}
    clusters = set(labels)
    # print(data.shape)
    for k in clusters:
        indices_of_k_, = np.where(labels == k)
        points = data.take(indices_of_k_,axis=0).reshape(1, len(indices_of_k_)).squeeze()
        back_to_mins_ = 24 * 60 / (2 * np.pi)
        cluster_dict_[k] = {
            'points': points.tolist(),
            'start_end' : '{0},{1}'.format(str(toHHMM(points.min() * back_to_mins_)),
                str(toHHMM(points.max() *  back_to_mins_)))
            }
    return cluster_dict_

def getTimeClusters(df):
    print df
    print df.index
    tmin = np.vectorize(to_mins)
    trad = np.vectorize(convert_to_radian)
    input_rad = trad(tmin(df.index))

    # Convert time to rad points
    X = input_rad[None,:] - input_rad[:,None]

    # Assign 'shortest distance to each point
    X[((X > np.pi) & (X <= (2*np.pi)))] = X[((X > np.pi) & (X <= (2*np.pi)))] -(2*np.pi)
    X[((X > (-2*np.pi)) & (X <= (-1*np.pi)))] = X[((X > (-2*np.pi)) & (X <= (-1*np.pi)))] + (2*np.pi)
    X = abs(X)

    db = DBSCAN(eps=EPS_RAD,min_samples=MINTPTS_RAD, metric='precomputed')
    db.fit(X)

    csi = db.core_sample_indices_
    components = db.components_
    labels = db.labels_
    return extract_clusters([x for x in labels if x >= 0], input_rad)

def latLongtoRad(x):
    return x * (np.pi / 180)

toRad_vec = np.vectorize(latLongtoRad)

def getReport(start_date_, end_date_):
    report = {}
    bites_df = getBitesDF()
    # Drop un-needed columns
    bites_df = bites_df.drop('image_Base64', 1)
    bites_df = bites_df.drop('image_name', 1)

    # Filter by dates, return false if invalid range given
    if(np.datetime64(start_date_) > np.datetime64(end_date_)):
        return False

    bites_df = bites_df.ix[start_date_:end_date_]
    rad_bites_df = pd.DataFrame()
    rad_bites_df['timestamp'] = bites_df.index
    rad_bites_df['lat_rad'] = toRad_vec(bites_df['latitude'])
    rad_bites_df['long_rad'] = toRad_vec(bites_df['longitude'])
    rad_bites_df = rad_bites_df.set_index('timestamp', drop=True)

    # Run DBSCAN to cluster hotspots
    X_bites = rad_bites_df.as_matrix()
    db = DBSCAN(eps=EPS_CARTESIAN, min_samples=MINPTS_CARTESIAN, metric='haversine')
    db.fit(X_bites)

    csi = db.core_sample_indices_
    components = db.components_
    labels = db.labels_

    # Get hotspots, -1 is noise
    hotspots = set([x for x in labels if x >= 0])
    # bites_df_noiseless = bites_df.ix[np.where(labels != -1)]
    # print bites_df_noiseless
    # print np.where(labels == 0)
    # print bites_df_noiseless.ix[np.where(labels == 0)]

    for hotspot in hotspots:
        print hotspot
        lat_long = bites_df.ix[np.where(labels == hotspot)].to_dict(orient='records')
        timeClusters = getTimeClusters(bites_df.ix[np.where(labels == hotspot)])
        report[hotspot] = {'loc_clusters':lat_long, 't_clusters': timeClusters}
    return report

def getMosquitoActivity(lat, long):
    bites_df = getBitesDF()
    bites_df = bites_df.drop('image_Base64', 1)
    bites_df = bites_df.drop('image_name', 1)
    day_1_ = pd.to_datetime(int(time.time()), unit='s')
    day_0_ = day_1_ - timedelta(days=7)
    mask = ((bites_df.index < day_1_) & (bites_df.index > day_0_))

    bites_df = bites_df.loc[mask]
    rad_bites_df = pd.DataFrame()
    rad_bites_df['timestamp'] = bites_df.index
    rad_bites_df['lat_rad'] = toRad_vec(bites_df['latitude'])
    rad_bites_df['long_rad'] = toRad_vec(bites_df['longitude'])
    rad_bites_df = rad_bites_df.set_index('timestamp', drop=True)


    bt = BallTree(rad_bites_df.as_matrix(), metric='haversine')
    indices,distances = bt.query_radius([latLongtoRad(float(lat)), latLongtoRad(float(long))], r=RADIUS_DEFAULT, return_distance=True)
    print indices
    print distances

    nn_list = indices[0].tolist()
    return bites_df.iloc[nn_list,:].to_dict(orient='records')
