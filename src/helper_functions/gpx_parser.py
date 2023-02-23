import gpxpy
import gpxpy.gpx

import numpy as np
import pandas as pd

from datetime import datetime

import haversine as hs
from haversine import Unit


def parse_gpx_basic(gpx_filename: str) -> np.array:
    """
    Function saving the parsed gpx to csv file.

    Args:
        gpx_filename (str): name of the GPX file

    Returns:
        np.array: array of data from GPX
    """
    # read and parse gpx file
    gpx_file = open(f"data/gpx/{gpx_filename}.gpx", 'r')
    gpx = gpxpy.parse(gpx_file)

    # list to keep data about each point
    route_data = []

    # latitude, longitude, elevation
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                route_data.append([float(point.latitude), 
                                   float(point.longitude), 
                                   float(point.elevation), 
                                   int(round(point.time.timestamp()))])

    # convert to numpy array         
    route_data_array = np.array(route_data)
    # save to csv
    np.savetxt(f"data/csv/{gpx_filename}.csv", route_data_array, 
               delimiter=',', header="lat,lon,elevation,time")

    return route_data_array


def parse_gpx_performance(gpx_filename: str) -> np.array:
    """
    Function saving the parsed GPX to csv file,
    when the GPX contains additional information about 
    the performance.

    Args:
        gpx_filename (str): name of the GPX file

    Returns:
        np.array: array of data from GPX
    """
    gpx_file = open(f"data/gpx/{gpx_filename}.gpx", 'r')
    gpx = gpxpy.parse(gpx_file)

    # get array with route information
    route_data_array = parse_gpx_basic(gpx_filename)

    # get the number of points 
    num_points = route_data_array.shape[0]

    performance_data = []

    # add heart rate and cadence data
    for i in range(num_points):
        hr = [el.text for el in gpx.tracks[0].segments[0].points[i].extensions[0] if 'hr' in el.tag][0]
        cadence = [el.text for el in gpx.tracks[0].segments[0].points[i].extensions[0] if 'cad' in el.tag][0]
        performance_data.append([float(hr), float(cadence)])

    performance_data_array = np.array(performance_data)

    full_data_array = np.concatenate((route_data_array, performance_data_array), 
                                      axis=1, dtype=float)

    # save to file
    np.savetxt(f"data/csv/{gpx_filename}_full.csv", full_data_array, 
               delimiter=',', header="lat,lon,elevation,time,hr,cadence")

    return full_data_array


def prepare_elevation_profile(csv_filename: str) -> tuple[np.array, np.array]:
    """
    Function preparing the distance and elevation arrays.

    Args:
        csv_filename (str): the name of the csv file with data

    Returns:
        tuple[np.array, np.array]: arrays of distances and elevations
    """
    data = pd.read_csv(f"data/csv/{csv_filename}.csv")

    route_points = data.groupby(['# lat', 'lon'], sort=False) \
                   .agg({'time': 'max', 'elevation': 'max'}).reset_index()

    route_points.columns = ['lat', 'lon', 'time', 'elevation']

    coordinates = route_points[['lat', 'lon']].to_numpy()

    points_num = coordinates.shape[0]
    distances = []

    for i in range(points_num - 1):
        loc1 = (coordinates[i, 0], coordinates[i, 1])
        loc2 = (coordinates[i + 1, 0], coordinates[i + 1, 1])
    
        distance = hs.haversine(loc1, loc2, unit=Unit.METERS)

        distances.append(distance)
    
    distances.insert(0, 0)
    distances_column = np.cumsum(np.array([distances]).T)

    elevation_column = route_points.iloc[:, 3].to_numpy()

    return (distances_column, elevation_column)






