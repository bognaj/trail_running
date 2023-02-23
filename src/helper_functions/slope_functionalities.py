import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def get_slope_at_distance(distance: float, x: np.array, y: np.array, 
                          num_points:int=10001, plot:bool=False) -> tuple[float, float, float]:
    """
    Function returning the slope at given point of the course.

    Args:
        distance (float): distance already run for which the slope
                          is supposed to be calculated
        x (np.array): points of distance
        y (np.array): elevation values at x points
        num_points (int, optional): Number of points of piecewise-linear interpolation. 
                                    Defaults to 10001.
        plot (bool, optional): Indicator whether to plot the interpolation. 
                               Defaults to False.

    Returns:
        tuple[float, float, float]: the slope expressed as tangent, in radians
                                    and in degrees
    """
    x_lower = min(x)
    x_upper = max(x)

    if distance == x_lower:
        slope = 0
        radians = np.arctan(slope)
        degrees = radians * 180/np.pi
        return (slope, radians, degrees)

    elif distance == x_upper:
        slope = 0
        radians = np.arctan(slope)
        degrees = radians * 180/np.pi
        return (slope, radians, degrees)

    f = interpolate.interp1d(x, y, kind='linear')
    eps = 1/(num_points - 1) * 0.1
    x_f = np.linspace(x_lower, x_upper, num_points)
    y_f = f(x_f)

    if plot:
        plt.scatter(x, y)
        plt.plot(x_f, y_f, 'r')

    dist_lower = distance - eps
    dist_upper = distance + eps

    f_dist_lower = f(dist_lower)
    f_dist_upper = f(dist_upper)
    f_dist = f(distance)

    delta_lower = (f_dist - f_dist_lower) / eps
    delta_upper = (f_dist_upper - f_dist) / eps

    if np.abs(delta_upper - delta_lower) < 10**(-9):
        slope = delta_lower
    else:
        dist_upper = distance - eps
        dist_lower = distance - 2 * eps

        f_dist_lower = f(dist_lower)
        f_dist_upper = f(dist_upper)

        slope = (f_dist_upper - f_dist_lower) / eps

    radians = np.arctan(slope)
    degrees = radians * 180/np.pi

    return (slope, radians, degrees)