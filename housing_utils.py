import os
from pathlib import Path

from urlpath import URL
from requests import get

import matplotlib.pyplot as plt


def download_data(url, filename, data_dir):
    """Download a dataset from url, and store it in data_dir
    :param url: online location of data set to download
    :param filename: name to use for stored file
    :param data_dir: location to store data on this computer
    :returns: error message or success message
    :rtype: string
    """
    os.makedirs(data_dir, exist_ok=True)
    r = get(url)
    file = data_dir / filename
    if not file.is_file():
        with file.open("w", encoding="utf-8") as f:
            f.write(r.text)
    return "file downloaded successfully"


HOUSING_URL = URL(
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
)
HOUSING_DIR = Path("datasets/housing/")


def download_housing_data(
    url=HOUSING_URL, file_name="housing.csv", data_dir=HOUSING_DIR
):
    """wrapper around download_data to download the housing data from handson-ml
    :param url: url to housing_data
    :param file_name: name to use for stored data
    :param data_dir: directory for stored data
    :returns: error or success message
    :rtype: string
    """
    download_data(url, file_name, data_dir)


def save_fig(
    fig_id,
    tight_layout=True,
    fig_extension="png",
    resolution=300,
    img_path="",
):
    """Simple function to save the most recently created figure in a
    standardised way

    :param fig_id: Name for the saved figure file
    :param tight_layout: Boolean denoted whether to use a tight layout
    (see matplotlib documentation)
    :param fig_extension: file type to be used for the figure
    :param resolution: image resolution of the saved image
    :param img_path: path to store image. Default is current folder
    :returns: None
    :rtype: None

    """
    path = os.path.join(img_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def true_false_plot(
    y_true,
    y_pred,
    fig_id,
    tight_layout=True,
    fig_extension="png",
    resolution=300,
    img_path="",
):
    """Function to draw true vs false predictions scatter plot and save to file

    :param y_true: array of true values
    :param y_pred: array of predicted values
    :param fig_id: name of saved figure
    :param tight_layout: Boolean denoted whether to use a tight layout
    (see matplotlib documentation)
    :param fig_extension: file type to be used for the figure
    :param resolution: image resolution of the saved image
    :param img_path: path to store image. Default is current folder
    :returns: Name of stored figure
    :rtype: string

    """
    plt.scatter(y_true, y_pred)
    xpoints = ypoints = plt.xlim()
    plt.plot(
        xpoints,
        ypoints,
        linestyle="--",
        color="k",
        lw=3,
        scalex=False,
        scaley=False,
    )
    plt.xlabel("True Median House Value")
    plt.ylabel("Predicted Median House Value")
    plt.title("True vs Predicted House Value")
    save_fig(fig_id, resolution=resolution, img_path=img_path)
    path = os.path.join(img_path, fig_id + "." + fig_extension)
    return ("stored plot to in:", path)
