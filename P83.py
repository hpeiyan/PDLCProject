from absl import app
from absl import flags
from absl import logging
import sys

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

SEED = 2017  # what's this

FLAGS = flags.FLAGS
flags.DEFINE_string(name='echos', default=None, help='text to echo')


def main(args):
    del args

    # print('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info),
    #       file=sys.stderr)
    # logging.info('echos is %s.', FLAGS.echos)

    iris = load_iris()
    idxs = np.where(iris.target < 2)
    x = iris.data[idxs]
    y = iris.target[idxs]

    # plt.scatter(x[y == 0][:, 0], x[y == 0][:, 2], color='green', lable='data')
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 2], color='green', label='data')
    # plt.scatter(x[y == 1][:, 0], x[y == 1][:, 2], color='red', lable='target')
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 2], color='red', label='target')
    plt.title('iris\'s database')
    plt.xlabel('sepal length in cm')
    plt.ylabel('sepal width in cm')
    plt.legend()
    plt.show()

    pass


if __name__ == '__main__':
    app.run(main)
