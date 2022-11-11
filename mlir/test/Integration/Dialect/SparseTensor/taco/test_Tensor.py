# RUN: SUPPORTLIB=%mlir_lib_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import filecmp
import numpy as np
import os
import sys
import tempfile

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from tools import mlir_pytaco_api as pt
from tools import testing_utils as utils

i, j, k, l, m = pt.get_index_vars(5)

# Set up scalar.
alpha = pt.tensor(42.0)

# Set up some sparse tensors with different dim annotations and ordering.
S = pt.tensor([8, 8, 8],
              pt.format([pt.compressed, pt.dense, pt.compressed], [1, 0, 2]))
X = pt.tensor([8, 8, 8],
              pt.format([pt.compressed, pt.compressed, pt.compressed],
                        [1, 0, 2]))
S.insert([0, 0, 0], 2.0)
S.insert([1, 1, 1], 3.0)
S.insert([4, 4, 4], 4.0)
S.insert([7, 7, 7], 5.0)

X[i, j, k] = alpha[0] * S[i, j, k]

# Set up tensors with a dense last dimension. This results in a full
# enveloping storage of all last "rows" with one or more nonzeros.
T = pt.tensor([1, 2, 3, 4, 5],
              pt.format([
                  pt.compressed, pt.compressed, pt.compressed, pt.compressed,
                  pt.dense
              ]))
Y = pt.tensor([1, 2, 3, 4, 5],
              pt.format([
                  pt.compressed, pt.compressed, pt.compressed, pt.compressed,
                  pt.dense
              ]))
T.insert([0, 1, 2, 3, 4], -2.0)

Y[i, j, k, l, m] = alpha[0] * T[i, j, k, l, m]

# Set up a sparse tensor and dense tensor with different access.
U = pt.tensor([2, 3], pt.format([pt.compressed, pt.compressed], [1, 0]))
Z = pt.tensor([3, 2], pt.format([pt.dense, pt.dense]))
U.insert([1, 2], 3.0)

Z[i, j] = alpha[0] * U[j, i]

x_expected = """; extended FROSTT format
3 4
8 8 8
1 1 1 84
2 2 2 126
5 5 5 168
8 8 8 210
"""

y_expected = """; extended FROSTT format
5 5
1 2 3 4 5
1 2 3 4 1 0
1 2 3 4 2 0
1 2 3 4 3 0
1 2 3 4 4 0
1 2 3 4 5 -84
"""

z_expected = """; extended FROSTT format
2 6
3 2
1 1 0
1 2 0
2 1 0
2 2 0
3 1 0
3 2 126
"""

# Force evaluation of the kernel by writing out X.
with tempfile.TemporaryDirectory() as test_dir:
  x_file = os.path.join(test_dir, 'X.tns')
  pt.write(x_file, X)
  y_file = os.path.join(test_dir, 'Y.tns')
  pt.write(y_file, Y)
  z_file = os.path.join(test_dir, 'Z.tns')
  pt.write(z_file, Z)
  #
  # CHECK: Compare result True True True
  #
  x_data = utils.file_as_string(x_file)
  y_data = utils.file_as_string(y_file)
  z_data = utils.file_as_string(z_file)
  print(
      f'Compare result {x_data == x_expected} {y_data == y_expected} {z_data == z_expected}'
  )
