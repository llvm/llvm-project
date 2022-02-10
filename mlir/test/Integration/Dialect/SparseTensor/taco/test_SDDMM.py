# RUN: SUPPORTLIB=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import filecmp
import numpy as np
import os
import sys
import tempfile

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from tools import mlir_pytaco_api as pt
from tools import testing_utils as utils

i, j, k = pt.get_index_vars(3)

# Set up dense matrices.
A = pt.from_array(np.full((8, 8), 2.0))
B = pt.from_array(np.full((8, 8), 3.0))

# Set up sparse matrices.
S = pt.tensor([8, 8], pt.format([pt.compressed, pt.compressed]))
X = pt.tensor([8, 8], pt.format([pt.compressed, pt.compressed]))
Y = pt.tensor([8, 8], pt.compressed)  # alternative syntax works too

S.insert([0, 7], 42.0)

# Define the SDDMM kernel. Since this performs the reduction as
#   sum(k, S[i, j] * A[i, k] * B[k, j])
# we only compute the intermediate dense matrix product that are actually
# needed to compute the result, with proper asymptotic complexity.
X[i, j] = S[i, j] * A[i, k] * B[k, j]

# Alternative way to define SDDMM kernel. Since this performs the reduction as
#   sum(k, A[i, k] * B[k, j]) * S[i, j]
# the MLIR lowering results in two separate tensor index expressions that
# need to be fused properly to guarantee proper asymptotic complexity.
Y[i, j] = A[i, k] * B[k, j] * S[i, j]

expected = """; extended FROSTT format
2 1
8 8
1 8 2016
"""

# Force evaluation of the kernels by writing out X and Y.
with tempfile.TemporaryDirectory() as test_dir:
  x_file = os.path.join(test_dir, "X.tns")
  y_file = os.path.join(test_dir, "Y.tns")
  pt.write(x_file, X)
  pt.write(y_file, Y)
  #
  # CHECK: Compare result True True
  #
  x_data = utils.file_as_string(x_file)
  y_data = utils.file_as_string(y_file)
  print(f"Compare result {x_data == expected} {y_data == expected}")
