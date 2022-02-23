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

# Set up scalar and sparse tensors.
alpha = pt.tensor(42.0)
S = pt.tensor([8, 8, 8],
              pt.format([pt.compressed, pt.compressed, pt.compressed]))
X = pt.tensor([8, 8, 8],
              pt.format([pt.compressed, pt.compressed, pt.compressed]))
S.insert([0, 0, 0], 2.0)
S.insert([1, 1, 1], 3.0)
S.insert([4, 4, 4], 4.0)
S.insert([7, 7, 7], 5.0)

# TODO: make this work:
# X[i, j, k] = alpha[0] * S[i, j, k]
X[i, j, k] = S[i, j, k]

expected = """; extended FROSTT format
3 4
8 8 8
1 1 1 2
2 2 2 3
5 5 5 4
8 8 8 5
"""

# Force evaluation of the kernel by writing out X.
with tempfile.TemporaryDirectory() as test_dir:
  x_file = os.path.join(test_dir, 'X.tns')
  pt.write(x_file, X)
  #
  # CHECK: Compare result True
  #
  x_data = utils.file_as_string(x_file)
  print(f'Compare result {x_data == expected}')
