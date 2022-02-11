# RUN: SUPPORTLIB=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

from string import Template

import numpy as np
import os
import sys
import tempfile

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import mlir_pytaco
from tools import testing_utils as testing_utils

# Define the aliases to shorten the code.
_COMPRESSED = mlir_pytaco.ModeFormat.COMPRESSED
_DENSE = mlir_pytaco.ModeFormat.DENSE


# CHECK-LABEL: test_tensor_all_dense_sparse
@testing_utils.run_test
def test_tensor_all_dense_sparse():
  a = mlir_pytaco.Tensor([4], [_DENSE])
  passed = (not a.is_dense())
  passed += (a.order == 1)
  passed += (a.shape[0] == 4)
  # CHECK: Number of passed: 3
  print("Number of passed:", passed)


# CHECK-LABEL: test_tensor_true_dense
@testing_utils.run_test
def test_tensor_true_dense():
  a = mlir_pytaco.Tensor.from_array(np.random.uniform(size=5))
  passed = a.is_dense()
  passed += (a.order == 1)
  passed += (a.shape[0] == 5)
  # CHECK: Number of passed: 3
  print("Number of passed:", passed)
