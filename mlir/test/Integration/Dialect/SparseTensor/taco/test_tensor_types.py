# RUN: SUPPORTLIB=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import numpy as np
import os
import sys

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import mlir_pytaco_api as pt

compressed = pt.compressed
dense = pt.dense

passed = 0
all_types = [
    pt.int8, pt.int16, pt.int32, pt.int64, pt.float16, pt.float32, pt.float64
]
for t in all_types:
  i, j = pt.get_index_vars(2)
  A = pt.tensor([2, 3], dtype=t)
  B = pt.tensor([2, 3], dtype=t)
  C = pt.tensor([2, 3], compressed, dtype=t)
  A.insert([0, 1], 10)
  A.insert([1, 2], 40)
  B.insert([0, 0], 20)
  B.insert([1, 2], 30)
  C[i, j] = A[i, j] + B[i, j]

  indices, values = C.get_coordinates_and_values()
  passed += isinstance(values[0], t.value)
  passed += np.array_equal(indices, [[0, 0], [0, 1], [1, 2]])
  passed += np.allclose(values, [20.0, 10.0, 70.0])

# CHECK: Number of passed: 21
print("Number of passed:", passed)
