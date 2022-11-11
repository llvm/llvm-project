# RUN: SUPPORTLIB=%mlir_lib_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import numpy as np
import os
import sys

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import mlir_pytaco_api as pt

i, j = pt.get_index_vars(2)
A = pt.tensor([2, 3])
B = pt.tensor([2, 3])
A.insert([0, 1], 10.3)
A.insert([1, 1], 40.7)
A.insert([0, 2], -11.3)
A.insert([1, 2], -41.7)

B[i, j] = abs(A[i, j])
indices, values = B.get_coordinates_and_values()
passed = np.array_equal(indices, [[0, 1], [0, 2], [1, 1], [1, 2]])
passed += np.allclose(values, [10.3, 11.3, 40.7, 41.7])

B[i, j] = pt.ceil(A[i, j])
indices, values = B.get_coordinates_and_values()
passed += np.array_equal(indices, [[0, 1], [0, 2], [1, 1], [1, 2]])
passed += np.allclose(values, [11, -11, 41, -41])

B[i, j] = pt.floor(A[i, j])
indices, values = B.get_coordinates_and_values()
passed += np.array_equal(indices, [[0, 1], [0, 2], [1, 1], [1, 2]])
passed += np.allclose(values, [10, -12, 40, -42])

B[i, j] = -A[i, j]
indices, values = B.get_coordinates_and_values()
passed += np.array_equal(indices, [[0, 1], [0, 2], [1, 1], [1, 2]])
passed += np.allclose(values, [-10.3, 11.3, -40.7, 41.7])

# CHECK: Number of passed: 8
print("Number of passed:", passed)
