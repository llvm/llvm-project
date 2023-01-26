# RUN: SUPPORTLIB=%mlir_lib_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import numpy as np
import os
import sys

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import mlir_pytaco_api as pt

compressed = pt.compressed

i, j = pt.get_index_vars(2)
A = pt.tensor([2, 3])
S = pt.tensor(3) # S is a scalar tensor.
B = pt.tensor([2, 3], compressed)
A.insert([0, 1], 10)
A.insert([1, 2], 40)

# Use [0] to index the scalar tensor.
B[i, j] = A[i, j] * S[0]

indices, values = B.get_coordinates_and_values()
passed = np.array_equal(indices, [[0, 1], [1, 2]])
passed += np.array_equal(values, [30.0, 120.0])

# Sum all the values in A.
S[0] = A[i, j]
passed += (S.get_scalar_value() == 50.0)

indices, values = S.get_coordinates_and_values()
passed += (len(indices)==0)
passed += (values == 50.0)

# CHECK: Number of passed: 5
print("Number of passed:", passed)
