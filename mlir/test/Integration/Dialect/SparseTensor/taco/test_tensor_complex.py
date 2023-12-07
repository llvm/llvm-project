# RUN: env SUPPORTLIB=%mlir_c_runner_utils %PYTHON %s | FileCheck %s
import numpy as np
import os
import sys

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import mlir_pytaco_api as pt

compressed = pt.compressed

passed = 0
all_types = [pt.complex64, pt.complex128]
for t in all_types:
    i, j = pt.get_index_vars(2)
    A = pt.tensor([2, 3], dtype=t)
    B = pt.tensor([2, 3], dtype=t)
    C = pt.tensor([2, 3], compressed, dtype=t)
    A.insert([0, 1], 10 + 20j)
    A.insert([1, 2], 40 + 0.5j)
    B.insert([0, 0], 20)
    B.insert([1, 2], 30 + 15j)
    C[i, j] = A[i, j] + B[i, j]

    indices, values = C.get_coordinates_and_values()
    passed += isinstance(values[0], t.value)
    passed += np.array_equal(indices, [[0, 0], [0, 1], [1, 2]])
    passed += np.allclose(values, [20, 10 + 20j, 70 + 15.5j])

# CHECK: Number of passed: 6
print("Number of passed:", passed)
