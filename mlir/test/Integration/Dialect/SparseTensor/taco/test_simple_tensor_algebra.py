# RUN: SUPPORTLIB=%mlir_lib_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import numpy as np
import os
import sys

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import mlir_pytaco_api as pt

compressed = pt.compressed
dense = pt.dense

# Ensure that we can run an unmodified PyTACO program with a simple tensor
# algebra expression using tensor index notation, and produce the expected
# result.
i, j = pt.get_index_vars(2)
A = pt.tensor([2, 3])
B = pt.tensor([2, 3])
C = pt.tensor([2, 3])
D = pt.tensor([2, 3], compressed)
A.insert([0, 1], 10)
A.insert([1, 2], 40)
B.insert([0, 0], 20)
B.insert([1, 2], 30)
C.insert([0, 1], 5)
C.insert([1, 2], 7)
D[i, j] = A[i, j] + B[i, j] - C[i, j]

indices, values = D.get_coordinates_and_values()
passed = np.array_equal(indices, [[0, 0], [0, 1], [1, 2]])
passed += np.allclose(values, [20.0, 5.0, 63.0])

# PyTACO doesn't allow the use of index values, but MLIR-PyTACO removes this
# restriction.
E = pt.tensor([3])
E[i] = i
indices, values = E.get_coordinates_and_values()
passed += np.array_equal(indices, [[0], [1], [2]])
passed += np.allclose(values, [0.0, 1.0, 2.0])

F = pt.tensor([3])
G = pt.tensor([3])
F.insert([0], 10)
F.insert([2], 40)
G[i] = F[i] + i
indices, values = G.get_coordinates_and_values()
passed += np.array_equal(indices, [[0], [1], [2]])
passed += np.allclose(values, [10.0, 1.0, 42.0])

H = pt.tensor([3])
I = pt.tensor([3])
H.insert([0], 10)
H.insert([2], 40)
I[i] = H[i] * i
indices, values = I.get_coordinates_and_values()
passed += np.array_equal(indices, [[0], [2]])
passed += np.allclose(values, [0.0, 80.0])

# CHECK: Number of passed: 8
print("Number of passed:", passed)
