# RUN: env SUPPORTLIB=%mlir_c_runner_utils %PYTHON %s | FileCheck %s

import numpy as np
import os
import sys

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import mlir_pytaco_api as pt

i, j = pt.get_index_vars(2)
# Both tensors are true dense tensors.
A = pt.from_array(np.full([2, 3], 1, dtype=np.float64))
B = pt.from_array(np.full([2, 3], 2, dtype=np.float64))
# Define the result tensor as a true dense tensor. The parameter is_dense=True
# is an MLIR-PyTACO extension.
C = pt.tensor([2, 3], dtype=pt.float64, is_dense=True)

C[i, j] = A[i, j] + B[i, j]

# CHECK: [3. 3. 3. 3. 3. 3.]
print(C.to_array().reshape(6))
