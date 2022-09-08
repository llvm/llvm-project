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

# Define the CSR format.
csr = pt.format([pt.dense, pt.compressed], [0, 1])

# Read matrices A and B from file, infer size of output matrix C.
A = pt.read(os.path.join(_SCRIPT_PATH, "data/A.mtx"), csr)
B = pt.read(os.path.join(_SCRIPT_PATH, "data/B.mtx"), csr)
C = pt.tensor([A.shape[0], B.shape[1]], csr)

# Define the kernel.
i, j, k = pt.get_index_vars(3)
C[i, j] = A[i, k] * B[k, j]

# Force evaluation of the kernel by writing out C.
with tempfile.TemporaryDirectory() as test_dir:
  golden_file = os.path.join(_SCRIPT_PATH, "data/gold_C.tns")
  out_file = os.path.join(test_dir, "C.tns")
  pt.write(out_file, C)
  #
  # CHECK: Compare result True
  #
  print(f"Compare result {utils.compare_sparse_tns(golden_file, out_file)}")
