# RUN: env SUPPORTLIB=%mlir_c_runner_utils %PYTHON %s | FileCheck %s

from string import Template

import numpy as np
import os
import sys
import tempfile

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import mlir_pytaco
from tools import mlir_pytaco_io
from tools import mlir_pytaco_utils as pytaco_utils
from tools import testing_utils as testing_utils


# Define the aliases to shorten the code.
_COMPRESSED = mlir_pytaco.ModeFormat.COMPRESSED
_DENSE = mlir_pytaco.ModeFormat.DENSE


_FORMAT = mlir_pytaco.Format([_COMPRESSED, _COMPRESSED])
_MTX_DATA = """%%MatrixMarket matrix coordinate real general
3 3 3
3 1 3
1 2 2
3 2 4
"""


# CHECK-LABEL: test_read_mtx_matrix_general
@testing_utils.run_test
def test_read_mtx_matrix_general():
    with tempfile.TemporaryDirectory() as test_dir:
        file_name = os.path.join(test_dir, "data.mtx")
        with open(file_name, "w") as file:
            file.write(_MTX_DATA)
        a = mlir_pytaco_io.read(file_name, _FORMAT)
    passed = 0
    # The value of a is stored as an MLIR sparse tensor.
    passed += not a.is_unpacked()
    a.unpack()
    passed += a.is_unpacked()
    coords, values = a.get_coordinates_and_values()
    passed += np.array_equal(coords, [[0, 1], [2, 0], [2, 1]])
    passed += np.allclose(values, [2.0, 3.0, 4.0])
    # CHECK: 4
    print(passed)


_TNS_DATA = """2 3
3 2
3 1 3
1 2 2
3 2 4
"""


# CHECK-LABEL: test_read_tns
@testing_utils.run_test
def test_read_tns():
    with tempfile.TemporaryDirectory() as test_dir:
        file_name = os.path.join(test_dir, "data.tns")
        with open(file_name, "w") as file:
            file.write(_TNS_DATA)
        a = mlir_pytaco_io.read(file_name, _FORMAT)
    passed = 0
    # The value of a is stored as an MLIR sparse tensor.
    passed += not a.is_unpacked()
    a.unpack()
    passed += a.is_unpacked()
    coords, values = a.get_coordinates_and_values()
    passed += np.array_equal(coords, [[0, 1], [2, 0], [2, 1]])
    passed += np.allclose(values, [2.0, 3.0, 4.0])
    # CHECK: 4
    print(passed)


# CHECK-LABEL: test_write_unpacked_tns
@testing_utils.run_test
def test_write_unpacked_tns():
    a = mlir_pytaco.Tensor([2, 3])
    a.insert([0, 1], 10)
    a.insert([1, 2], 40)
    a.insert([0, 0], 20)
    with tempfile.TemporaryDirectory() as test_dir:
        file_name = os.path.join(test_dir, "data.tns")
        try:
            mlir_pytaco_io.write(file_name, a)
        except ValueError as e:
            # CHECK: Writing unpacked sparse tensors to file is not supported
            print(e)


# CHECK-LABEL: test_write_packed_tns
@testing_utils.run_test
def test_write_packed_tns():
    a = mlir_pytaco.Tensor([2, 3])
    a.insert([0, 1], 10)
    a.insert([1, 2], 40)
    a.insert([0, 0], 20)
    b = mlir_pytaco.Tensor([2, 3])
    i, j = mlir_pytaco.get_index_vars(2)
    b[i, j] = a[i, j] + a[i, j]
    with tempfile.TemporaryDirectory() as test_dir:
        file_name = os.path.join(test_dir, "data.tns")
        mlir_pytaco_io.write(file_name, b)
        with open(file_name, "r") as file:
            lines = file.readlines()
    passed = 0
    # Skip the comment line in the output.
    if lines[1:] == ["2 3\n", "2 3\n", "1 1 40\n", "1 2 20\n", "2 3 80\n"]:
        passed = 1
    # CHECK: 1
    print(passed)
