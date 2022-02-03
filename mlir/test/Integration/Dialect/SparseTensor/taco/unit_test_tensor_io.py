# RUN: SUPPORTLIB=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

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

# Define the aliases to shorten the code.
_COMPRESSED = mlir_pytaco.ModeFormat.COMPRESSED
_DENSE = mlir_pytaco.ModeFormat.DENSE


def _run(f):
  print(f.__name__)
  f()
  return f


_FORMAT = mlir_pytaco.Format([_COMPRESSED, _COMPRESSED])
_MTX_DATA_TEMPLATE = Template(
    """%%MatrixMarket matrix coordinate real $general_or_symmetry
3 3 3
3 1 3
1 2 2
3 2 4
""")


def _get_mtx_data(value):
  mtx_data = _MTX_DATA_TEMPLATE
  return mtx_data.substitute(general_or_symmetry=value)


# CHECK-LABEL: test_read_mtx_matrix_general
@_run
def test_read_mtx_matrix_general():
  with tempfile.TemporaryDirectory() as test_dir:
    file_name = os.path.join(test_dir, "data.mtx")
    with open(file_name, "w") as file:
      file.write(_get_mtx_data("general"))
    a = mlir_pytaco_io.read(file_name, _FORMAT)
  passed = 0
  # The value of a is stored as an MLIR sparse tensor.
  passed += (not a.is_unpacked())
  a.unpack()
  passed += (a.is_unpacked())
  coords, values = a.get_coordinates_and_values()
  passed += np.allclose(coords, [[0, 1], [2, 0], [2, 1]])
  passed += np.allclose(values, [2.0, 3.0, 4.0])
  # CHECK: 4
  print(passed)


# CHECK-LABEL: test_read_mtx_matrix_symmetry
@_run
def test_read_mtx_matrix_symmetry():
  with tempfile.TemporaryDirectory() as test_dir:
    file_name = os.path.join(test_dir, "data.mtx")
    with open(file_name, "w") as file:
      file.write(_get_mtx_data("symmetric"))
    a = mlir_pytaco_io.read(file_name, _FORMAT)
  passed = 0
  # The value of a is stored as an MLIR sparse tensor.
  passed += (not a.is_unpacked())
  a.unpack()
  passed += (a.is_unpacked())
  coords, values = a.get_coordinates_and_values()
  print(coords)
  print(values)
  passed += np.allclose(coords,
                        [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])
  passed += np.allclose(values, [2.0, 3.0, 2.0, 4.0, 3.0, 4.0])
  # CHECK: 4
  print(passed)


_TNS_DATA = """2 3
3 2
3 1 3
1 2 2
3 2 4
"""


# CHECK-LABEL: test_read_tns
@_run
def test_read_tns():
  with tempfile.TemporaryDirectory() as test_dir:
    file_name = os.path.join(test_dir, "data.tns")
    with open(file_name, "w") as file:
      file.write(_TNS_DATA)
    a = mlir_pytaco_io.read(file_name, _FORMAT)
  passed = 0
  # The value of a is stored as an MLIR sparse tensor.
  passed += (not a.is_unpacked())
  a.unpack()
  passed += (a.is_unpacked())
  coords, values = a.get_coordinates_and_values()
  passed += np.allclose(coords, [[0, 1], [2, 0], [2, 1]])
  passed += np.allclose(values, [2.0, 3.0, 4.0])
  # CHECK: 4
  print(passed)
