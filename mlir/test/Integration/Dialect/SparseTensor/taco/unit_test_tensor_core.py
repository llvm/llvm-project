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


def _init_3d(T, I, J, K):
  for i in range(I):
    for j in range(J):
      for k in range(K):
        T.insert([i, j, k], i + j + k + 1)


def _init_2d(T, I, J):
  for i in range(I):
    for j in range(J):
      T.insert([i, j], i + j + 1)


def _init_1d_with_value(T, I, v):
  for i in range(I):
    T.insert([i], v)


def test_expect_error(name, code, error):
  """Executes the code then verifies the expected error message."""
  try:
    exec(code)
  except ValueError as e:
    passed = "passed" if (str(e).startswith(error)) else "failed"
    print(f"test_{name}: {passed}")


# CHECK-LABEL: test_tensor_dtype
@testing_utils.run_test
def test_tensor_dtype():
  passed = mlir_pytaco.DType(mlir_pytaco.Type.INT16).is_int()
  passed += mlir_pytaco.DType(mlir_pytaco.Type.INT32).is_int()
  passed += mlir_pytaco.DType(mlir_pytaco.Type.INT64).is_int()
  passed += mlir_pytaco.DType(mlir_pytaco.Type.FLOAT32).is_float()
  passed += mlir_pytaco.DType(mlir_pytaco.Type.FLOAT64).is_float()
  # CHECK: Number of passed: 5
  print("Number of passed:", passed)


# CHECK: test_mode_ordering_not_int: passed
test_expect_error("mode_ordering_not_int",
                  "m = mlir_pytaco.ModeOrdering(['x'])",
                  "Ordering must be a list of integers")

# CHECK: test_mode_ordering_not_permutation: passed
test_expect_error("mode_ordering_not_permutation",
                  "m = mlir_pytaco.ModeOrdering([2, 1])", "Invalid ordering")

# CHECK: test_mode_format_invalid: passed
test_expect_error("mode_format_invalid",
                  "m = mlir_pytaco.ModeFormatPack(['y'])",
                  "Formats must be a list of ModeFormat")

# CHECK: test_expect_mode_format_pack: passed
test_expect_error("expect_mode_format_pack", ("""
mode_ordering = mlir_pytaco.ModeOrdering([0, 1, 2])
f = mlir_pytaco.Format(["x"], mode_ordering)
    """), "Expected a list of ModeFormat")

# CHECK: test_expect_mode_ordering: passed
test_expect_error("expect_mode_ordering", ("""
mode_format_pack = mlir_pytaco.ModeFormatPack([_COMPRESSED, _COMPRESSED])
f = mlir_pytaco.Format(mode_format_pack, "x")
    """), "Expected ModeOrdering")

# CHECK: test_inconsistent_mode_format_pack_and_mode_ordering: passed
test_expect_error("inconsistent_mode_format_pack_and_mode_ordering", ("""
mode_format_pack = mlir_pytaco.ModeFormatPack([_COMPRESSED, _COMPRESSED])
mode_ordering = mlir_pytaco.ModeOrdering([0, 1, 2])
f = mlir_pytaco.Format(mode_format_pack, mode_ordering)
    """), "Inconsistent ModeFormatPack and ModeOrdering")


# CHECK-LABEL: test_format_default_ordering
@testing_utils.run_test
def test_format_default_ordering():
  f = mlir_pytaco.Format([_COMPRESSED, _COMPRESSED])
  passed = 0
  passed += np.array_equal(f.ordering.ordering, [0, 1])
  # CHECK: Number of passed: 1
  print("Number of passed:", passed)


# CHECK-LABEL: test_format_explicit_ordering
@testing_utils.run_test
def test_format_explicit_ordering():
  f = mlir_pytaco.Format([_COMPRESSED, _DENSE], [1, 0])
  passed = 0
  passed += np.array_equal(f.ordering.ordering, [1, 0])
  # CHECK: Number of passed: 1
  print("Number of passed:", passed)


# CHECK-LABEL: test_index_var
@testing_utils.run_test
def test_index_var():
  i = mlir_pytaco.IndexVar()
  j = mlir_pytaco.IndexVar()
  passed = (i.name != j.name)

  vars = mlir_pytaco.get_index_vars(10)
  passed += (len(vars) == 10)
  passed += (all([isinstance(e, mlir_pytaco.IndexVar) for e in vars]))

  # CHECK: Number of passed: 3
  print("Number of passed:", passed)


# CHECK: test_tensor_invalid_first_argument: passed
test_expect_error("tensor_invalid_first_argument",
                  "t = mlir_pytaco.Tensor('f')", "Invalid first argument")

# CHECK: test_tensor_inconsistent_shape_and_format: passed
test_expect_error("tensor_inconsistent_shape_and_format", ("""
mode_format_pack = mlir_pytaco.ModeFormatPack([_COMPRESSED, _COMPRESSED])
mode_ordering = mlir_pytaco.ModeOrdering([0, 1])
f = mlir_pytaco.Format(mode_format_pack, mode_ordering)
t = mlir_pytaco.Tensor([3], f)
    """), "Inconsistent shape and format")

# CHECK: test_tensor_invalid_format: passed
test_expect_error("tensor_invalid_format", "t = mlir_pytaco.Tensor([3], 'f')",
                  "Invalid format argument")

# CHECK: test_tensor_insert_nonlist_coordinate: passed
test_expect_error("tensor_insert_nonlist_coordinate", ("""
t = mlir_pytaco.Tensor([3])
t.insert(1, 0)
    """), "Non list coordinate detected")

# CHECK: test_tensor_insert_too_much_coordinate: passed
test_expect_error("tensor_insert_too_much_coordinate", ("""
t = mlir_pytaco.Tensor([3])
t.insert([0, 0], 0)
    """), "Invalid coordinate")

# CHECK: test_tensor_insert_coordinate_outof_range: passed
test_expect_error("tensor_insert_coordinate_outof_range", ("""
t = mlir_pytaco.Tensor([1, 1])
t.insert([1, 0], 0)
    """), "Invalid coordinate")

# CHECK: test_tensor_insert_coordinate_nonint: passed
test_expect_error("tensor_insert_coordinate_nonint", ("""
t = mlir_pytaco.Tensor([1, 1])
t.insert([0, "xy"], 0)
    """), "Non integer coordinate detected")

# CHECK: test_tensor_insert_invalid_value: passed
test_expect_error("tensor_insert_invalid_value", ("""
t = mlir_pytaco.Tensor([1, 1])
t.insert([0, 0], "x")
    """), "Value is neither int nor float")

# CHECK: test_access_non_index_var_index: passed
test_expect_error("access_non_index_var_index", ("""
t = mlir_pytaco.Tensor([5, 6])
i = mlir_pytaco.IndexVar()
a = mlir_pytaco.Access(t, (i, "j"))
    """), "Indices contain non IndexVar")

# CHECK: test_access_inconsistent_rank_indices: passed
test_expect_error("access_inconsistent_rank_indices", ("""
t = mlir_pytaco.Tensor([5, 6])
i = mlir_pytaco.IndexVar()
a = mlir_pytaco.Access(t, (i,))
    """), "Invalid indices for rank")

# CHECK: test_access_invalid_indices_for_rank: passed
test_expect_error("access_invalid_indices_for_rank", ("""
t = mlir_pytaco.Tensor([5, 6])
i, j, k = mlir_pytaco.get_index_vars(3)
a = mlir_pytaco.Access(t, (i,j, k))
    """), "Invalid indices for rank")

# CHECK: test_invalid_indices: passed
test_expect_error("invalid_indices", ("""
i, j = mlir_pytaco.get_index_vars(2)
A = mlir_pytaco.Tensor([2, 3])
B = mlir_pytaco.Tensor([2, 3])
C = mlir_pytaco.Tensor([2, 3], _DENSE)
C[i, j] = A[1, j] + B[i, j]
    """), "Expected IndexVars")

# CHECK: test_inconsistent_rank_indices: passed
test_expect_error("inconsistent_rank_indices", ("""
i, j = mlir_pytaco.get_index_vars(2)
A = mlir_pytaco.Tensor([2, 3])
C = mlir_pytaco.Tensor([2, 3], _DENSE)
C[i, j] = A[i]
    """), "Invalid indices for rank")

# CHECK: test_destination_index_not_used_in_source: passed
test_expect_error("destination_index_not_used_in_source", ("""
i, j = mlir_pytaco.get_index_vars(2)
A = mlir_pytaco.Tensor([3])
C = mlir_pytaco.Tensor([3], _DENSE)
C[j] = A[i]
C.evaluate()
    """), "Destination IndexVar not used in the source expression")

# CHECK: test_destination_dim_not_consistent_with_source: passed
test_expect_error("destination_dim_not_consistent_with_source", ("""
i = mlir_pytaco.IndexVar()
A = mlir_pytaco.Tensor([3])
C = mlir_pytaco.Tensor([5], _DENSE)
C[i] = A[i]
C.evaluate()
    """), "Inconsistent destination dimension for IndexVar")

# CHECK: test_inconsistent_source_dim: passed
test_expect_error("inconsistent_source_dim", ("""
i = mlir_pytaco.IndexVar()
A = mlir_pytaco.Tensor([3])
B = mlir_pytaco.Tensor([5])
C = mlir_pytaco.Tensor([3], _DENSE)
C[i] = A[i] + B[i]
C.evaluate()
    """), "Inconsistent source dimension for IndexVar")

# CHECK: test_index_var_outside_domain: passed
test_expect_error("index_var_outside_domain", ("""
i, j = mlir_pytaco.get_index_vars(2)
A = mlir_pytaco.Tensor([3])
B = mlir_pytaco.Tensor([3])
B[i] = A[i] + j
B.evaluate()
    """), "IndexVar is not part of the iteration domain")


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


# CHECK-LABEL: test_tensor_copy
@testing_utils.run_test
def test_tensor_copy():
  i, j = mlir_pytaco.get_index_vars(2)
  I = 2
  J = 3
  A = mlir_pytaco.Tensor([I, J])
  A.insert([0, 1], 5.0)
  A.insert([1, 2], 6.0)
  B = mlir_pytaco.Tensor([I, J])
  B[i, j] = A[i, j]
  passed = (B._assignment is not None)
  passed += (B._engine is None)
  try:
    B.compute()
  except ValueError as e:
    passed += (str(e).startswith("Need to invoke compile"))
  B.compile()
  passed += (B._engine is not None)
  B.compute()
  passed += (B._assignment is None)
  passed += (B._engine is None)
  indices, values = B.get_coordinates_and_values()
  passed += np.array_equal(indices, [[0, 1], [1, 2]])
  passed += np.allclose(values, [5.0, 6.0])
  # No temporary tensor is used.
  passed += (B._stats.get_total() == 0)
  # CHECK: Number of passed: 9
  print("Number of passed:", passed)


# CHECK-LABEL: test_tensor_trivial_reduction
@testing_utils.run_test
def test_tensor_trivial_reduction():
  i, j = mlir_pytaco.get_index_vars(2)
  I = 2
  J = 3
  A = mlir_pytaco.Tensor([I, J])
  A.insert([0, 1], 5.0)
  A.insert([0, 2], 3.0)
  A.insert([1, 2], 6.0)
  B = mlir_pytaco.Tensor([I])
  B[i] = A[i, j]
  indices, values = B.get_coordinates_and_values()
  passed = np.array_equal(indices, [[0], [1]])
  passed += np.allclose(values, [8.0, 6.0])
  # No temporary tensor is used.
  passed += (B._stats.get_total() == 0)

  # CHECK: Number of passed: 3
  print("Number of passed:", passed)


# CHECK-LABEL: test_binary_add
@testing_utils.run_test
def test_binary_add():
  i = mlir_pytaco.IndexVar()
  A = mlir_pytaco.Tensor([4])
  B = mlir_pytaco.Tensor([4])
  C = mlir_pytaco.Tensor([4])
  A.insert([1], 10)
  A.insert([2], 1)
  B.insert([3], 20)
  B.insert([2], 2)
  C[i] = A[i] + B[i]
  indices, values = C.get_coordinates_and_values()
  passed = np.array_equal(indices, [[1], [2], [3]])
  passed += np.array_equal(values, [10., 3., 20.])
  # No temporary tensor is used.
  passed += (C._stats.get_total() == 0)
  # CHECK: Number of passed: 3
  print("Number of passed:", passed)


# CHECK-LABEL: test_binary_add_sub
@testing_utils.run_test
def test_binary_add_sub():
  i = mlir_pytaco.IndexVar()
  j = mlir_pytaco.IndexVar()
  A = mlir_pytaco.Tensor([2, 3])
  B = mlir_pytaco.Tensor([2, 3])
  C = mlir_pytaco.Tensor([2, 3])
  D = mlir_pytaco.Tensor([2, 3])
  A.insert([0, 1], 10)
  A.insert([1, 2], 40)
  B.insert([0, 0], 20)
  B.insert([1, 2], 30)
  C.insert([0, 1], 5)
  C.insert([1, 2], 7)
  D[i, j] = A[i, j] + B[i, j] - C[i, j]
  indices, values = D.get_coordinates_and_values()
  passed = np.array_equal(indices, [[0, 0], [0, 1], [1, 2]])
  passed += np.array_equal(values, [20., 5., 63.])
  # No temporary tensor is used.
  passed += (D._stats.get_total() == 0)
  # CHECK: Number of passed: 3
  print("Number of passed:", passed)


# CHECK-LABEL: test_binary_mul_add
@testing_utils.run_test
def test_binary_mul_add():
  i = mlir_pytaco.IndexVar()
  j = mlir_pytaco.IndexVar()
  A = mlir_pytaco.Tensor([2, 3])
  B = mlir_pytaco.Tensor([2, 3])
  C = mlir_pytaco.Tensor([2, 3])
  D = mlir_pytaco.Tensor([2, 3])
  A.insert([0, 1], 10)
  A.insert([1, 2], 40)
  B.insert([0, 0], 20)
  B.insert([1, 2], 30)
  C.insert([0, 1], 5)
  C.insert([1, 2], 7)
  D[i, j] = A[i, j] * C[i, j] + B[i, j]
  indices, values = D.get_coordinates_and_values()
  passed = np.array_equal(indices, [[0, 0], [0, 1], [1, 2]])
  passed += np.array_equal(values, [20., 50., 310.])
  # No temporary tensor is used.
  passed += (D._stats.get_total() == 0)
  # CHECK: Number of passed: 3
  print("Number of passed:", passed)


# CHECK-LABEL: test_binary_add_reduce_at_root
@testing_utils.run_test
def test_binary_add_reduce_at_root():
  i = mlir_pytaco.IndexVar()
  j = mlir_pytaco.IndexVar()
  A = mlir_pytaco.Tensor([2, 3])
  B = mlir_pytaco.Tensor([2, 3])
  C = mlir_pytaco.Tensor([2], _DENSE)
  A.insert([0, 1], 10)
  A.insert([1, 2], 40)
  B.insert([0, 0], 20)
  B.insert([1, 2], 30)
  C[i] = A[i, j] + B[i, j]
  indices, values = C.get_coordinates_and_values()
  passed = np.array_equal(indices, [[0], [1]])
  passed += np.array_equal(values, [30., 70.])
  # No temporary tensor is used.
  passed += (C._stats.get_total() == 0)
  # CHECK: Number of passed: 3
  print("Number of passed:", passed)


# CHECK-LABEL: test_binary_add_reduce_at_child
@testing_utils.run_test
def test_binary_add_reduce_at_child():
  i = mlir_pytaco.IndexVar()
  j = mlir_pytaco.IndexVar()
  I = 2
  J = 3
  A = mlir_pytaco.Tensor([I, J])
  B = mlir_pytaco.Tensor([J])
  C = mlir_pytaco.Tensor([I])
  D = mlir_pytaco.Tensor([I], _DENSE)

  _init_2d(A, I, J)
  _init_1d_with_value(C, I, 2)
  _init_1d_with_value(B, J, 1)

  D[i] = A[i, j] * B[j] + C[i]
  indices, values = D.get_coordinates_and_values()
  passed = np.array_equal(indices, [[0], [1]])
  passed += np.array_equal(values, [8., 11.])

  # The expression is implemented as:
  #    temp0[i] = A[i, j] * B[i]
  #    D[i] = temp0[i] + C[i]
  # Check the temporary tensor introduced by the implementation.
  stats = D._stats
  passed += (stats.get_total() == 1)
  passed += (stats.get_formats(0) == (_COMPRESSED,))
  passed += (stats.get_dimensions(0) == (I,))
  # CHECK: Number of passed: 5
  print("Number of passed:", passed)


# CHECK-LABEL: test_binary_add_reduce_3d_1
@testing_utils.run_test
def test_binary_add_reduce_3d_1():
  i, j, k, l = mlir_pytaco.get_index_vars(4)
  I = 2
  J = 3
  K = 4
  L = 5
  A = mlir_pytaco.Tensor([I, J, K])
  B = mlir_pytaco.Tensor([I, J, L])
  C = mlir_pytaco.Tensor([K])
  D = mlir_pytaco.Tensor([L])
  E = mlir_pytaco.Tensor([I], _DENSE)

  _init_3d(A, I, J, K)
  _init_3d(B, I, J, L)
  _init_1d_with_value(C, K, 1)
  _init_1d_with_value(D, L, 2)

  E[i] = A[i, j, k] * C[k] + B[i, j, l] * D[l]
  indices, values = E.get_coordinates_and_values()
  passed = np.array_equal(indices, [[0], [1]])
  passed += np.array_equal(values, [162., 204.])

  # The expression is implemented as:
  #    temp0[i, j] = A[i, j, k] * C[k]
  #    temp1[i, j] = B[i, j, l] * D[l]
  #    E[i] = temp0[i, j] + temp1[i, j]
  # Check the two temporary tensors introduced by the implementation.
  stats = E._stats
  passed += (stats.get_total() == 2)
  passed += (stats.get_formats(0) == (_COMPRESSED, _COMPRESSED))
  passed += (stats.get_dimensions(0) == (I, J))
  passed += (stats.get_formats(1) == (_COMPRESSED, _COMPRESSED))
  passed += (stats.get_dimensions(1) == (I, J))
  # CHECK: Number of passed: 7
  print("Number of passed:", passed)


# CHECK-LABEL: test_binary_add_reduce_3d_2
@testing_utils.run_test
def test_binary_add_reduce_3d_2():
  i, j, k, l = mlir_pytaco.get_index_vars(4)
  I = 2
  J = 3
  K = 4
  L = 5
  A = mlir_pytaco.Tensor([I, J, K], [_COMPRESSED, _COMPRESSED, _DENSE])
  B = mlir_pytaco.Tensor([I, L, K], [_DENSE, _COMPRESSED, _COMPRESSED])
  C = mlir_pytaco.Tensor([J, K], [_COMPRESSED, _COMPRESSED])
  D = mlir_pytaco.Tensor([L])
  E = mlir_pytaco.Tensor([I], _DENSE)

  _init_3d(A, I, J, K)
  _init_3d(B, I, L, K)
  _init_2d(C, J, K)
  _init_1d_with_value(D, L, 2)

  E[i] = A[i, j, k] + C[j, k] + B[i, l, k] * D[l]
  indices, values = E.get_coordinates_and_values()
  passed = np.array_equal(indices, [[0], [1]])
  passed += np.array_equal(values, [264., 316.])

  # The expression is implemented as:
  #    temp0[i, k] = A[i, j, k] + C[j, k]
  #    temp1[i, k] = B[i, l, k] * D[l]
  #    E[i] = temp0[i, k] + temp1[i, k]
  # Check the two temporary tensors introduced by the implementation.
  stats = E._stats
  passed += (stats.get_total() == 2)
  passed += (stats.get_formats(0) == (_COMPRESSED, _DENSE))
  passed += (stats.get_dimensions(0) == (I, K))
  passed += (stats.get_formats(1) == (_DENSE, _COMPRESSED))
  passed += (stats.get_dimensions(1) == (I, K))
  # CHECK: Number of passed: 7
  print("Number of passed:", passed)
