# RUN: env SUPPORTLIB=%mlir_c_runner_utils %PYTHON %s | FileCheck %s

from typing import Sequence
import dataclasses
import numpy as np
import os
import sys
import tempfile

from mlir.dialects import sparse_tensor

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import mlir_pytaco
from tools import mlir_pytaco_utils as pytaco_utils

# Define the aliases to shorten the code.
_COMPRESSED = mlir_pytaco.ModeFormat.COMPRESSED
_DENSE = mlir_pytaco.ModeFormat.DENSE


def _to_string(s: Sequence[int]) -> str:
    """Converts a sequence of integer to a space separated value string."""
    return " ".join(map(lambda e: str(e), s))


def _add_one(s: Sequence[int]) -> Sequence[int]:
    """Adds one to each element in the sequence of integer."""
    return [i + 1 for i in s]


@dataclasses.dataclass(frozen=True)
class _SparseTensorCOO:
    """Values for a COO-flavored format sparse tensor.

    Attributes:
      rank: An integer rank for the tensor.
      nse: An integer for the number of non-zero values.
      shape: A sequence of integer for the dimension size.
      values: A sequence of float for the non-zero values of the tensor.
      indices: A sequence of coordinate, each coordinate is a sequence of integer.
    """

    rank: int
    nse: int
    shape: Sequence[int]
    values: Sequence[float]
    indices: Sequence[Sequence[int]]


def _coo_values_to_tns_format(t: _SparseTensorCOO) -> str:
    """Converts a sparse tensor COO-flavored values to TNS text format."""
    # The coo_value_str contains one line for each (coordinate value) pair.
    # Indices are 1-based in TNS text format but 0-based in MLIR.
    coo_value_str = "\n".join(
        map(
            lambda i: _to_string(_add_one(t.indices[i])) + " " + str(t.values[i]),
            range(t.nse),
        )
    )

    # Returns the TNS text format representation for the tensor.
    return f"""{t.rank} {t.nse}
{_to_string(t.shape)}
{coo_value_str}
"""


def _implement_read_tns_test(
    t: _SparseTensorCOO, sparsity_codes: Sequence[sparse_tensor.DimLevelType]
) -> int:
    tns_data = _coo_values_to_tns_format(t)

    # Write sparse tensor data to a file.
    with tempfile.TemporaryDirectory() as test_dir:
        file_name = os.path.join(test_dir, "data.tns")
        with open(file_name, "w") as file:
            file.write(tns_data)

        # Read the data from the file and construct an MLIR sparse tensor.
        sparse_tensor, o_shape = pytaco_utils.create_sparse_tensor(
            file_name, sparsity_codes, "f64"
        )

    passed = 0

    # Verify the output shape for the tensor.
    if np.array_equal(o_shape, t.shape):
        passed += 1

    # Use the output MLIR sparse tensor pointer to retrieve the COO-flavored
    # values and verify the values.
    (
        o_rank,
        o_nse,
        o_shape,
        o_values,
        o_indices,
    ) = pytaco_utils.sparse_tensor_to_coo_tensor(sparse_tensor, np.float64)
    if (
        o_rank == t.rank
        and o_nse == t.nse
        and np.array_equal(o_shape, t.shape)
        and np.allclose(o_values, t.values)
        and np.array_equal(o_indices, t.indices)
    ):
        passed += 1

    return passed


# A 2D sparse tensor data in COO-flavored format.
_rank = 2
_nse = 3
_shape = [4, 5]
_values = [3.0, 2.0, 4.0]
_indices = [[0, 4], [1, 0], [3, 1]]

_t = _SparseTensorCOO(_rank, _nse, _shape, _values, _indices)
_s = [_COMPRESSED, _COMPRESSED]
# CHECK: PASSED 2D: 2
print("PASSED 2D: ", _implement_read_tns_test(_t, _s))


# A 3D sparse tensor data in COO-flavored format.
_rank = 3
_nse = 3
_shape = [2, 5, 4]
_values = [3.0, 2.0, 4.0]
_indices = [[0, 4, 3], [1, 3, 0], [1, 3, 1]]

_t = _SparseTensorCOO(_rank, _nse, _shape, _values, _indices)
_s = [_DENSE, _COMPRESSED, _COMPRESSED]
# CHECK: PASSED 3D: 2
print("PASSED 3D: ", _implement_read_tns_test(_t, _s))
