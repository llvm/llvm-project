#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains the utilities to process sparse tensor outputs.

from typing import Tuple
import ctypes
import functools
import numpy as np


@functools.lru_cache()
def _get_c_shared_lib(lib_name: str) -> ctypes.CDLL:
  """Loads and returns the requested C shared library.

  Args:
    lib_name: A string representing the C shared library.

  Returns:
    The C shared library.

  Raises:
    OSError: If there is any problem in loading the shared library.
    ValueError: If the shared library doesn't contain the needed routines.
  """
  # This raises OSError exception if there is any problem in loading the shared
  # library.
  c_lib = ctypes.CDLL(lib_name)

  try:
    c_lib.convertToMLIRSparseTensor.restype = ctypes.c_void_p
  except Exception as e:
    raise ValueError("Missing function convertToMLIRSparseTensor from "
                     f"the supporting C shared library: {e} ") from e

  try:
    c_lib.convertFromMLIRSparseTensor.restype = ctypes.c_void_p
  except Exception as e:
    raise ValueError("Missing function convertFromMLIRSparseTensor from "
                     f"the C shared library: {e} ") from e

  return c_lib


def sparse_tensor_to_coo_tensor(
    lib_name: str,
    sparse_tensor: ctypes.c_void_p,
    dtype: np.dtype,
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
  """Converts an MLIR sparse tensor to a COO-flavored format tensor.

  Args:
     lib_name: A string for the supporting C shared library.
     sparse_tensor: A ctypes.c_void_p to the MLIR sparse tensor descriptor.
     dtype: The numpy data type for the tensor elements.

  Returns:
    A tuple that contains the following values for the COO-flavored format
    tensor:
    rank: An integer for the rank of the tensor.
    nse: An interger for the number of non-zero values in the tensor.
    shape: A 1D numpy array of integers, for the shape of the tensor.
    values: A 1D numpy array, for the non-zero values in the tensor.
    indices: A 2D numpy array of integers, representing the indices for the
      non-zero values in the tensor.

  Raises:
    OSError: If there is any problem in loading the shared library.
    ValueError: If the shared library doesn't contain the needed routines.
  """
  c_lib = _get_c_shared_lib(lib_name)

  rank = ctypes.c_ulonglong(0)
  nse = ctypes.c_ulonglong(0)
  shape = ctypes.POINTER(ctypes.c_ulonglong)()
  values = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))()
  indices = ctypes.POINTER(ctypes.c_ulonglong)()
  c_lib.convertFromMLIRSparseTensor(sparse_tensor, ctypes.byref(rank),
                                    ctypes.byref(nse), ctypes.byref(shape),
                                    ctypes.byref(values), ctypes.byref(indices))

  # Convert the returned values to the corresponding numpy types.
  shape = np.ctypeslib.as_array(shape, shape=[rank.value])
  values = np.ctypeslib.as_array(values, shape=[nse.value])
  indices = np.ctypeslib.as_array(indices, shape=[nse.value, rank.value])
  return rank, nse, shape, values, indices


def coo_tensor_to_sparse_tensor(lib_name: str, np_shape: np.ndarray,
                                np_values: np.ndarray,
                                np_indices: np.ndarray) -> int:
  """Converts a COO-flavored format sparse tensor to an MLIR sparse tensor.

  Args:
     lib_name: A string for the supporting C shared library.
     np_shape: A 1D numpy array of integers, for the shape of the tensor.
     np_values: A 1D numpy array, for the non-zero values in the tensor.
     np_indices: A 2D numpy array of integers, representing the indices for the
       non-zero values in the tensor.

  Returns:
     An integer for the non-null ctypes.c_void_p to the MLIR sparse tensor
     descriptor.

  Raises:
    OSError: If there is any problem in loading the shared library.
    ValueError: If the shared library doesn't contain the needed routines.
  """

  rank = ctypes.c_ulonglong(len(np_shape))
  nse = ctypes.c_ulonglong(len(np_values))
  shape = np_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
  values = np_values.ctypes.data_as(
      ctypes.POINTER(np.ctypeslib.as_ctypes_type(np_values.dtype)))
  indices = np_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))

  c_lib = _get_c_shared_lib(lib_name)
  ptr = c_lib.convertToMLIRSparseTensor(rank, nse, shape, values, indices)
  assert ptr is not None, "Problem with calling convertToMLIRSparseTensor"
  return ptr
