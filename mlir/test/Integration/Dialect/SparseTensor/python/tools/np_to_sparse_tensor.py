#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#  This file contains functions to process sparse tensor outputs.

import ctypes
import functools
import numpy as np


@functools.lru_cache()
def _get_c_shared_lib(lib_name: str):
  """Loads and returns the requested C shared library.

  Args:
    lib_name: A string representing the C shared library.

  Returns:
    The C shared library.

  Raises:
    OSError: If there is any problem in loading the shared library.
    ValueError:  If the shared library doesn't contain the needed routine.
  """
  # This raises OSError exception if there is any problem in loading the shared
  # library.
  c_lib = ctypes.CDLL(lib_name)

  try:
    c_lib.convertFromMLIRSparseTensorF64.restype = ctypes.c_void_p
  except Exception as e:
    raise ValueError('Missing function convertFromMLIRSparseTensorF64 from '
                     f'the C shared library: {e} ') from e

  return c_lib


def sparse_tensor_to_coo_tensor(support_lib, sparse, dtype):
  """Converts a sparse tensor to COO-flavored format.

  Args:
     support_lib: A string for the supporting C shared library.
     sparse: A ctypes.pointer to the sparse tensor descriptor.
     dtype: The numpy data type for the tensor elements.

  Returns:
    A tuple that contains the following values:
    rank: An integer for the rank of the tensor.
    nse: An integer for the number of non-zero values in the tensor.
    shape: A 1D numpy array of integers, for the shape of the tensor.
    values: A 1D numpy array, for the non-zero values in the tensor.
    indices: A 2D numpy array of integers, representing the indices for the
      non-zero values in the tensor.

  Raises:
    OSError: If there is any problem in loading the shared library.
    ValueError:  If the shared library doesn't contain the needed routine.
  """
  c_lib = _get_c_shared_lib(support_lib)

  rank = ctypes.c_ulonglong(0)
  nse = ctypes.c_ulonglong(0)
  shape = ctypes.POINTER(ctypes.c_ulonglong)()
  values = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))()
  indices = ctypes.POINTER(ctypes.c_ulonglong)()
  c_lib.convertFromMLIRSparseTensorF64(sparse, ctypes.byref(rank),
                                       ctypes.byref(nse), ctypes.byref(shape),
                                       ctypes.byref(values),
                                       ctypes.byref(indices))
  # Convert the returned values to the corresponding numpy types.
  shape = np.ctypeslib.as_array(shape, shape=[rank.value])
  values = np.ctypeslib.as_array(values, shape=[nse.value])
  indices = np.ctypeslib.as_array(indices, shape=[nse.value, rank.value])
  return rank, nse, shape, values, indices
