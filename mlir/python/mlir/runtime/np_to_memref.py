#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#  This file contains functions to convert between Memrefs and NumPy arrays and vice-versa.

import numpy as np
import ctypes


class C128(ctypes.Structure):
  """A ctype representation for MLIR's Double Complex."""
  _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


class C64(ctypes.Structure):
  """A ctype representation for MLIR's Float Complex."""
  _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class F16(ctypes.Structure):
  """A ctype representation for MLIR's Float16."""
  _fields_ = [("f16", ctypes.c_int16)]


def as_ctype(dtp):
  """Converts dtype to ctype."""
  if dtp is np.dtype(np.complex128):
    return C128
  if dtp is np.dtype(np.complex64):
    return C64
  if dtp is np.dtype(np.float16):
    return F16
  return np.ctypeslib.as_ctypes_type(dtp)


def to_numpy(array):
  """Converts ctypes array back to numpy dtype array."""
  if array.dtype == C128:
    return array.view("complex128")
  if array.dtype == C64:
    return array.view("complex64")
  if array.dtype == F16:
    return array.view("float16")
  return array


def make_nd_memref_descriptor(rank, dtype):

  class MemRefDescriptor(ctypes.Structure):
    """Builds an empty descriptor for the given rank/dtype, where rank>0."""

    _fields_ = [
        ("allocated", ctypes.c_longlong),
        ("aligned", ctypes.POINTER(dtype)),
        ("offset", ctypes.c_longlong),
        ("shape", ctypes.c_longlong * rank),
        ("strides", ctypes.c_longlong * rank),
    ]

  return MemRefDescriptor


def make_zero_d_memref_descriptor(dtype):

  class MemRefDescriptor(ctypes.Structure):
    """Builds an empty descriptor for the given dtype, where rank=0."""

    _fields_ = [
        ("allocated", ctypes.c_longlong),
        ("aligned", ctypes.POINTER(dtype)),
        ("offset", ctypes.c_longlong),
    ]

  return MemRefDescriptor


class UnrankedMemRefDescriptor(ctypes.Structure):
  """Creates a ctype struct for memref descriptor"""
  _fields_ = [("rank", ctypes.c_longlong), ("descriptor", ctypes.c_void_p)]


def get_ranked_memref_descriptor(nparray):
  """Returns a ranked memref descriptor for the given numpy array."""
  ctp = as_ctype(nparray.dtype)
  if nparray.ndim == 0:
    x = make_zero_d_memref_descriptor(ctp)()
    x.allocated = nparray.ctypes.data
    x.aligned = nparray.ctypes.data_as(ctypes.POINTER(ctp))
    x.offset = ctypes.c_longlong(0)
    return x

  x = make_nd_memref_descriptor(nparray.ndim, ctp)()
  x.allocated = nparray.ctypes.data
  x.aligned = nparray.ctypes.data_as(ctypes.POINTER(ctp))
  x.offset = ctypes.c_longlong(0)
  x.shape = nparray.ctypes.shape

  # Numpy uses byte quantities to express strides, MLIR OTOH uses the
  # torch abstraction which specifies strides in terms of elements.
  strides_ctype_t = ctypes.c_longlong * nparray.ndim
  x.strides = strides_ctype_t(*[x // nparray.itemsize for x in nparray.strides])
  return x


def get_unranked_memref_descriptor(nparray):
  """Returns a generic/unranked memref descriptor for the given numpy array."""
  d = UnrankedMemRefDescriptor()
  d.rank = nparray.ndim
  x = get_ranked_memref_descriptor(nparray)
  d.descriptor = ctypes.cast(ctypes.pointer(x), ctypes.c_void_p)
  return d


def unranked_memref_to_numpy(unranked_memref, np_dtype):
  """Converts unranked memrefs to numpy arrays."""
  ctp = as_ctype(np_dtype)
  descriptor = make_nd_memref_descriptor(unranked_memref[0].rank, ctp)
  val = ctypes.cast(unranked_memref[0].descriptor, ctypes.POINTER(descriptor))
  np_arr = np.ctypeslib.as_array(val[0].aligned, shape=val[0].shape)
  strided_arr = np.lib.stride_tricks.as_strided(
      np_arr,
      np.ctypeslib.as_array(val[0].shape),
      np.ctypeslib.as_array(val[0].strides) * np_arr.itemsize,
  )
  return to_numpy(strided_arr)


def ranked_memref_to_numpy(ranked_memref):
  """Converts ranked memrefs to numpy arrays."""
  np_arr = np.ctypeslib.as_array(
      ranked_memref[0].aligned, shape=ranked_memref[0].shape)
  strided_arr = np.lib.stride_tricks.as_strided(
      np_arr,
      np.ctypeslib.as_array(ranked_memref[0].shape),
      np.ctypeslib.as_array(ranked_memref[0].strides) * np_arr.itemsize,
  )
  return to_numpy(strided_arr)
