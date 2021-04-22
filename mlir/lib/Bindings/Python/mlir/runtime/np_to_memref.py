#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#  This file contains functions to convert between Memrefs and NumPy arrays and vice-versa.

import numpy as np
import ctypes


def make_nd_memref_descriptor(rank, dtype):
    class MemRefDescriptor(ctypes.Structure):
        """
        Build an empty descriptor for the given rank/dtype, where rank>0.
        """

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
        """
        Build an empty descriptor for the given dtype, where rank=0.
        """

        _fields_ = [
            ("allocated", ctypes.c_longlong),
            ("aligned", ctypes.POINTER(dtype)),
            ("offset", ctypes.c_longlong),
        ]

    return MemRefDescriptor


class UnrankedMemRefDescriptor(ctypes.Structure):
    """ Creates a ctype struct for memref descriptor"""

    _fields_ = [("rank", ctypes.c_longlong), ("descriptor", ctypes.c_void_p)]


def get_ranked_memref_descriptor(nparray):
    """
    Return a ranked memref descriptor for the given numpy array.
    """
    if nparray.ndim == 0:
        x = make_zero_d_memref_descriptor(np.ctypeslib.as_ctypes_type(nparray.dtype))()
        x.allocated = nparray.ctypes.data
        x.aligned = nparray.ctypes.data_as(
            ctypes.POINTER(np.ctypeslib.as_ctypes_type(nparray.dtype))
        )
        x.offset = ctypes.c_longlong(0)
        return x

    x = make_nd_memref_descriptor(
        nparray.ndim, np.ctypeslib.as_ctypes_type(nparray.dtype)
    )()
    x.allocated = nparray.ctypes.data
    x.aligned = nparray.ctypes.data_as(
        ctypes.POINTER(np.ctypeslib.as_ctypes_type(nparray.dtype))
    )
    x.offset = ctypes.c_longlong(0)
    x.shape = nparray.ctypes.shape

    # Numpy uses byte quantities to express strides, MLIR OTOH uses the
    # torch abstraction which specifies strides in terms of elements.
    strides_ctype_t = ctypes.c_longlong * nparray.ndim
    x.strides = strides_ctype_t(*[x // nparray.itemsize for x in nparray.strides])
    return x


def get_unranked_memref_descriptor(nparray):
    """
    Return a generic/unranked memref descriptor for the given numpy array.
    """
    d = UnrankedMemRefDescriptor()
    d.rank = nparray.ndim
    x = get_ranked_memref_descriptor(nparray)
    d.descriptor = ctypes.cast(ctypes.pointer(x), ctypes.c_void_p)
    return d


def unranked_memref_to_numpy(unranked_memref, np_dtype):
    """
    Converts unranked memrefs to numpy arrays.
    """
    descriptor = make_nd_memref_descriptor(
        unranked_memref[0].rank, np.ctypeslib.as_ctypes_type(np_dtype)
    )
    val = ctypes.cast(unranked_memref[0].descriptor, ctypes.POINTER(descriptor))
    np_arr = np.ctypeslib.as_array(val[0].aligned, shape=val[0].shape)
    strided_arr = np.lib.stride_tricks.as_strided(
        np_arr,
        np.ctypeslib.as_array(val[0].shape),
        np.ctypeslib.as_array(val[0].strides) * np_arr.itemsize,
    )
    return strided_arr


def ranked_memref_to_numpy(ranked_memref):
    """
    Converts ranked memrefs to numpy arrays.
    """
    np_arr = np.ctypeslib.as_array(
        ranked_memref[0].aligned, shape=ranked_memref[0].shape
    )
    strided_arr = np.lib.stride_tricks.as_strided(
        np_arr,
        np.ctypeslib.as_array(ranked_memref[0].shape),
        np.ctypeslib.as_array(ranked_memref[0].strides) * np_arr.itemsize,
    )
    return strided_arr
