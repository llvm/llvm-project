#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains the utilities to process sparse tensor outputs.

from typing import Sequence, Tuple
import ctypes
import functools
import numpy as np
import os

# Import MLIR related modules.
from mlir import all_passes_registration  # Register MLIR compiler passes.
from mlir import execution_engine
from mlir import ir
from mlir import runtime
from mlir.dialects import sparse_tensor
from mlir.passmanager import PassManager

# The name for the environment variable that provides the full path for the
# supporting library.
_SUPPORTLIB_ENV_VAR = "SUPPORTLIB"
# The default supporting library if the environment variable is not provided.
_DEFAULT_SUPPORTLIB = "libmlir_c_runner_utils.so"

# The JIT compiler optimization level.
_OPT_LEVEL = 2
# The entry point to the JIT compiled program.
_ENTRY_NAME = "main"


@functools.lru_cache()
def _get_support_lib_name() -> str:
  """Gets the string name for the supporting C shared library."""
  return os.getenv(_SUPPORTLIB_ENV_VAR, _DEFAULT_SUPPORTLIB)


@functools.lru_cache()
def _get_c_shared_lib() -> ctypes.CDLL:
  """Loads the supporting C shared library with the needed routines.

  The name of the supporting C shared library is either provided by an
  an environment variable or a default value.

  Returns:
    The supporting C shared library.

  Raises:
    OSError: If there is any problem in loading the shared library.
    ValueError: If the shared library doesn't contain the needed routines.
  """
  # This raises OSError exception if there is any problem in loading the shared
  # library.
  c_lib = ctypes.CDLL(_get_support_lib_name())

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
    sparse_tensor: ctypes.c_void_p,
    dtype: np.dtype,
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
  """Converts an MLIR sparse tensor to a COO-flavored format tensor.

  Args:
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
  c_lib = _get_c_shared_lib()

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
  return rank.value, nse.value, shape, values, indices


def coo_tensor_to_sparse_tensor(np_shape: np.ndarray, np_values: np.ndarray,
                                np_indices: np.ndarray) -> int:
  """Converts a COO-flavored format sparse tensor to an MLIR sparse tensor.

  Args:
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

  c_lib = _get_c_shared_lib()
  ptr = c_lib.convertToMLIRSparseTensor(rank, nse, shape, values, indices)
  assert ptr is not None, "Problem with calling convertToMLIRSparseTensor"
  return ptr


def compile_and_build_engine(
    module: ir.Module) -> execution_engine.ExecutionEngine:
  """Compiles an MLIR module and builds a JIT execution engine.

  Args:
    module: The MLIR module.

  Returns:
    A JIT execution engine for the MLIR module.

  """
  pipeline = (
      f"sparsification,"
      f"sparse-tensor-conversion,"
      f"builtin.func(linalg-bufferize,convert-linalg-to-loops,convert-vector-to-scf),"
      f"convert-scf-to-cf,"
      f"func-bufferize,"
      f"arith-bufferize,"
      f"builtin.func(tensor-bufferize,finalizing-bufferize),"
      f"convert-vector-to-llvm{{reassociate-fp-reductions=1 enable-index-optimizations=1}},"
      f"lower-affine,"
      f"convert-memref-to-llvm,"
      f"convert-std-to-llvm,"
      f"reconcile-unrealized-casts")
  PassManager.parse(pipeline).run(module)
  return execution_engine.ExecutionEngine(
      module, opt_level=_OPT_LEVEL, shared_libs=[_get_support_lib_name()])


class _SparseTensorDescriptor(ctypes.Structure):
  """A C structure for an MLIR sparse tensor."""
  _fields_ = [
      # A pointer for the MLIR sparse tensor storage.
      ("storage", ctypes.POINTER(ctypes.c_ulonglong)),
      # An MLIR MemRef descriptor for the shape of the sparse tensor.
      ("shape", runtime.make_nd_memref_descriptor(1, ctypes.c_ulonglong)),
  ]


def _output_one_dim(dim: int, rank: int, shape: str) -> str:
  """Produces the MLIR text code to output the size for the given dimension."""
  return f"""
  %c{dim} = arith.constant {dim} : index
  %d{dim} = tensor.dim %t, %c{dim} : tensor<{shape}xf64, #enc>
  memref.store %d{dim}, %b[%c{dim}] : memref<{rank}xindex>
"""


# TODO: With better support from MLIR, we may improve the current implementation
# by doing the following:
# (1) Use Python code to generate the kernel instead of doing MLIR text code
#     stitching.
# (2) Use scf.for instead of an unrolled loop to write out the dimension sizes
#     when tensor.dim supports non-constant dimension value.
def _get_create_sparse_tensor_kernel(
    sparsity_codes: Sequence[sparse_tensor.DimLevelType]) -> str:
  """Creates an MLIR text kernel to contruct a sparse tensor from a file.

  The kernel returns a _SparseTensorDescriptor structure.
  """
  rank = len(sparsity_codes)

  # Use ? to represent a dimension in the dynamic shape string representation.
  shape = "x".join(map(lambda d: "?", range(rank)))

  # Convert the encoded sparsity values to a string representation.
  sparsity = ", ".join(
      map(lambda s: '"compressed"' if s.value else '"dense"', sparsity_codes))

  # Get the MLIR text code to write the dimension sizes to the output buffer.
  output_dims = "\n".join(
      map(lambda d: _output_one_dim(d, rank, shape), range(rank)))

  # Return the MLIR text kernel.
  return f"""
!Ptr = type !llvm.ptr<i8>
#enc = #sparse_tensor.encoding<{{
  dimLevelType = [ {sparsity} ]
}}>
func @{_ENTRY_NAME}(%filename: !Ptr) -> (tensor<{shape}xf64, #enc>, memref<{rank}xindex>)
attributes {{ llvm.emit_c_interface }} {{
  %t = sparse_tensor.new %filename : !Ptr to tensor<{shape}xf64, #enc>
  %b = memref.alloc() : memref<{rank}xindex>
  {output_dims}
  return %t, %b : tensor<{shape}xf64, #enc>, memref<{rank}xindex>
}}"""


def create_sparse_tensor(
    filename: str, sparsity: Sequence[sparse_tensor.DimLevelType]
) -> Tuple[ctypes.c_void_p, np.ndarray]:
  """Creates an MLIR sparse tensor from the input file.

  Args:
    filename: A string for the name of the file that contains the tensor data in
      a COO-flavored format.
    sparsity: A sequence of DimLevelType values, one for each dimension of the
      tensor.

  Returns:
    A Tuple containing the following values:
    storage: A ctypes.c_void_p for the MLIR sparse tensor storage.
    shape: A 1D numpy array of integers, for the shape of the tensor.

  Raises:
    OSError: If there is any problem in loading the supporting C shared library.
    ValueError:  If the shared library doesn't contain the needed routine.
  """
  with ir.Context() as ctx, ir.Location.unknown():
    module = _get_create_sparse_tensor_kernel(sparsity)
    module = ir.Module.parse(module)
    engine = compile_and_build_engine(module)

  # A sparse tensor descriptor to receive the kernel result.
  c_tensor_desc = _SparseTensorDescriptor()
  # Convert the filename to a byte stream.
  c_filename = ctypes.c_char_p(bytes(filename, "utf-8"))

  arg_pointers = [
      ctypes.byref(ctypes.pointer(c_tensor_desc)),
      ctypes.byref(c_filename)
  ]

  # Invoke the execution engine to run the module and return the result.
  engine.invoke(_ENTRY_NAME, *arg_pointers)
  shape = runtime.ranked_memref_to_numpy(ctypes.pointer(c_tensor_desc.shape))
  return c_tensor_desc.storage, shape


# TODO: With better support from MLIR, we may improve the current implementation
# by using Python code to generate the kernel instead of doing MLIR text code
# stitching.
def _get_output_sparse_tensor_kernel(
    sparsity_codes: Sequence[sparse_tensor.DimLevelType]) -> str:
  """Creates an MLIR text kernel to output a sparse tensor to a file.

  The kernel returns void.
  """
  rank = len(sparsity_codes)

  # Use ? to represent a dimension in the dynamic shape string representation.
  shape = "x".join(map(lambda d: "?", range(rank)))

  # Convert the encoded sparsity values to a string representation.
  sparsity = ", ".join(
      map(lambda s: '"compressed"' if s.value else '"dense"', sparsity_codes))

  # Return the MLIR text kernel.
  return f"""
!Ptr = type !llvm.ptr<i8>
#enc = #sparse_tensor.encoding<{{
  dimLevelType = [ {sparsity} ]
}}>
func @{_ENTRY_NAME}(%t: tensor<{shape}xf64, #enc>, %filename: !Ptr)
attributes {{ llvm.emit_c_interface }} {{
  sparse_tensor.out %t, %filename : tensor<{shape}xf64, #enc>, !Ptr
  std.return
}}"""


def output_sparse_tensor(
    tensor: ctypes.c_void_p, filename: str,
    sparsity: Sequence[sparse_tensor.DimLevelType]) -> None:
  """Outputs an MLIR sparse tensor to the given file.

  Args:
    tensor: A C pointer to the MLIR sparse tensor.
    filename: A string for the name of the file that contains the tensor data in
      a COO-flavored format.
    sparsity: A sequence of DimLevelType values, one for each dimension of the
      tensor.

  Raises:
    OSError: If there is any problem in loading the supporting C shared library.
    ValueError:  If the shared library doesn't contain the needed routine.
  """
  with ir.Context() as ctx, ir.Location.unknown():
    module = _get_output_sparse_tensor_kernel(sparsity)
    module = ir.Module.parse(module)
    engine = compile_and_build_engine(module)

  # Convert the filename to a byte stream.
  c_filename = ctypes.c_char_p(bytes(filename, "utf-8"))

  arg_pointers = [
      ctypes.byref(ctypes.cast(tensor, ctypes.c_void_p)),
      ctypes.byref(c_filename)
  ]

  # Invoke the execution engine to run the module and return the result.
  engine.invoke(_ENTRY_NAME, *arg_pointers)
