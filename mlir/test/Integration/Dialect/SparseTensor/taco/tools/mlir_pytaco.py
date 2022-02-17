#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Experimental MLIR-PyTACO with sparse tensor support.

See http://tensor-compiler.org/ for TACO tensor compiler.

This module implements the Python classes for PyTACO index notation. These
include classes for data types, tensor dimension formats (aka mode formats),
tensor dimension orderings (aka mode ordering), tensor storage formats, and
tensors.

The PyTACO API doesn't follow the naming conversion required by the style guide
for this module. As such, we first implement the supporting classes and routines
following the style guide, and then define the type aliases and constants to
support the PyTACO API in the pytaco_api module.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import abc
import ctypes
import dataclasses
import enum
import numpy as np
import functools
import operator
import os
import threading

# Import MLIR related modules.
from mlir import ir
from mlir import runtime
from mlir.dialects import arith
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std
from mlir.dialects import sparse_tensor
from mlir.dialects.linalg.opdsl import lang

from . import mlir_pytaco_utils as utils

# TACO naming prefixes.
_TACO_INDEX_PREFIX = "i"
_TACO_TENSOR_PREFIX = "A"

# Bitwidths for pointers and indices.
_POINTER_BIT_WIDTH = 0
_INDEX_BIT_WIDTH = 0
# The entry point to the JIT compiled program.
_ENTRY_NAME = "main"

# Type aliases for type annotation.
_BinaryOp = Callable[[Any, Any], Any]
_ExprVisitor = Callable[..., None]
_ExprInfoDict = Dict["IndexExpr", "_ExprInfo"]
_LogicalOp = Callable[[bool, bool], bool]
_ModeFormatOp = Callable[["ModeFormat", "ModeFormat"], "ModeFormat"]
_SubtreeLeafChecker = Optional[Callable[..., bool]]


class Type(enum.Enum):
  """The data types supported by TACO.

  We use numpy data types to implement the enum data types.
  """
  INT16 = np.int16
  INT32 = np.int32
  INT64 = np.int64
  # numpy _ctype_from_dtype_scalar can't handle np.float16 yet.
  FLOAT32 = np.float32
  FLOAT64 = np.float64


# All floating point type enums.
_FLOAT_TYPES = (Type.FLOAT32, Type.FLOAT64)
# All integral type enums.
_INT_TYPES = (Type.INT16, Type.INT32, Type.INT64)
# Type alias for any numpy type used to implement the runtime support for the
# enum data types.
_AnyRuntimeType = Union[np.int16, np.int32, np.int64, np.float32, np.float64]


@dataclasses.dataclass(frozen=True)
class DType:
  """The data type class.

  We support the TACO API dtype class with an alias of this class.

  The following methods are defined by the TACO API:
    is_float: Returns whether the data type represents a floating point value.
    is_int:   Returns whether the data type represents an integral value.

  Attributes:
    kind: A Type enum representing the data type.
    value: The numpy data type for the TACO data type.
  """
  kind: Type = Type.FLOAT64

  def is_float(self) -> bool:
    """Returns whether the data type represents a floating point value."""
    return self.kind in _FLOAT_TYPES

  def is_int(self) -> bool:
    """Returns whether the data type represents an integral value."""
    return self.kind in _INT_TYPES

  @property
  def value(self) -> _AnyRuntimeType:
    """Returns the numpy dtype for the data type."""
    return self.kind.value


def _mlir_type_from_taco_type(dtype: DType) -> ir.Type:
  """Returns the MLIR type corresponding to the given TACO type."""
  dtype_to_irtype = {
      Type.INT16: ir.IntegerType.get_signless(16),
      Type.INT32: ir.IntegerType.get_signless(32),
      Type.INT64: ir.IntegerType.get_signless(64),
      Type.FLOAT32: ir.F32Type.get(),
      Type.FLOAT64: ir.F64Type.get()
  }
  return dtype_to_irtype[dtype.kind]


def _ctype_pointer_from_array(array: np.ndarray) -> ctypes.pointer:
  """Returns the ctype pointer for the given numpy array."""
  return ctypes.pointer(
      ctypes.pointer(runtime.get_ranked_memref_descriptor(array)))


class ModeFormat(enum.Enum):
  """The tensor dimension storage format class.

  We support the TACO API mode_format class with an alias of this class.

  In TACO, a tensor dimension is called a mode and the storage format for a
  tensor dimension is called a mode format.
  """
  DENSE = sparse_tensor.DimLevelType.dense
  COMPRESSED = sparse_tensor.DimLevelType.compressed


def _mode_format_operation(a: ModeFormat, b: ModeFormat,
                           op: _LogicalOp) -> ModeFormat:
  """Implements the given operator on ModeFormat."""
  return (ModeFormat.COMPRESSED
          if op(a == ModeFormat.COMPRESSED, b == ModeFormat.COMPRESSED) else
          ModeFormat.DENSE)


def _mode_format_estimator(op: _BinaryOp) -> _ModeFormatOp:
  """Produces a ModeFormat operator for the given binary operator.

  The ModeFormat operator is used as a heuristic to derive the destination
  dimension sparsity from the source dimension sparsity. In particular, if the
  binary operator produces a disjunction of the zero values from its source
  operands, such as the MUL operator, we return a ModeFormat operator that
  uses operator.or_. That is, we estimate that a dimension for the MUL
  operation result to be sparse if either of its source operands is sparse.

  On the other hand, if the binary operator produces a conjunction of the
  zero values from its source operands, such as the ADD operator, we return
  a ModeFormat operator that uses operator.and_. In this case, we estimate
  that a dimension for the ADD operation result to be sparse if both of its
  source operands are sparse.

  Args:
    op: A _BinaryOp object representing a supporting operator on tensors.

  Returns:
    A ModeFormatOp for estimating the destination dimension sparsity from
    the source dimension sparsity.
  """
  conjunction = functools.partial(_mode_format_operation, op=operator.and_)
  disjunction = functools.partial(_mode_format_operation, op=operator.or_)
  return conjunction if op(0, 1) != 0 else disjunction


def _all_instance_of(collection: Iterable, cls: Any) -> bool:
  """Returns true if all elements of the iterable is an instance of cls."""
  return all(isinstance(e, cls) for e in collection)


def _identity_ordering(rank: int) -> List[int]:
  """Returns the identity ordering for tensor of given rank."""
  return list(range(rank))


@dataclasses.dataclass(frozen=True)
class ModeOrdering:
  """The tensor dimension ordering class.

  We support the TACO API mode_ordering class with an alias of this class.

  Attributes:
    ordering: A list of integers representing the ordering of the tensor
      dimensions.
  """
  ordering: List[int]

  def __post_init__(self) -> None:
    """Verifies the value in ordering.

    Raises:
       ValueError: If ordering is not a list of integers.
    """
    if (not isinstance(self.ordering, list) or
        not _all_instance_of(self.ordering, int)):
      raise ValueError("Ordering must be a list of integers: "
                       f"{self.ordering}")
    # Check that ordering is a permutation of the dimension numbers.
    if sorted(self.ordering) != _identity_ordering(self.rank()):
      raise ValueError(f"Invalid ordering: {self.ordering} != "
                       f"permutation{_identity_ordering(self.rank())}.")

  def rank(self) -> int:
    """Returns the number of dimensions represented by the ordering."""
    return len(self.ordering)


@dataclasses.dataclass(frozen=True)
class ModeFormatPack:
  """The tensor dimension format class.

  We support the TACO API mode_format_pack class with an alias of this class.

  The storage format of a tensor contains one mode_format for each tensor
  dimension.

  Attributes:
    formats: A list of ModeFormat representing the storage format for each of
      the tensor dimension.
  """
  formats: List[ModeFormat]

  def __post_init__(self) -> None:
    """Verifies the value in formats.

    Raises:
       ValueError: If formats is not a list of ModeFormats.
    """
    if (not isinstance(self.formats, list) or
        not _all_instance_of(self.formats, ModeFormat)):
      raise ValueError("Formats must be a list of ModeFormat: "
                       f"{self.formats}")

  def rank(self) -> int:
    """Returns the number of dimensions represented by the format pack."""
    return len(self.formats)


@dataclasses.dataclass
class Format:
  """The tensor format class defined by the TACO API.

  Attributes:
    format_pack: A ModeFormatPack representing the storage format for the tensor
      dimensions.
    ordering: A ModeOrdering representing the tensor dimension ordering in the
      storage.
  """
  format_pack: ModeFormatPack
  ordering: Optional[ModeOrdering] = None

  def __post_init__(self) -> None:
    """Verifies and fixes up the values in format_pack and ordering.

    Verifies and fixes up the values in format_pack and ordering to supports the
    initializer syntax defined by the TACO API. If format_pack is a list of
    ModeFormat, replaces it with ModeFormatPack constructed from the list. If
    ordering is not provided, set ordering to the natural ordering for the rank
    corresponding to format_pack.

    Raises:
       ValueError: If format_pack is not an instance of ModeFormatPack or if
         ordering is not an instance of ModeOrdering.
    """
    if isinstance(self.format_pack, list):
      if not _all_instance_of(self.format_pack, ModeFormat):
        raise ValueError(f"Expected a list of ModeFormat: {self.format_pack}")
      self.format_pack = ModeFormatPack(self.format_pack)
    if not isinstance(self.format_pack, ModeFormatPack):
      raise ValueError(f"Expected ModeFormatpack: {self.format_pack}")

    if self.ordering is None:
      self.ordering = ModeOrdering(list(range(self.rank())))
    if isinstance(self.ordering, list):
      if not _all_instance_of(self.ordering, int):
        raise ValueError(f"Expected a list of integer: {self.ordering}")
      self.ordering = ModeOrdering(self.ordering)
    if not isinstance(self.ordering, ModeOrdering):
      raise ValueError(f"Expected ModeOrdering: {self.ordering}")

    if self.format_pack.rank() != self.ordering.rank():
      raise ValueError("Inconsistent ModeFormatPack and ModeOrdering: "
                       f"len({self.format_pack}) != "
                       f"len({self.ordering})")

  def rank(self) -> int:
    """Returns the number of dimensions represented by the format."""
    return self.format_pack.rank()

  def mlir_tensor_attr(self) -> Optional[sparse_tensor.EncodingAttr]:
    """Constructs the MLIR attributes for the tensor format."""
    order = (
        range(self.rank()) if
        (self.ordering is None) else self.ordering.ordering)
    mlir_storage_format = [f.value for f in self.format_pack.formats]
    return sparse_tensor.EncodingAttr.get(mlir_storage_format,
                                          ir.AffineMap.get_permutation(order),
                                          _POINTER_BIT_WIDTH, _INDEX_BIT_WIDTH)


def _make_format(formats: List[ModeFormat],
                 ordering: Optional[List[int]] = None) -> Format:
  """Constructs a format from a list of ModeFormat and an optional ordering.

  Args:
    formats: A list of ModeFormat, one for each dimension of a tensor.
    ordering: An optional list of integer, for the ordering of the tensor
      dimensions. When an ordering is not given, the identity ordering is used.

  Returns:
    A tensor format object.

  Raises:
    ValueError: If formats is not a list of ModeFormat or the length of formats
      is not consistent with the len of ordering.
  """
  ordering = ordering or _identity_ordering(len(formats))
  return Format(ModeFormatPack(formats), ModeOrdering(ordering))


class _AtomicCounter:
  """An atomic counter."""

  def __init__(self):
    self._counter = 0
    self._counter_lock = threading.Lock()

  def increment(self) -> int:
    """Increments the counter by one and returns the old value."""
    old_value = self._counter
    with self._counter_lock:
      self._counter = self._counter + 1
    return old_value


class IndexVar:
  """The tensor index class.

  We support the TACO API index_var class with an alias of this class.

  An IndexVar object represents an index variable in tensor index notation.

  Attributes:
    name: A unique string name of the IndexVar.
  """
  _counter = _AtomicCounter()

  def __init__(self):
    id = self._counter.increment()
    self._name = f"{_TACO_INDEX_PREFIX}{id}"

  def __repr__(self) -> str:
    return f"IndexVar(name={repr(self._name)})"

  @property
  def name(self) -> str:
    """Returns the name of the IndexVar."""
    return self._name


def get_index_vars(n: int) -> List[IndexVar]:
  """Returns a list of n IndexVar.

  This routine is defined by the TACO API.

  Args:
    n: An integer representing the number of IndexVar to get.

  Returns:
    A list of IndexVar.

  Raises:
    ValueError: if n is not a positive integer.
  """
  if not isinstance(n, int) or n <= 0:
    raise ValueError(f"Expected an integer: {n}.")
  # If lock contention ever becomes an issue, we could implement a bulk getter
  # that returns a range by only claiming the lock once.
  return [IndexVar() for i in range(n)]


def _mlir_symbols_from_index_vars(
    index_vars: Tuple[IndexVar, ...]) -> Tuple[lang.SymbolDef, ...]:
  """Returns a tuple of MLIR symbols for the given tuple of index_var."""
  return tuple(getattr(lang.S, i.name) for i in index_vars)


def _mlir_dimensions_from_index_vars(
    index_vars: Tuple[IndexVar, ...]) -> Tuple[lang.DimDef, ...]:
  """Returns a tuple of MLIR dimensions for the given tuple of index_var."""
  return tuple(getattr(lang.D, i.name) for i in index_vars)


def _mlir_tensor_type(
    dtype: DType, shape: Tuple[int, ...],
    attr: Optional[sparse_tensor.EncodingAttr]) -> ir.RankedTensorType:
  """Returns an MLIR tensor type.

  Args:
    dtype: An DType object for the element data type of the tensor.
    shape: A tuple of integer for the shape of the tensor.
    attr: An optional MLIR sparse tensor attribute, only provided if the tensor
      is a sparse tensor.

  Returns:
    An MLIR ranked tensor type.
  """
  ir_type = _mlir_type_from_taco_type(dtype)
  return ir.RankedTensorType.get(shape, ir_type, attr)


def _verify_and_normalize_indices(indices) -> Tuple[IndexVar, ...]:
  """Verifies and normalizes the indices for a tensor access.

  Args:
    indices: The index expression used to access a tensor, which could be any
      Python object from user inputs.

  Returns:
    A tuple of IndexVar.

  Raises:
    ValueError: If indices is not an IndexVar or a tuple of IndexVar.
  """
  if isinstance(indices, IndexVar):
    return (indices,)
  elif isinstance(indices, tuple) and _all_instance_of(indices, IndexVar):
    return indices

  raise ValueError(f"Expected IndexVars: {indices}")


@dataclasses.dataclass(frozen=True)
class _StructOpInfo:
  """Information for generating a structured op in the linalg dialect.

  This information is associated with an expression node that serves as the
  root for an expression subtree implemented with a structured op.

  Attributes:
    dst_indices: A tuple of IndexVar, representing the result dimensions of the
      structured op. This is used to construct the temporary variable for the
      tensor to hold the structured op result.
    dst_dims: A tuple of int, representing the result shape of the structured
      op.
    dst_dtype: A DType representing the data type of the structured op result.
    dst_name: A string representing the name of the structured op result.
    dst_format: An optional Format object representing the destination tensor
      format. None represents a true dense tensor.
  """
  dst_indices: Tuple[IndexVar, ...]
  dst_dims: Tuple[int, ...]
  dst_dtype: DType
  dst_name: str
  dst_format: Optional[Format]

  def __post_init__(self) -> None:
    """Verifies the integrity of the attribute values."""
    assert len(self.dst_indices) == len(self.dst_dims)

  def emit_tensor_init(self) -> ir.RankedTensorType:
    """Returns an initialization for the destination tensor."""
    if self.dst_format is None:
      # Initialize the dense tensor.
      ir_type = _mlir_type_from_taco_type(self.dst_dtype)
      tensor = linalg.InitTensorOp(self.dst_dims, ir_type).result
      zero = arith.ConstantOp(ir_type, 0.0)
      return linalg.FillOp(output=tensor, value=zero).results[0]

    # Initialize the sparse tensor.
    mlir_type = _mlir_tensor_type(self.dst_dtype, self.dst_dims,
                                  self.dst_format.mlir_tensor_attr())
    index_type = ir.IndexType.get()
    dims = [arith.ConstantOp(index_type, d).result for d in mlir_type.shape]
    return sparse_tensor.InitOp(mlir_type, dims)


class _Stats:
  """Information to describe how a tensor expression is implemented.

  Currently, we only record the temporary tensors introduced for splitting the
  original expression.
  """

  def __init__(self):
    self._temps = []

  def __repr__(self) -> str:
    return f"_Stats({repr(self._temps)})"

  def add_element(self, structop: _StructOpInfo):
    """Adds a temporary tensor."""
    self._temps.append(structop)

  def get_total(self) -> int:
    """Gets the total number of temporary tensors."""
    return len(self._temps)

  def _get_element(self, idx: int) -> _StructOpInfo:
    """Gets the ith temporary tensor."""
    assert idx < self.get_total()
    return self._temps[idx]

  def get_dimensions(self, idx: int) -> Tuple[int]:
    """Gets the dimensions for the ith temporary tensor."""
    return self._get_element(idx).dst_dims

  def get_formats(self, idx: int) -> Tuple[ModeFormat]:
    """Gets the ModeFormats for the ith temporary tensor."""
    return tuple(self._get_element(idx).dst_format.format_pack.formats)


class _SparseValueInfo(enum.Enum):
  """Describes how a sparse tensor value is stored.
  _UNPACKED: The sparse tensor value is stored as (coordnates, values) in
    Python.
  _PACKED: The sparse tensor value is stored as a C pointer to a packed MLIR
    sparse tensor.
  """
  _UNPACKED = 0
  _PACKED = 1


@dataclasses.dataclass(frozen=True)
class _Assignment:
  """Records an assignment to a tensor T as T[indices] = expression."""
  indices: Tuple["IndexVar", ...]
  expression: "IndexExpr"


class Tensor:
  """The tensor class.

  We support the TACO API tensor class with an alias of this class.

  This class is part of the TACO API with the following methods:
    insert: Inserts a value to the given coordinate in the tensor.
    to_array: Returns a numpy ndarray for the tensor.

  TACO API also defines the following arrtibutes for the class:
    dtype: A dtype object representing the data type of the tensor.
    format: A format object representing the storage format of the tensor.
    name: A string object representing the name of the tensor.
    order: An integral rank of the tensor.
    shape: A list of integers representing the shape of the tensor.

  We currently ignore the tensor dimension ordering for dense tensor.
  """
  _counter = _AtomicCounter()

  def _get_unique_name(self) -> str:
    """Returns a unique name for creating a new Tensor."""
    return f"{_TACO_TENSOR_PREFIX}{self._counter.increment()}"

  def _init_format(self, fmt: Union[ModeFormat, List[ModeFormat],
                                    Format]) -> None:
    """Process the fmt argument for the Tensor constructor.

    Args:
      fmt: This argument can be a ModeFormat, List[ModeFormat], or format. If
        this argument is a ModeFormat, uses this ModeFormat for all the tensor
        dimensions. If this argument is a list of ModeFormat, the len of the
        list should equal to the rank of the tensor. If this argument is a
        format, uses it for the format of the tensor.

    Raises:
      ValueError: If fmt is not one of the expected type or is inconsistent
        with the rank of the tensor. This is because fmt could be an users
        input.
    """
    if isinstance(fmt, ModeFormat):
      self._format = _make_format([fmt] * self.order)
    elif isinstance(fmt, list):
      if len(fmt) == self.order and isinstance(fmt[0], ModeFormat):
        self._format = _make_format(fmt)
      else:
        raise ValueError("Inconsistent shape and format: "
                         f"{self._shape}, {fmt}.")
    elif isinstance(fmt, Format):
      if fmt.rank() != self.order:
        raise ValueError("Inconsistent shape and format: "
                         f"{self._shape}, {fmt}.")
      else:
        self._format = fmt
    else:
      raise ValueError(f"Invalid format argument: {fmt}.")

  def __init__(self,
               value_or_shape: Optional[Union[List[int], Tuple[int, ...], float,
                                              int]] = None,
               fmt: Optional[Union[ModeFormat, List[ModeFormat],
                                   Format]] = None,
               dtype: Optional[DType] = None,
               name: Optional[str] = None,
               is_dense: bool = False):
    """The tensor constructor interface defined by TACO API.

    Args:
      value_or_shape: This argument is optional and can be int, float,
        List[int], or Tuple[int, ...]. If this argument is an int or float,
        creates a scalar tensor and initializes it with the value. If this
        argument is a list or tuple of int, uses it as the shape to create a
        tensor.
      fmt: This argument can be a ModeFormat, List[ModeFormat], or format. If
        this argument is a ModeFormat, uses this ModeFormat for all the tensor
        dimensions. If this argument is a list of ModeFormat, the len of the
        list should equal to the rank of the tensor. If this argument is a
        format, uses it for the format of the tensor.
      dtype: An object of dtype, representing the data type of the tensor.
      name: A string name of the tensor. If a name is not given, creates a
        unique name for the tensor.
      is_dense: A boolean variable to indicate whether the tensor is a dense
        tensor without any sparsity annotation.

    Raises:
      ValueError: If there is any inconsistency among the input arguments.
    """
    # Take care of the argument default values common to both sparse tensors
    # and dense tensors.
    dtype = dtype or DType(Type.FLOAT64)
    self._name = name or self._get_unique_name()
    self._assignment = None
    self._sparse_value_location = _SparseValueInfo._UNPACKED
    self._dense_storage = None
    self._dtype = dtype

    if is_dense:
      assert (fmt is None)
      assert (isinstance(value_or_shape, tuple) or isinstance(
          value_or_shape, list)) and _all_instance_of(value_or_shape, int)
      self._shape = value_or_shape
      self._format = None
      return

    fmt = fmt or ModeFormat.COMPRESSED
    # We currently use _coords and _values to host the sparse tensor value with
    # COO format, and _dense_storage to host the dense tensor value. We don't
    # support the conversion between the two storages.
    self._coords = []
    self._values = []
    self._stats = _Stats()
    if value_or_shape is None or isinstance(value_or_shape, int) or isinstance(
        value_or_shape, float):
      # Create a scalar tensor and ignore the fmt parameter.
      self._shape = []
      self._format = _make_format([], [])
      if value_or_shape is not None:
        self._dense_storage = np.array(value_or_shape, dtype=self._dtype.value)
    elif (isinstance(value_or_shape, tuple) or isinstance(
        value_or_shape, list)) and _all_instance_of(value_or_shape, int):
      # Create a tensor with the specified shape and format.
      self._shape = list(value_or_shape)
      self._init_format(fmt)
    else:
      raise ValueError("Invalid first argument. "
                       "Must be a tuple or list for a shape or a single value"
                       f"if initializing a scalar tensor: {value_or_shape}.")

  def _set_packed_sparse_tensor(self, pointer: ctypes.c_void_p) -> None:
    """Records the MLIR sparse tensor pointer."""
    self._sparse_value_location = _SparseValueInfo._PACKED
    self._packed_sparse_value = pointer

  def is_unpacked(self) -> bool:
    """Returns true if the tensor value is not packed as MLIR sparse tensor."""
    return (self._sparse_value_location == _SparseValueInfo._UNPACKED)

  def unpack(self) -> None:
    """Unpacks the MLIR sparse tensor representation."""
    if self.is_dense() or self.is_unpacked():
      return

    # Use the output MLIR sparse tensor pointer to retrieve the COO-flavored
    # values and verify the values.
    rank, nse, shape, values, indices = utils.sparse_tensor_to_coo_tensor(
        self._packed_sparse_value, np.float64)
    assert rank == self.order
    assert np.allclose(self.shape, shape)
    assert nse == len(values)
    self._coords = indices
    self._values = values
    self._sparse_value_location = _SparseValueInfo._UNPACKED

  def __repr__(self) -> str:
    self._sync_value()
    self.unpack()
    value_str = (f"{repr(self._dense_storage)})" if self.is_dense() else
                 f"{repr(self._coords)} {repr(self._values)})")
    return (f"Tensor(_name={repr(self._name)} "
            f"_dtype={repr(self._dtype)} : ") + value_str

  def insert(self, coords: List[int], val: Union[float, int]) -> None:
    """Inserts a value to the given coordinate.

    Args:
      coords: A list of integer coordinates. The length of the list must be the
        same as the rank of the tensor.
      val: A value being inserted. It is either an integral or a floating point
        value. This value will be converted to the data type of the tensor.

    Raises:
      ValueError: When there is any problem in the parameters.
    """
    if self.is_dense():
      raise ValueError("Insert method is not supported for dense tensors.")
    if self._assignment != None or not self.is_unpacked():
      raise ValueError(
          "Can't use Insert method for a tensor constructed from a file.")
    if not isinstance(coords, list):
      raise ValueError(f"Non list coordinate detected: {coords}.")
    if not _all_instance_of(coords, int):
      raise ValueError(f"Non integer coordinate detected: {coords}.")
    if (len(coords) != self.order or
        any([c < 0 or c >= self._shape[i] for i, c in enumerate(coords)])):
      raise ValueError("Invalid coordinate for rank: "
                       f"{self.order}, {coords}.")

    if not isinstance(val, int) and not isinstance(val, float):
      raise ValueError(f"Value is neither int nor float: {val}.")

    self._coords.append(tuple(coords))
    self._values.append(self._dtype.value(val))

  def is_dense(self) -> bool:
    """Returns true if the tensor doesn't have sparsity annotation."""
    return self._format is None

  def to_array(self) -> np.ndarray:
    """Returns the numpy array for the Tensor.

    This is currenly only implemented for dense Tensor.
    """
    if not self.is_dense():
      raise ValueError("Conversion from non-dense Tensor "
                       "to numpy array not supported yet.")

    self._sync_value()

    return self._dense_storage

  @staticmethod
  def from_array(array: np.ndarray) -> "Tensor":
    """Returns a dense tensor with the value copied from the input array.

    We currently only support the conversion of float64 numpy arrays to Tensor.

    Args:
      array: The numpy array that provides the data type, shape and value for
        the tensor.

    Returns:
      A Tensor object.

    Raises:
      ValueError if the data type of the numpy array is not float64.
    """
    if array.dtype != np.float64:
      raise ValueError(f"Expected float64 value type: {array.dtype}.")
    tensor = Tensor(array.shape, is_dense=True)
    tensor._dense_storage = np.copy(array)
    return tensor

  @staticmethod
  def from_coo(
      coordinates: List[Tuple[int, ...]],
      values: List[_AnyRuntimeType],
      fmt: Format,
      dtype: DType,
  ) -> "Tensor":
    """Converts coordinates and values to a sparse tensor representation.

    Args:
      coordinates: A list of coordinates with non-zero values.
      values: The non-zero values.
      fmt: The tensor storage format.
      dtype: The tensor element data type.

    Returns:
      A tensor with the given non-zero values and storage format. The shape of
      the tensor has the minimum size for each dimension to make the given
      coordinates valid.
    """
    assert (isinstance(coordinates, List) and
            _all_instance_of(coordinates, Tuple))
    assert (isinstance(values, List) and _all_instance_of(values, dtype.value))
    assert isinstance(fmt, Format)

    rank = fmt.rank()
    assert all(len(c) == rank and _all_instance_of(c, int) for c in coordinates)

    # Find the maximum coordinate value for each dimension.
    max_coordinate = list(map(max, zip(*coordinates)))
    # The size of each dimension is one more that such a maximum coordinate
    # value.
    shape = [c + 1 for c in max_coordinate]
    tensor = Tensor(shape, fmt)
    tensor._coords = coordinates
    tensor._values = values

    return tensor

  @staticmethod
  def from_file(
      filename: str,
      fmt: Format,
      dtype: DType,
  ) -> "Tensor":
    """Constructs a sparse tensor using the COO-flavored values from a file.

    Args:
      filename: A string for the name of the file that contains the sparse
        tensor data.
      fmt: The tensor storage format.
      dtype: The tensor element data type.

    Returns:
      A tensor with the given non-zero values and storage format. The tensor
      value is stored as an MLIR sparse tensor.
    """
    sparse_tensor, shape = utils.create_sparse_tensor(filename,
                                                      fmt.format_pack.formats)
    tensor = Tensor(shape.tolist(), fmt)
    tensor._set_packed_sparse_tensor(sparse_tensor)

    return tensor

  def to_file(self, filename: str) -> None:
    """Output the tensor value to a file.

    This method evaluates any pending assignment to the tensor and outputs the
    tensor value.

    Args:
      filename: A string file name.

    Raises:
       ValueError: If the tensor is dense, or an unpacked sparse tensor.
    """
    self._sync_value()

    if self.is_dense():
      raise ValueError("Writing dense tensors without sparsity annotation to "
                       "file is not supported.")

    if self.is_unpacked():
      raise ValueError("Writing unpacked sparse tensors to file is not "
                       "supported.")

    utils.output_sparse_tensor(self._packed_sparse_value, filename,
                               self._format.format_pack.formats)

  @property
  def dtype(self) -> DType:
    """Returns the data type for the Tensor."""
    return self._dtype

  @property
  def format(self) -> Format:
    """Returns the storage format for the Tensor."""
    return self._format

  @property
  def name(self) -> str:
    """Returns the name for the Tensor."""
    return self._name

  @property
  def order(self) -> int:
    """Returns the rank of the Tensor."""
    return len(self._shape)

  @property
  def shape(self) -> List[int]:
    """Returns the shape of the Tensor."""
    return self._shape

  def __getitem__(self, key) -> "Access":
    """Verifies and processes a tensor access.

    In the tensor index notation, a tensor access T[i, j] is represented as
    retrieving a value with key (i, j) from the tensor object T in Python. This
    routine verifies the key for the tensor access and returns a tensor access
    object.

    Args:
      key: The key used to access the tensor, which could be any Python object
        from user inputs.

    Returns:
      The corresponding tensor access object.

    Raises:
      ValueError: If key is not an IndexVar or a tuple of IndexVar.
    """
    indices = _verify_and_normalize_indices(key)
    return Access(self, indices)

  def __setitem__(self, key, value) -> None:
    """Verifies and processes a tensor assignment.

    In the tensor index notation, a tensor assignment "T[i, j] = ..." is
    represented as setting a value for a tensor object T via key (i, j) in
    Python. This routine verifies the key, evaluates the value, and assigns the
    value to the tensor.

    We only support assignment of dense tensor currently.

    Args:
      key: The key used to access the tensor, which could be any Python object
        from user inputs.
      value: The value assigned to the tensor, which could be any Python object
        from user inputs.

    Raises:
      ValueError: If tensor is not a dense tensor, or the key is not an IndexVar
        or a tuple of IndexVar, or the length of the indices is not the same as
        the rank of the tensor.
    """
    indices = _verify_and_normalize_indices(key)
    if len(indices) != self.order:
      raise ValueError("Mismatch between indices and tensor rank: "
                       f"len({indices}) != {self.order}.")

    self._assignment = _Assignment(indices, value)

  def evaluate(self) -> None:
    """Evaluates the assignment to the tensor."""
    result = self._assignment.expression.evaluate(self,
                                                  self._assignment.indices)
    self._assignment = None
    if self.is_dense():
      assert isinstance(result, np.ndarray)
      self._dense_storage = result
    else:
      self._set_packed_sparse_tensor(result)

  def _sync_value(self) -> None:
    """Updates the tensor value by evaluating the pending assignment."""
    if self._assignment is not None:
      self.evaluate()

  def mlir_tensor_type(self) -> ir.RankedTensorType:
    """Returns the MLIR type for the tensor."""
    mlir_attr = None if (
        self._format is None) else self._format.mlir_tensor_attr()
    return _mlir_tensor_type(self._dtype, tuple(self._shape), mlir_attr)

  def dense_dst_ctype_pointer(self) -> ctypes.pointer:
    """Returns the ctypes pointer for the pointer to an MemRefDescriptor.

    For a dense tensor output, the MLIR compiler allocates the storage for
    the tensor. This routine returns the pointer to an MLIR MemRefDescriptor for
    receiving the tensor.
    """
    assert self.is_dense()
    mem_ref_desc = runtime.make_nd_memref_descriptor(
        self.order, np.ctypeslib.as_ctypes_type(self.dtype.value))()
    return ctypes.pointer(ctypes.pointer(mem_ref_desc))

  def ctype_pointer(self) -> ctypes.pointer:
    """Returns the ctypes pointer for the pointer to the input tensor."""
    if self.is_dense():
      if self._dense_storage is None:
        self._dense_storage = np.zeros(self._shape, self._dtype.value)
      return _ctype_pointer_from_array(self._dense_storage)

    if self.is_unpacked():
      shape = np.array(self._shape, np.int64)
      indices = np.array(self._coords, np.int64)
      values = np.array(self._values, self._dtype.value)
      ptr = utils.coo_tensor_to_sparse_tensor(shape, values, indices)
    else:
      ptr = self._packed_sparse_value

    return ctypes.pointer(ctypes.cast(ptr, ctypes.c_void_p))

  def get_coordinates_and_values(
      self) -> Tuple[List[Tuple[int, ...]], List[_AnyRuntimeType]]:
    """Returns the coordinates and values for the non-zero elements.

    This method also evaluate the assignment to the tensor and unpack the
    sparse tensor.
    """
    self._sync_value()

    if not self.is_dense():
      self.unpack()
      return (self._coords, self._values)

    # Coordinates for non-zero elements, grouped by dimensions.
    coords_by_dims = self._dense_storage.nonzero()
    # Coordinates for non-zero elements, grouped by elements.
    coords = np.transpose(coords_by_dims)
    values = self._dense_storage[coords_by_dims]
    return (coords, values)

  def _record_stats(self, structop: "_StructOpInfo"):
    """Collects information for temporary tensors."""
    # Exclude user specified destination tensors.
    if structop.dst_name == self.name:
      return

    self._stats.add_element(structop)


def _emit_operand(op_def: lang.LinalgOpDef, indices: Tuple[IndexVar, ...],
                  name: str, kind: lang.OperandKind) -> lang.OperandDef:
  """Emits an operand for a tensor access in the current linalg operation.

  Args:
    op_def: A LinalgOpDef representing the current linalg dialect operation.
    indices: A tuple of IndexVar used to access the tensor.
    name: A unique string name of the tensor.
    kind: An OperandKind for the operand.

  Returns:
    An OperandDef representing the operand.
  """
  dim_sym = _mlir_symbols_from_index_vars(indices)
  opnd = lang.OperandDef(kind, lang.T, dim_sym)
  op_def.add_operand(name, opnd)
  return opnd


@dataclasses.dataclass(frozen=True)
class _DimInfo:
  """Information for an operand dimension.

  Attributes:
    dim: An integer for the size of the dimension.
    mode_format: A ModeFormat for the dimension sparsity.
  """
  dim: int
  mode_format: ModeFormat


@dataclasses.dataclass()
class _ExprInfo:
  """Expression information for validation and code generation.

  Attributes:
    src_indices: A tuple of IndexVar for the indices used by the tensors in the
      expression tree.
    dim_infos: A tuple of _DimInfo, representing the dimension information
      corresponding to the src_indices.
    reduce_indices: A set of IndexVar for the indices reduced by the expression.
    acc_reduce_indices: An accumulated set of IndexVar for the indices reduced
      by the expression and its children.
    structop_info: Information to support the code generation for a structured
      op in the linalg dialect, if the corresponding expression node is the root
      of a subtree for a structured op.
    mlir_value: The MLIR value generated for the structured op.
  """
  src_indices: Tuple[IndexVar, ...]
  dim_infos: Tuple[_DimInfo, ...]
  reduce_indices: Optional[Set[IndexVar]] = None
  acc_reduce_indices: Optional[Set[IndexVar]] = None
  structop_info: Optional[_StructOpInfo] = None
  mlir_value: Optional[ir.Value] = None

  def __post_init__(self) -> None:
    """Verifies and fix up attribute values.

    Verifies the consistency of the attributes and modifies the default values
    to support convenient initializer syntax.
    """
    assert len(self.src_indices) == len(self.dim_infos)
    self.reduce_indices = self.reduce_indices or set()
    self.acc_reduce_indices = self.acc_reduce_indices or set()


class IndexExpr(abc.ABC):
  """The index notation base class.

  We support the TACO API index_expression class with an alias of this class.
  """

  def _verify_operand_and_build_expr(self, rhs, op: _BinaryOp) -> "_BinaryExpr":
    """Verifies the RHS operand and returns a binary expression.

    Args:
      rhs: The RHS of the binary operation, which could be any Python object
        from user inputs.
      op: A _BinaryOp object representing the binary operator.

    Raises:
      ValueError: If rhs is not an IndexExpr.
    """
    if not isinstance(rhs, IndexExpr):
      raise ValueError(f"Expected IndexExpr: {rhs}")
    return _BinaryExpr(op, self, rhs)

  def __add__(self, rhs) -> "_BinaryExpr":
    """Defines the operator +.

    Args:
      rhs: The value being added, which could be any Python object from user
        inputs.

    Returns:
      A _BinaryExpr object representing the operation.

    Raises:
      ValueError: If rhs is not an IndexExpr.
    """
    return self._verify_operand_and_build_expr(rhs, operator.add)

  def __mul__(self, rhs) -> "_BinaryExpr":
    """Defines the operator *.

    Args:
      rhs: The value being multiplied, which could be any Python object from
        user inputs.

    Returns:
      A _BinaryExpr object representing the operation.

    Raises:
      ValueError: If rhs is not an IndexExpr.
    """
    return self._verify_operand_and_build_expr(rhs, operator.mul)

  def __sub__(self, rhs) -> "_BinaryExpr":
    """Defines the operator -.

    Args:
      rhs: The value being subtracted, which could be any Python object from
        user inputs.

    Returns:
      A _BinaryExpr object representing the operation.

    Raises:
      ValueError: If rhs is not an IndexExpr.
    """
    return self._verify_operand_and_build_expr(rhs, operator.sub)

  @abc.abstractmethod
  def _visit(self,
             func: _ExprVisitor,
             args,
             *,
             leaf_checker: _SubtreeLeafChecker = None) -> None:
    """A post-order visitor.

    Args:
      func: A callable applied to each node in the expression tree.
      args: The variable-length arguments passed to the callable. These
        arguments are grouped as an iterable and will be unpacked before passing
        to the callable. This is to enable the keyword argument only syntax
        after this argument.
      leaf_checker: A callable object to identify nodes that should be treated
        as leaf nodes to support partial tree visiting.
    """
    pass

  @abc.abstractmethod
  def _emit_expression(
      self,
      expr_to_opnd: Dict["IndexExpr", lang.OperandDef],
      expr_to_info: _ExprInfoDict,
  ) -> lang.ScalarExpression:
    """Emits MLIR for the expression tree.

    Args:
      expr_to_opnd: A dictionary for looking up structured op input operands for
        the input nodes of the structured op.
      expr_to_info: A dictionary for looking up code generation information for
        expressions.

    Returns:
      A linalg dialect ScalarExpression for the expression.
    """
    pass

  @abc.abstractmethod
  def dtype(self) -> DType:
    """Returns the data type for the result of the expression."""
    pass

  def _emit_structured_op(self, expr_to_info: _ExprInfoDict) -> None:
    """Emits a structured op in the linalg dialect for the expression tree.

    We define a DefineOpcallable in the domain specific language for the linalg
    dialect and execute the callable to generate the structured op. Self is the
    root of the expression tree for the structured op.

    Args:
      expr_to_info: A dictionary for looking up code generation information for
        expressions.
    """
    op_info = expr_to_info[self].structop_info
    op_name = op_info.dst_name
    op_def = lang.LinalgOpDef(name=op_name)
    op_callable = lang.DefinedOpCallable(op_name, op_def)

    # Collect the input expression nodes for the structured op.
    expr_inputs = []
    self._visit(
        _gather_structured_op_input,
        (self, expr_to_info, expr_inputs),
        leaf_checker=_is_structured_op_leaf,
    )

    # Create a linalg structured op operand for each input expression node and
    # build a dictionary for looking up the information.
    expr_to_input_opnd = {
        e: _emit_structured_op_input(e, expr_to_info, op_def)
        for e in expr_inputs
    }

    # Emit the expression tree, which produces the value assigned to the
    # destination tensor.
    value = self._emit_expression(expr_to_input_opnd, expr_to_info)
    # Emit the structured op representation for the destination tensor.
    dst_opnd = _emit_operand(op_def, op_info.dst_indices, op_info.dst_name,
                             lang.OperandKind.OutputTensor)
    dst_dim_syms = _mlir_dimensions_from_index_vars(op_info.dst_indices)
    dst_use = lang.TensorUse(dst_opnd, dst_dim_syms)

    expr_info = expr_to_info[self]
    # If the structured op reduces some indices, explicitly represent the
    # reduction. This is done by generating a ReduceFn for the dimensions being
    # reduced in the linalg dialect and calling the function with the value
    # being reduced. We only support add reduction currently.
    if expr_info.reduce_indices:
      reduce_dims = _mlir_dimensions_from_index_vars(expr_info.reduce_indices)
      value = lang.ReduceFn.add[reduce_dims](value)

    # Emit the assignment as a comprehension in the linalg dialect.
    comp = lang.Comprehension((dst_use, value))
    op_def.comprehensions.append(comp)

    # The structured op in the linalg dialect requires an explicit
    # initialization for the destination tensor. Emit MLIR to initialize the
    # destination tensor.
    init = op_info.emit_tensor_init()

    # Collect MLIR values for the linalg input operands, with the assumption
    # that dictionary preserves the insertion order.
    args = [
        expr_to_info[expr].mlir_value
        for expr, opnd in expr_to_input_opnd.items()
    ]
    # Execute the DefineOpcallable object for the linalg dialect operation to
    # emit MLIR for the linalg structured op.
    expr_info.mlir_value = op_callable(*args, outs=[init])

  def _identify_structured_ops(
      self,
      expr_to_info: _ExprInfoDict,
      dst: Tensor,
      dst_indices: Tuple[IndexVar, ...],
  ) -> List["IndexExpr"]:
    """Returns expression nodes for the roots of the identified structured ops.

    A structured op in the linalg dialect only supports reduction performed on
    the whole expression. If the expression tree contains reduction that are
    performed on part of the expression tree, the expression tree needs to be
    implemented with multiple structured ops. This routine identifies all the
    expression nodes that contain reduction as the root of structured ops in the
    linalg dialect.

    Args:
      expr_to_info: A dictionary for looking up code generation information for
        expressions.
      dst: A destination Tensor that accepts the value of the expression tree.
      dst_indices: The indices used by the destination index expression.

    Returns:
      An ordered list of IndexExpr for the root expressions of the structured
      ops, where child expressions go before parent expressions that use their
      results.
    """
    reduce_indices = tuple(
        set(expr_to_info[self].src_indices) - set(dst_indices))
    for reduce_index in reduce_indices:
      _mark_structured_op_root(self, reduce_index, expr_to_info)

    self._visit(_accumulate_reduce_indices, (expr_to_info,))
    structop_roots = []
    self._visit(_gather_structured_op, (expr_to_info, structop_roots))

    # Handle the root of the top level expression.
    if not structop_roots or structop_roots[-1] != self:
      # The top level expression is not a reduction. Add the top level
      # expression as a structured op root.
      structop_roots.append(self)

    # Use user specified information for the destination tensor to build an
    # _StructOpInfo for the top level expression.
    expr_to_info[self].structop_info = _StructOpInfo(dst_indices,
                                                     tuple(dst.shape),
                                                     self.dtype(), dst.name,
                                                     dst.format)

    return structop_roots

  def _validate_and_collect_expr_info(
      self,
      dst: Tensor,
      dst_indices: Tuple[IndexVar, ...],
  ) -> _ExprInfoDict:
    """Propagates expression information for validation.

    Propagates the indices used by child expression nodes to parent expression
    nodes. Also collects and validates the sizes for the dimensions
    corresponding to the indices.

    Args:
      dst: A destination Tensor that accepts the value of the expression tree.
      dst_indices: The indices used by the destination index expression.

    Raises:
      ValueError if there is any inconsistency in indices or dimensional
      values.

    Returns:
      A dictionary of (IndexExpr, _ExprInfo).
    """
    expr_to_info = {}
    # Validate the expression tree and construct expression information.
    self._visit(_validate_and_collect_expr_info, (expr_to_info,))

    # Validate the destination dimension information.
    info = expr_to_info[self]
    index_to_dim_info = {i: d for i, d in zip(info.src_indices, info.dim_infos)}
    for i, d, in zip(dst_indices, dst.shape):
      if i not in index_to_dim_info:
        raise ValueError("Destination IndexVar not used in the "
                         f"source expression: {i}")
      else:
        if d != index_to_dim_info[i].dim:
          raise ValueError(f"Inconsistent destination dimension for {i}: "
                           f"{d} vs {index_to_dim_info[i].dim}")

    return expr_to_info

  def _emit_assignment(
      self,
      module: ir.Module,
      dst: Tensor,
      dst_indices: Tuple[IndexVar, ...],
      expr_to_info: _ExprInfoDict,
      input_accesses: List["Access"],
  ) -> None:
    """Emits an MLIR function for assigning the expression to a tensor."""
    input_types = [a.tensor.mlir_tensor_type() for a in input_accesses]

    # Build the kernel for the operations.
    with ir.InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(*input_types, name=_ENTRY_NAME)
      def linalg_funcop(*args):
        # Set up the mapping from the Access nodes to their MLIR values.
        for e, mlir in zip(input_accesses, args):
          expr_to_info[e].mlir_value = mlir

        # Emit structured ops in the linalg dialect to implement the assignment.
        for structop_root in self._identify_structured_ops(
            expr_to_info, dst, dst_indices):
          structop_root._emit_structured_op(expr_to_info)
          dst._record_stats(expr_to_info[structop_root].structop_info)

        # The function returns the MLIR value of the root expression.
        return expr_to_info[self].mlir_value

      linalg_funcop.func_op.attributes[
          "llvm.emit_c_interface"] = ir.UnitAttr.get()

  def evaluate(
      self,
      dst: Tensor,
      dst_indices: Tuple[IndexVar, ...],
  ) -> Union[np.ndarray, ctypes.c_void_p]:
    """Evaluates tensor assignment dst[dst_indices] = expression.

    Args:
      dst: The destination tensor.
      dst_indices: The tuple of IndexVar used to access the destination tensor.

    Returns:
      The result of the dense tensor represented in numpy ndarray or the pointer
      to the MLIR sparse tensor.

    Raises:
      ValueError: If the expression is not proper or not supported.
    """
    expr_to_info = self._validate_and_collect_expr_info(dst, dst_indices)

    # Compute a list of input accesses.
    input_accesses = []
    self._visit(_gather_input_accesses_index_vars, (input_accesses,))

    # Build and compile the module to produce the execution engine.
    with ir.Context(), ir.Location.unknown():
      module = ir.Module.create()
      self._emit_assignment(module, dst, dst_indices, expr_to_info,
                            input_accesses)
      engine = utils.compile_and_build_engine(module)

    # Gather the pointers for the input buffers.
    input_pointers = [a.tensor.ctype_pointer() for a in input_accesses]
    if dst.is_dense():
      # The pointer to receive dense output is the first argument to the
      # execution engine.
      arg_pointers = [dst.dense_dst_ctype_pointer()] + input_pointers
    else:
      # The pointer to receive sparse output is the last argument to the
      # execution engine. The pointer to receive a sparse tensor output is a
      # pointer to pointer of char.
      arg_pointers = input_pointers + [
          ctypes.pointer(ctypes.pointer(ctypes.c_char(0)))
      ]

    # Invoke the execution engine to run the module and return the result.
    engine.invoke(_ENTRY_NAME, *arg_pointers)

    if dst.is_dense():
      return runtime.ranked_memref_to_numpy(arg_pointers[0][0])

    # Return the sparse tensor pointer.
    return arg_pointers[-1][0]


@dataclasses.dataclass(frozen=True)
class Access(IndexExpr):
  """The tensor access class.

  We support the TACO API access class with an alias of this class.

  Attributes:
    tensor: A Tensor being accessed.
    indices: A tuple of IndexVar, representing the indices used to access the
      Tensor.
  """
  tensor: Tensor
  indices: Tuple[IndexVar, ...]

  def __post_init__(self) -> None:
    """Verifies the tensor and indices for a tensor access.

    Raises:
       ValueError: If indices is not a list of IndexVar or the len of indices
       doesn't equal to the rank of the tensor.
    """
    if (not isinstance(self.indices, tuple) or
        not _all_instance_of(self.indices, IndexVar)):
      raise ValueError(f"Indices contain non IndexVar: {str(self.indices)}.")
    if self.tensor.order != len(self.indices):
      raise ValueError("Invalid indices for rank: "
                       f"str{self.tensor.order} != len({str(self.indices)}).")

  def __repr__(self) -> str:
    # The Tensor __repr__ method evaluates the pending assignment to the tensor.
    # We want to define the __repr__ method here to avoid such evaluation of the
    # tensor assignment.
    indices_str = ", ".join(map(lambda i: i.name, self.indices))
    return (f"Tensor({self.tensor.name}) " f"Indices({indices_str})")

  def _emit_expression(
      self,
      expr_to_opnd: Dict[IndexExpr, lang.OperandDef],
      expr_to_info: _ExprInfoDict,
  ) -> lang.ScalarExpression:
    """Emits a linalg dialect TensorUse expression for the tensor access."""
    assert self in expr_to_opnd
    dims = _mlir_dimensions_from_index_vars(self.indices)
    return lang.TensorUse(expr_to_opnd[self], dims)

  def _visit(self,
             func: _ExprVisitor,
             args,
             *,
             leaf_checker: _SubtreeLeafChecker = None) -> None:
    if leaf_checker:
      assert leaf_checker(self, *args)
    func(self, *args)

  def dtype(self) -> DType:
    return self.tensor.dtype


def _gather_input_accesses_index_vars(
    expr: IndexExpr,
    input_accesses: List[Access],
) -> None:
  """Collects Access nodes."""
  if isinstance(expr, Access) and expr not in input_accesses:
    input_accesses.append(expr)


def _op_to_callable(op: _BinaryOp) -> lang.ArithFnType:
  """Returns the linalg dialect function object for the given operation."""
  op_to_callable = {
      operator.add: lang.ArithFn.add,
      operator.sub: lang.ArithFn.sub,
      operator.mul: lang.ArithFn.mul,
  }
  return op_to_callable[op]


@dataclasses.dataclass(frozen=True)
class _BinaryExpr(IndexExpr):
  """The representation for a binary operation.

  Attributes:
  op: A _BinaryOp representing the binary operation.
  a: An IndexExpr representing the first operand of the operation.
  b: An IndexExpr representing the second operand of the operation.
  """
  op: _BinaryOp
  a: IndexExpr
  b: IndexExpr

  def __post_init__(self) -> None:
    """Verifies that the operands being added are IndexExpr."""
    assert isinstance(self.a, IndexExpr) and isinstance(self.b, IndexExpr)

  def _emit_expression(
      self,
      expr_to_opnd: Dict[IndexExpr, lang.OperandDef],
      expr_to_info: _ExprInfoDict,
  ) -> lang.ScalarExpression:
    """Emits the expression tree and returns the expression."""
    # The current expression node is an internal node of the structured op.
    if self not in expr_to_opnd:
      a = self.a._emit_expression(expr_to_opnd, expr_to_info)
      b = self.b._emit_expression(expr_to_opnd, expr_to_info)
      return _op_to_callable(self.op)(a, b)

    # The current expression is a leaf node of the structured op. That is, it is
    # a temporary tensor generated by its child structured op.
    op_info = expr_to_info[self].structop_info
    assert op_info is not None
    dims = _mlir_dimensions_from_index_vars(op_info.dst_indices)
    return lang.TensorUse(expr_to_opnd[self], dims)

  def _visit(self,
             func: _ExprVisitor,
             args,
             *,
             leaf_checker: _SubtreeLeafChecker = None) -> None:
    """A post-order visitor."""
    if leaf_checker is None or not leaf_checker(self, *args):
      self.a._visit(func, args, leaf_checker=leaf_checker)
      self.b._visit(func, args, leaf_checker=leaf_checker)
    func(self, *args)

  def dtype(self) -> DType:
    """Returns the data type of the binary operation."""
    return self.a.dtype()


def _validate_and_collect_dim_info(
    index_to_dim_info: Dict[IndexVar, _DimInfo],
    indices: Tuple[IndexVar, ...],
    dim_infos: Tuple[_DimInfo, ...],
    expr: _BinaryExpr,
) -> None:
  """Validates and collects the dimension information for an index notation.

  Validates (indices, dim_infos) against the information collected from other
  source operands and is represented by index_to_dim_info. In particular, we
  ensure that each IndexVar corresponds to only one dimension size. We also
  aggregate the new information represented in (indices, dim_infos) to
  index_to_dim_info.

  Args:
    index_to_dim: A dictionary of (IndexVar, _DimInfo) collected from the
      previous operands.
    indices: The IndexVars to be validated.
    dim_infos: The dimension information for the IndexVars to be validated.
    expr: The binary expression where (indices, dim_infos) is used.

  Raises:
    ValueError if there is any problem in the IndexVars or dimensional values.
  """
  assert len(indices) == len(dim_infos)
  for i, d in zip(indices, dim_infos):
    if i not in index_to_dim_info:
      index_to_dim_info[i] = d
    else:
      if d.dim != index_to_dim_info[i].dim:
        raise ValueError(f"Inconsistent source dimension for {i}: "
                         f"{d.dim} vs {index_to_dim_info[i].dim}")
      mode_format = _mode_format_estimator(expr.op)(
          index_to_dim_info[i].mode_format, d.mode_format)
      index_to_dim_info[i] = _DimInfo(d.dim, mode_format)


def _validate_and_collect_expr_info(
    expr: IndexExpr,
    expr_to_info: _ExprInfoDict,
) -> None:
  """Validates dimension information and constructs _ExprInfo.

  Validates that dimensional values for the same IndexVar are the same. Collects
  a list of IndexVar used by the expression and their corresponding dimensional
  values. Constructs an _ExprInfo object to record the information for the
  IndexExpr.

  This routine is passed to the post-order visitor as an _ExprVisitor object.

  Args:
    expr: The IndexExpr being validated.
    expr_to_info: The dictionary of (IndexExpr, _ExprInfo) for recording the
      expression information.

  Raises:
    ValueError if there is any problem in the IndexVars or dimensional values.
  """
  # Objects of class Access can be shared by different expressions. Avoid
  # processing Access objects multiple times by skipping the processing if expr
  # is already in the dictionary.
  if expr in expr_to_info:
    return

  if isinstance(expr, Access):
    src_indices = expr.indices
    src_dims = tuple(expr.tensor.shape)
    if expr.tensor.format is None:
      # Treat each dimension of a dense tensor as DENSE for the purpose of
      # calculating temporary tensor storage format.
      mode_formats = tuple([ModeFormat.DENSE] * len(src_dims))
    else:
      mode_formats = tuple(expr.tensor.format.format_pack.formats)
    assert len(src_dims) == len(mode_formats)
    dim_infos = tuple([_DimInfo(d, m) for d, m in zip(src_dims, mode_formats)])
  else:
    assert isinstance(expr, _BinaryExpr)
    a_info = expr_to_info[expr.a]
    index_to_dim_info = {
        i: d for i, d in zip(a_info.src_indices, a_info.dim_infos)
    }
    b_info = expr_to_info[expr.b]
    _validate_and_collect_dim_info(index_to_dim_info, b_info.src_indices,
                                   b_info.dim_infos, expr)
    # Here we rely on the fact that dictionaries keep the insertion order for
    # keys and values.
    src_indices = tuple(index_to_dim_info.keys())
    dim_infos = tuple(index_to_dim_info.values())

  expr_to_info[expr] = _ExprInfo(src_indices, dim_infos)


def _mark_structured_op_root(
    expr: IndexExpr,
    reduce_index: IndexVar,
    expr_to_info: _ExprInfoDict,
) -> None:
  """Identifies the root expression for a structured op in the linalg dialect.

  An linalg structured op can only perform reduction on the whole expression.
  For a TACO tensor algebra expression, the reduction on an IndexVar is done at
  the smallest expression that contains all the uses of the IndexVar. If such an
  expression is only part of the whole expression, we need to split this
  sub-expression tree out from its parent and implement the sub-expression as a
  structured op.

  This routine identifies the root expression node for performing a reduction on
  the given IndexVar. If the reduction of the given IndexVar should be performed
  on expression X, then the IndexVar is added to expr_to_info[X].reduce_indices

  Args:
    expr: The root IndexExpr for the tensor algebra expression.
    reduce_index: The IndexVar which we want to find out the proper expression
      to perform a reduction.
    expr_to_info: The dictionary to look up _ExprInfo for IndexExpr.
  """
  expr_info = expr_to_info[expr]
  if isinstance(expr, Access):
    # Handle simple reduction expression in the format of A[i] = B[i, j].
    if reduce_index in expr_info.src_indices:
      expr_info.reduce_indices.add(reduce_index)
    return

  assert (isinstance(expr, _BinaryExpr))
  a_info = expr_to_info[expr.a]
  b_info = expr_to_info[expr.b]

  if reduce_index in a_info.src_indices and reduce_index in b_info.src_indices:
    expr_info.reduce_indices.add(reduce_index)
    return

  if reduce_index in a_info.src_indices:
    _mark_structured_op_root(expr.a, reduce_index, expr_to_info)
  elif reduce_index in b_info.src_indices:
    _mark_structured_op_root(expr.b, reduce_index, expr_to_info)
  else:
    assert False, "Unreachable path"


def _accumulate_reduce_indices(
    expr: IndexExpr,
    expr_to_info: _ExprInfoDict,
) -> None:
  """Propagates reduction indices from child expressions to parent expressions.

  This routine is passed to the post-order visitor as an _ExprVisitor object.

  Args:
    expr: The IndexExpr being visited.
    expr_to_info: The dictionary of (IndexExpr, _ExprInfo) for recording the
      expression information.
  """
  assert expr in expr_to_info
  expr_info = expr_to_info[expr]

  if isinstance(expr, _BinaryExpr):
    a_info = expr_to_info[expr.a]
    b_info = expr_to_info[expr.b]
    expr_info.acc_reduce_indices = (
        a_info.acc_reduce_indices | b_info.acc_reduce_indices
        | expr_info.reduce_indices)
  else:
    assert isinstance(expr, Access)
    # Handle simple reduction expression in the format of A[i] = B[i, j].
    expr_info.acc_reduce_indices = expr_info.reduce_indices



def _gather_structured_op(
    expr: IndexExpr,
    expr_to_info: _ExprInfoDict,
    structop_roots: List[IndexExpr],
) -> None:
  """Adds structured op root expression information to structop_roots.

  This routine is passed to the post-order visitor as an _ExprVisitor object.

  Args:
    expr: The IndexExpr being visited.
    expr_to_info: The dictionary to look up _ExprInfo for IndexExpr.
    structop_roots: The resulting list of IndexExpr that are the roots for
      linalg structured ops.
  """
  if not expr_to_info[expr].reduce_indices:
    return

  # If the expression is the root for reducing some indices, collect the indices
  # and dimensions for the reduction result.
  dst_indices = []
  dst_dims = []
  mode_fmts = []
  for i, d in zip(expr_to_info[expr].src_indices, expr_to_info[expr].dim_infos):
    if i not in expr_to_info[expr].acc_reduce_indices:
      dst_indices.append(i)
      dst_dims.append(d.dim)
      mode_fmts.append(d.mode_format)

  # Add the information to the dictionary.
  op_info = _StructOpInfo(
      tuple(dst_indices),
      tuple(dst_dims),
      expr.dtype(),
      f"temp{len(structop_roots)}",
      _make_format(mode_fmts),
  )
  expr_to_info[expr].structop_info = op_info

  # Add the expression to the list of structured op roots.
  structop_roots.append(expr)


def _is_structured_op_leaf(
    expr: IndexExpr,
    root: IndexExpr,
    expr_to_info: _ExprInfoDict,
    *unused_args,
) -> bool:
  """Returns true iff the expression is a leaf node for a structured op.

  The root of a structured op is a leaf of its parent structured op that uses
  its result. An expression node is a leaf node for the current structured op if
  it is an Access node or the root for a structured op that is not the current
  structured op.

  This routine is passed to the post-order visitor as a _SubtreeLeafChecker
  object. Because the post-order visitor pass the same parameters to both
  _SubtreeLeafChecker and _ExprVisitor, this routine may received unused
  parameters.

  Args:
    expr: The IndexExpr being visited.
    root: The root of the current structured op.
    expr_to_info: The dictionary to look up _ExprInfo for IndexExpr.

  Returns:
    True if the current IndexExpr is a leaf for the current structured op.
  """
  return (expr != root and
          expr_to_info[expr].structop_info is not None) or isinstance(
              expr, Access)


def _gather_structured_op_input(
    expr: IndexExpr,
    root: IndexExpr,
    expr_to_info: _ExprInfoDict,
    structop_inputs: List[IndexExpr],
) -> None:
  """Adds the IndexExpr to structop_inputs if it is an input.

  If the current IndexExpr is an input for the current structured op, adds it to
  structop_inputs. The current IndexExpr is an input if it is an Access node or
  if it is the root for a structured op that is not the current structured op.

  This routine is passed to the post-order visitor as an _ExprVisitor object.

  Args:
    expr: The IndexExpr being visited.
    root: The root of the current structured op.
    expr_to_info: The dictionary to look up _ExprInfo for IndexExpr.
    structop_inputs: The resulting list of IndexExpr that provide input to the
      current structured op.
  """
  if ((expr != root or isinstance(expr, Access)) and
      expr not in structop_inputs) and (isinstance(expr, Access) or
                                        (expr in expr_to_info and
                                         expr_to_info[expr].structop_info)):
    structop_inputs.append(expr)


def _emit_structured_op_input(
    expr: IndexExpr,
    expr_to_info: _ExprInfoDict,
    op_def: lang.LinalgOpDef,
) -> lang.OperandDef:
  """Emits OperandDef in the linalg dialect for the input IndexExpr.

  Args:
    expr: The input IndexExpr for the current structured op.
    expr_to_info: The dictionary to look up _ExprInfo for IndexExpr.
    op_def: The linalg operation for the current structured op.

  Returns:
    An OperandDef in the linalg dialect for the input IndexExpr.
  """
  op_info = expr_to_info[expr].structop_info
  if op_info and not isinstance(expr, Access):
    # The input is a temporary tensor produced by another structured op.
    indices = op_info.dst_indices
    name = op_info.dst_name
  else:
    # The input is a user provided tensor.
    assert isinstance(expr, Access)
    indices = expr.indices
    name = expr.tensor.name

  dim_sym = _mlir_symbols_from_index_vars(indices)
  opnd = lang.OperandDef(lang.OperandKind.InputTensor, lang.T, dim_sym)
  op_def.add_operand(name, opnd)
  return opnd
