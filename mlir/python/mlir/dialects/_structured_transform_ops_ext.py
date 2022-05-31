#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from ..ir import *
  from ._ods_common import get_op_result_or_value as _get_op_result_or_value
  from ..dialects import pdl
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e

from typing import List, Optional, Sequence, Union

IntOrAttrList = Sequence[Union[IntegerAttr, int]]
OptionalIntList = Optional[Union[ArrayAttr, IntOrAttrList]]


def _get_array_attr(
    values: Optional[Union[ArrayAttr, Sequence[Attribute]]]) -> ArrayAttr:
  """Creates an array attribute from its operand."""
  if values is None:
    return ArrayAttr.get([])
  if isinstance(values, ArrayAttr):
    return values

  return ArrayAttr.get(values)


def _get_int_array_attr(
    values: Optional[Union[ArrayAttr, Sequence[Union[IntegerAttr, int]]]]
) -> ArrayAttr:
  """Creates an integer array attribute from its operand.

  If the operand is already an array attribute, forwards it. Otherwise treats
  the operand as a list of attributes or integers, possibly intersperced, to
  create a new array attribute containing integer attributes. Expects the
  thread-local MLIR context to have been set by the context manager.
  """
  if values is None:
    return ArrayAttr.get([])
  if isinstance(values, ArrayAttr):
    return values

  attributes = []
  for value in values:
    if isinstance(value, IntegerAttr):
      attributes.append(value)
    else:
      attributes.append(IntegerAttr.get(IntegerType.get_signless(64), value))
  return ArrayAttr.get(attributes)


def _get_int_int_array_attr(
    values: Optional[Union[ArrayAttr, Sequence[Union[ArrayAttr,
                                                     IntOrAttrList]]]]
) -> ArrayAttr:
  """Creates an array attribute containing array attributes of integers.

  If the operand is already an array attribute, forwards it. Otherwise treats
  the operand as a list of attributes or integers, potentially interpserced, to
  create a new array-of-array attribute. Expects the thread-local MLIR context
  to have been set by the context manager.
  """
  if values is None:
    return ArrayAttr.get([])
  if isinstance(values, ArrayAttr):
    return values

  return ArrayAttr.get([_get_int_array_attr(value) for value in values])


class DecomposeOp:
  """Specialization for DecomposeOp class."""

  def __init__(self, target: Union[Operation, Value], *, loc=None, ip=None):
    super().__init__(
        pdl.OperationType.get(),
        _get_op_result_or_value(target),
        loc=loc,
        ip=ip)


class GeneralizeOp:
  """Specialization for GeneralizeOp class."""

  def __init__(self, target: Union[Operation, Value], *, loc=None, ip=None):
    super().__init__(
        pdl.OperationType.get(),
        _get_op_result_or_value(target),
        loc=loc,
        ip=ip)


class InterchangeOp:
  """Specialization for InterchangeOp class."""

  def __init__(self,
               target: Union[Operation, Value],
               *,
               iterator_interchange: OptionalIntList = None,
               loc=None,
               ip=None):
    pdl_operation_type = pdl.OperationType.get()
    interchange_attr = _get_int_array_attr(iterator_interchange)
    super().__init__(
        pdl_operation_type,
        _get_op_result_or_value(target),
        iterator_interchange=interchange_attr,
        loc=loc,
        ip=ip)


class PadOp:
  """Specialization for PadOp class."""

  def __init__(self,
               target: Union[Operation, Value],
               *,
               padding_values: Optional[Union[ArrayAttr,
                                              Sequence[Attribute]]] = None,
               padding_dimensions: OptionalIntList = None,
               pack_paddings: OptionalIntList = None,
               hoist_paddings: OptionalIntList = None,
               transpose_paddings: Optional[Union[ArrayAttr, Sequence[Union[
                   ArrayAttr, IntOrAttrList]]]] = None,
               loc=None,
               ip=None):
    pdl_operation_type = pdl.OperationType.get()
    padding_values_attr = _get_array_attr(padding_values)
    padding_dimensions_attr = _get_int_array_attr(padding_dimensions)
    pack_paddings_attr = _get_int_array_attr(pack_paddings)
    hoist_paddings_attr = _get_int_array_attr(hoist_paddings)
    transpose_paddings_attr = _get_int_int_array_attr(transpose_paddings)
    super().__init__(
        pdl_operation_type,
        _get_op_result_or_value(target),
        padding_values=padding_values_attr,
        padding_dimensions=padding_dimensions_attr,
        pack_paddings=pack_paddings_attr,
        hoist_paddings=hoist_paddings_attr,
        transpose_paddings=transpose_paddings_attr,
        loc=loc,
        ip=ip)


class ScalarizeOp:
  """Specialization for ScalarizeOp class."""

  def __init__(self, target: Union[Operation, Value], *, loc=None, ip=None):
    pdl_operation_type = pdl.OperationType.get()
    super().__init__(
        pdl_operation_type, _get_op_result_or_value(target), loc=loc, ip=ip)


class TileOp:
  """Specialization for TileOp class."""

  def __init__(self,
               target: Union[Operation, Value],
               *,
               sizes: OptionalIntList = None,
               interchange: OptionalIntList = None,
               loc=None,
               ip=None):
    pdl_operation_type = pdl.OperationType.get()
    sizes_attr = _get_int_array_attr(sizes)
    num_loops = sum(
        v if v == 0 else 1 for v in self.__extract_values(sizes_attr))
    super().__init__(
        pdl_operation_type, [pdl_operation_type] * num_loops,
        _get_op_result_or_value(target),
        sizes=sizes_attr,
        interchange=_get_int_array_attr(interchange) if interchange else None,
        loc=loc,
        ip=ip)

  def __extract_values(self, attr: Optional[ArrayAttr]) -> List[int]:
    if not attr:
      return []
    return [IntegerAttr(element).value for element in attr]


class VectorizeOp:
  """Specialization for VectorizeOp class."""

  def __init__(self,
               target: Union[Operation, Value],
               *,
               vectorize_padding: Union[bool, BoolAttr] = False,
               loc=None,
               ip=None):
    pdl_operation_type = pdl.OperationType.get()
    if isinstance(vectorize_padding, bool):
      vectorize_padding = BoolAttr.get(vectorize_padding)
    super().__init__(
        pdl_operation_type,
        _get_op_result_or_value(target),
        vectorize_padding=vectorize_padding,
        loc=loc,
        ip=ip)
