#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from ..ir import *
  from ..dialects import pdl
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e

from typing import Union, Optional, Sequence, List, Mapping
from ._ods_common import get_op_result_or_value as _get_value, get_op_results_or_values as _get_values


def _get_int_attr(bits: int, value: Union[IntegerAttr, int]) -> IntegerAttr:
  """Converts the given value to signless integer attribute of given bit width."""
  if isinstance(value, int):
    ty = IntegerType.get_signless(bits)
    return IntegerAttr.get(ty, value)
  else:
    return value


def _get_array_attr(attrs: Union[ArrayAttr, Sequence[Attribute]]) -> ArrayAttr:
  """Converts the given value to array attribute."""
  if isinstance(attrs, ArrayAttr):
    return attrs
  else:
    return ArrayAttr.get(list(attrs))


def _get_str_array_attr(attrs: Union[ArrayAttr, Sequence[str]]) -> ArrayAttr:
  """Converts the given value to string array attribute."""
  if isinstance(attrs, ArrayAttr):
    return attrs
  else:
    return ArrayAttr.get([StringAttr.get(s) for s in attrs])


def _get_str_attr(name: Union[StringAttr, str]) -> Optional[StringAttr]:
  """Converts the given value to string attribute."""
  if isinstance(name, str):
    return StringAttr.get(name)
  else:
    return name


def _get_type_attr(type: Union[TypeAttr, Type]) -> TypeAttr:
  """Converts the given value to type attribute."""
  if isinstance(type, Type):
    return TypeAttr.get(type)
  else:
    return type


class ApplyNativeConstraintOp:
  """Specialization for PDL apply native constraint op class."""

  def __init__(self,
               name: Union[str, StringAttr],
               args: Sequence[Union[OpView, Operation, Value]] = [],
               *,
               loc=None,
               ip=None):
    name = _get_str_attr(name)
    args = _get_values(args)
    super().__init__(name, args, loc=loc, ip=ip)


class ApplyNativeRewriteOp:
  """Specialization for PDL apply native rewrite op class."""

  def __init__(self,
               results: Sequence[Type],
               name: Union[str, StringAttr],
               args: Sequence[Union[OpView, Operation, Value]] = [],
               *,
               loc=None,
               ip=None):
    name = _get_str_attr(name)
    args = _get_values(args)
    super().__init__(results, name, args, loc=loc, ip=ip)


class AttributeOp:
  """Specialization for PDL attribute op class."""

  def __init__(self,
               type: Optional[Union[OpView, Operation, Value]] = None,
               value: Optional[Attribute] = None,
               *,
               loc=None,
               ip=None):
    type = type if type is None else _get_value(type)
    result = pdl.AttributeType.get()
    super().__init__(result, type, value, loc=loc, ip=ip)


class EraseOp:
  """Specialization for PDL erase op class."""

  def __init__(self,
               operation: Optional[Union[OpView, Operation, Value]] = None,
               *,
               loc=None,
               ip=None):
    operation = _get_value(operation)
    super().__init__(operation, loc=loc, ip=ip)


class OperandOp:
  """Specialization for PDL operand op class."""

  def __init__(self,
               type: Optional[Union[OpView, Operation, Value]] = None,
               *,
               loc=None,
               ip=None):
    type = type if type is None else _get_value(type)
    result = pdl.ValueType.get()
    super().__init__(result, type, loc=loc, ip=ip)


class OperandsOp:
  """Specialization for PDL operands op class."""

  def __init__(self,
               types: Optional[Union[OpView, Operation, Value]] = None,
               *,
               loc=None,
               ip=None):
    types = types if types is None else _get_value(types)
    result = pdl.RangeType.get(pdl.ValueType.get())
    super().__init__(result, types, loc=loc, ip=ip)


class OperationOp:
  """Specialization for PDL operand op class."""

  def __init__(self,
               name: Optional[Union[str, StringAttr]] = None,
               args: Sequence[Union[OpView, Operation, Value]] = [],
               attributes: Mapping[str, Union[OpView, Operation, Value]] = {},
               types: Sequence[Union[OpView, Operation, Value]] = [],
               *,
               loc=None,
               ip=None):
    name = name if name is None else _get_str_attr(name)
    args = _get_values(args)
    attributeNames = []
    attributeValues = []
    for attrName, attrValue in attributes.items():
      attributeNames.append(StringAttr.get(attrName))
      attributeValues.append(_get_value(attrValue))
    attributeNames = ArrayAttr.get(attributeNames)
    types = _get_values(types)
    result = pdl.OperationType.get()
    super().__init__(result, name, args, attributeValues, attributeNames, types, loc=loc, ip=ip)


class PatternOp:
  """Specialization for PDL pattern op class."""

  def __init__(self,
               benefit: Union[IntegerAttr, int],
               name: Optional[Union[StringAttr, str]] = None,
               *,
               loc=None,
               ip=None):
    """Creates an PDL `pattern` operation."""
    name_attr = None if name is None else _get_str_attr(name)
    benefit_attr = _get_int_attr(16, benefit)
    super().__init__(benefit_attr, name_attr, loc=loc, ip=ip)
    self.regions[0].blocks.append()

  @property
  def body(self):
    """Return the body (block) of the pattern."""
    return self.regions[0].blocks[0]


class ReplaceOp:
  """Specialization for PDL replace op class."""

  def __init__(self,
               op: Union[OpView, Operation, Value],
               *,
               with_op: Optional[Union[OpView, Operation, Value]] = None,
               with_values: Sequence[Union[OpView, Operation, Value]] = [],
               loc=None,
               ip=None):
    op = _get_value(op)
    with_op = with_op if with_op is None else _get_value(with_op)
    with_values = _get_values(with_values)
    super().__init__(op, with_op, with_values, loc=loc, ip=ip)


class ResultOp:
  """Specialization for PDL result op class."""

  def __init__(self,
               parent: Union[OpView, Operation, Value],
               index: Union[IntegerAttr, int],
               *,
               loc=None,
               ip=None):
    index = _get_int_attr(32, index)
    parent = _get_value(parent)
    result = pdl.ValueType.get()
    super().__init__(result, parent, index, loc=loc, ip=ip)


class ResultsOp:
  """Specialization for PDL results op class."""

  def __init__(self,
               result: Type,
               parent: Union[OpView, Operation, Value],
               index: Optional[Union[IntegerAttr, int]] = None,
               *,
               loc=None,
               ip=None):
    parent = _get_value(parent)
    index = index if index is None else _get_int_attr(32, index)
    super().__init__(result, parent, index, loc=loc, ip=ip)


class RewriteOp:
  """Specialization for PDL rewrite op class."""

  def __init__(self,
               root: Optional[Union[OpView, Operation, Value]] = None,
               name: Optional[Union[StringAttr, str]] = None,
               args: Sequence[Union[OpView, Operation, Value]] = [],
               *,
               loc=None,
               ip=None):
    root = root if root is None else _get_value(root)
    name = name if name is None else _get_str_attr(name)
    args = _get_values(args)
    super().__init__(root, name, args, loc=loc, ip=ip)

  def add_body(self):
    """Add body (block) to the rewrite."""
    self.regions[0].blocks.append()
    return self.body

  @property
  def body(self):
    """Return the body (block) of the rewrite."""
    return self.regions[0].blocks[0]


class TypeOp:
  """Specialization for PDL type op class."""

  def __init__(self,
               type: Optional[Union[TypeAttr, Type]] = None,
               *,
               loc=None,
               ip=None):
    type = type if type is None else _get_type_attr(type)
    result = pdl.TypeType.get()
    super().__init__(result, type, loc=loc, ip=ip)


class TypesOp:
  """Specialization for PDL types op class."""

  def __init__(self,
               types: Sequence[Union[TypeAttr, Type]] = [],
               *,
               loc=None,
               ip=None):
    types = _get_array_attr([_get_type_attr(ty) for ty in types])
    types = None if not types else types
    result = pdl.RangeType.get(pdl.TypeType.get())
    super().__init__(result, types, loc=loc, ip=ip)
