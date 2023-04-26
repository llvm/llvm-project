#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from ..ir import *
  from ..dialects import pdl
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e

from typing import Union, Optional, Sequence, Mapping
from ._ods_common import (
    get_op_result_or_value as _get_value,
    get_op_results_or_values as _get_values,
)


class ApplyNativeConstraintOp:
  """Specialization for PDL apply native constraint op class."""

  def __init__(
      self,
      name: Union[str, StringAttr],
      args: Optional[Sequence[Union[OpView, Operation, Value]]] = None,
      *,
      loc=None,
      ip=None,
  ):
    if args is None:
      args = []
    args = _get_values(args)
    super().__init__(name, args, loc=loc, ip=ip)


class ApplyNativeRewriteOp:
  """Specialization for PDL apply native rewrite op class."""

  def __init__(
      self,
      results: Sequence[Type],
      name: Union[str, StringAttr],
      args: Optional[Sequence[Union[OpView, Operation, Value]]] = None,
      *,
      loc=None,
      ip=None,
  ):
    if args is None:
      args = []
    args = _get_values(args)
    super().__init__(results, name, args, loc=loc, ip=ip)


class AttributeOp:
  """Specialization for PDL attribute op class."""

  def __init__(
      self,
      valueType: Optional[Union[OpView, Operation, Value]] = None,
      value: Optional[Attribute] = None,
      *,
      loc=None,
      ip=None,
  ):
    valueType = valueType if valueType is None else _get_value(valueType)
    result = pdl.AttributeType.get()
    super().__init__(result, valueType=valueType, value=value, loc=loc, ip=ip)


class EraseOp:
  """Specialization for PDL erase op class."""

  def __init__(
      self,
      operation: Optional[Union[OpView, Operation, Value]] = None,
      *,
      loc=None,
      ip=None,
  ):
    operation = _get_value(operation)
    super().__init__(operation, loc=loc, ip=ip)


class OperandOp:
  """Specialization for PDL operand op class."""

  def __init__(
      self,
      type: Optional[Union[OpView, Operation, Value]] = None,
      *,
      loc=None,
      ip=None,
  ):
    type = type if type is None else _get_value(type)
    result = pdl.ValueType.get()
    super().__init__(result, valueType=type, loc=loc, ip=ip)


class OperandsOp:
  """Specialization for PDL operands op class."""

  def __init__(
      self,
      types: Optional[Union[OpView, Operation, Value]] = None,
      *,
      loc=None,
      ip=None,
  ):
    types = types if types is None else _get_value(types)
    result = pdl.RangeType.get(pdl.ValueType.get())
    super().__init__(result, valueType=types, loc=loc, ip=ip)


class OperationOp:
  """Specialization for PDL operand op class."""

  def __init__(
      self,
      name: Optional[Union[str, StringAttr]] = None,
      args: Optional[Sequence[Union[OpView, Operation, Value]]] = None,
      attributes: Optional[Mapping[str, Union[OpView, Operation,
                                              Value]]] = None,
      types: Optional[Sequence[Union[OpView, Operation, Value]]] = None,
      *,
      loc=None,
      ip=None,
  ):
    if types is None:
      types = []
    if attributes is None:
      attributes = {}
    if args is None:
      args = []
    args = _get_values(args)
    attrNames = []
    attrValues = []
    for attrName, attrValue in attributes.items():
      attrNames.append(StringAttr.get(attrName))
      attrValues.append(_get_value(attrValue))
    attrNames = ArrayAttr.get(attrNames)
    types = _get_values(types)
    result = pdl.OperationType.get()
    super().__init__(result,
                     args,
                     attrValues,
                     attrNames,
                     types,
                     opName=name,
                     loc=loc,
                     ip=ip)


class PatternOp:
  """Specialization for PDL pattern op class."""

  def __init__(
      self,
      benefit: Union[IntegerAttr, int],
      name: Optional[Union[StringAttr, str]] = None,
      *,
      loc=None,
      ip=None,
  ):
    """Creates an PDL `pattern` operation."""
    super().__init__(benefit, sym_name=name, loc=loc, ip=ip)
    self.regions[0].blocks.append()

  @property
  def body(self):
    """Return the body (block) of the pattern."""
    return self.regions[0].blocks[0]


class ReplaceOp:
  """Specialization for PDL replace op class."""

  def __init__(
      self,
      op: Union[OpView, Operation, Value],
      *,
      with_op: Optional[Union[OpView, Operation, Value]] = None,
      with_values: Optional[Sequence[Union[OpView, Operation, Value]]] = None,
      loc=None,
      ip=None,
  ):
    if with_values is None:
      with_values = []
    op = _get_value(op)
    with_op = with_op if with_op is None else _get_value(with_op)
    with_values = _get_values(with_values)
    super().__init__(op, with_values, replOperation=with_op, loc=loc, ip=ip)


class ResultOp:
  """Specialization for PDL result op class."""

  def __init__(
      self,
      parent: Union[OpView, Operation, Value],
      index: Union[IntegerAttr, int],
      *,
      loc=None,
      ip=None,
  ):
    parent = _get_value(parent)
    result = pdl.ValueType.get()
    super().__init__(result, parent, index, loc=loc, ip=ip)


class ResultsOp:
  """Specialization for PDL results op class."""

  def __init__(
      self,
      result: Type,
      parent: Union[OpView, Operation, Value],
      index: Optional[Union[IntegerAttr, int]] = None,
      *,
      loc=None,
      ip=None,
  ):
    parent = _get_value(parent)
    super().__init__(result, parent, index=index, loc=loc, ip=ip)


class RewriteOp:
  """Specialization for PDL rewrite op class."""

  def __init__(
      self,
      root: Optional[Union[OpView, Operation, Value]] = None,
      name: Optional[Union[StringAttr, str]] = None,
      args: Optional[Sequence[Union[OpView, Operation, Value]]] = None,
      *,
      loc=None,
      ip=None,
  ):
    if args is None:
      args = []
    root = root if root is None else _get_value(root)
    args = _get_values(args)
    super().__init__(args, root=root, name=name, loc=loc, ip=ip)

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
               constantType: Optional[Union[TypeAttr, Type]] = None,
               *,
               loc=None,
               ip=None):
    result = pdl.TypeType.get()
    super().__init__(result, constantType=constantType, loc=loc, ip=ip)


class TypesOp:
  """Specialization for PDL types op class."""

  def __init__(
      self,
      constantTypes: Optional[Sequence[Union[TypeAttr, Type]]] = None,
      *,
      loc=None,
      ip=None,
  ):
    if constantTypes is None:
      constantTypes = []
    result = pdl.RangeType.get(pdl.TypeType.get())
    super().__init__(result, constantTypes=constantTypes, loc=loc, ip=ip)
