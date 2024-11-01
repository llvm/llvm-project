#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from ..ir import *
  from ._ods_common import get_op_result_or_value as _get_op_result_or_value, get_op_results_or_values as _get_op_results_or_values
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e

from argparse import SUPPRESS
from typing import Optional, overload, Sequence, Union


def _get_symbol_ref_attr(value: Union[Attribute, str]):
  if isinstance(value, Attribute):
    return value
  return FlatSymbolRefAttr.get(value)


class CastOp:

  def __init__(self, result_type: Type, target: Union[Operation, Value], *, loc=None, ip=None):
    super().__init__(
      result_type,
      _get_op_result_or_value(target),
      loc=loc,
      ip=ip)


class GetClosestIsolatedParentOp:

  def __init__(self, result_type: Type, target: Union[Operation, Value], *, loc=None, ip=None):
    super().__init__(
        result_type,
        _get_op_result_or_value(target),
        loc=loc,
        ip=ip)


class MergeHandlesOp:

  def __init__(self,
               handles: Sequence[Union[Operation, Value]],
               *,
               deduplicate: bool = False,
               loc=None,
               ip=None):
    super().__init__(
        [_get_op_result_or_value(h) for h in handles],
        deduplicate=deduplicate,
        loc=loc,
        ip=ip)


class PDLMatchOp:

  def __init__(self,
               result_type: Type,
               target: Union[Operation, Value],
               pattern_name: Union[Attribute, str],
               *,
               loc=None,
               ip=None):
    super().__init__(
        result_type,
        _get_op_result_or_value(target),
        _get_symbol_ref_attr(pattern_name),
        loc=loc,
        ip=ip)


class ReplicateOp:

  def __init__(self,
               pattern: Union[Operation, Value],
               handles: Sequence[Union[Operation, Value]],
               *,
               loc=None,
               ip=None):
    super().__init__(
        [_get_op_result_or_value(h).type for h in handles],
        _get_op_result_or_value(pattern),
        [_get_op_result_or_value(h) for h in handles],
        loc=loc,
        ip=ip)


class SequenceOp:

  def __init__(self, failure_propagation_mode, results: Sequence[Type],
               target: Union[Operation, Value, Type],
               extra_bindings: Optional[Union[Sequence[Value], Sequence[Type],
                                              Operation, OpView]] = None):
    root = _get_op_result_or_value(target) if isinstance(
        target, (Operation, Value)) else None
    root_type = root.type if not isinstance(target, Type) else target
    if not isinstance(failure_propagation_mode, Attribute):
      failure_propagation_mode_attr = IntegerAttr.get(
          IntegerType.get_signless(32), failure_propagation_mode._as_int())
    else:
      failure_propagation_mode = failure_propagation_mode

    if extra_bindings is None:
      extra_bindings = []
    if isinstance(extra_bindings, (Operation, OpView)):
      extra_bindings = _get_op_results_or_values(extra_bindings)

    extra_binding_types = []
    if len(extra_bindings) != 0:
      if isinstance(extra_bindings[0], Type):
        extra_binding_types = extra_bindings
        extra_bindings = []
      else:
        extra_binding_types = [v.type for v in extra_bindings]

    super().__init__(results_=results,
                     failure_propagation_mode=failure_propagation_mode_attr,
                     root=root,
                     extra_bindings=extra_bindings)
    self.regions[0].blocks.append(*tuple([root_type] + extra_binding_types))

  @property
  def body(self) -> Block:
    return self.regions[0].blocks[0]

  @property
  def bodyTarget(self) -> Value:
    return self.body.arguments[0]

  @property
  def bodyExtraArgs(self) -> BlockArgumentList:
    return self.body.arguments[1:]


class WithPDLPatternsOp:

  def __init__(self,
               target: Union[Operation, Value, Type],
               *,
               loc=None,
               ip=None):
    root = _get_op_result_or_value(target) if not isinstance(target,
                                                             Type) else None
    root_type = target if isinstance(target, Type) else root.type
    super().__init__(
        root=root,
        loc=loc,
        ip=ip)
    self.regions[0].blocks.append(root_type)

  @property
  def body(self) -> Block:
    return self.regions[0].blocks[0]

  @property
  def bodyTarget(self) -> Value:
    return self.body.arguments[0]


class YieldOp:

  def __init__(self,
               operands: Union[Operation, Sequence[Value]] = [],
               *,
               loc=None,
               ip=None):
    super().__init__(_get_op_results_or_values(operands), loc=loc, ip=ip)
