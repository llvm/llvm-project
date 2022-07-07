#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from ..ir import *
  from ._ods_common import get_op_result_or_value as _get_op_result_or_value, get_op_results_or_values as _get_op_results_or_values
  from ..dialects import pdl
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, overload, Sequence, Union


def _get_symbol_ref_attr(value: Union[Attribute, str]):
  if isinstance(value, Attribute):
    return value
  return FlatSymbolRefAttr.get(value)


class GetClosestIsolatedParentOp:

  def __init__(self, target: Union[Operation, Value], *, loc=None, ip=None):
    super().__init__(
        pdl.OperationType.get(),
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
        pdl.OperationType.get(), [_get_op_result_or_value(h) for h in handles],
        deduplicate=deduplicate,
        loc=loc,
        ip=ip)


class PDLMatchOp:

  def __init__(self,
               target: Union[Operation, Value],
               pattern_name: Union[Attribute, str],
               *,
               loc=None,
               ip=None):
    super().__init__(
        pdl.OperationType.get(),
        _get_op_result_or_value(target),
        _get_symbol_ref_attr(pattern_name),
        loc=loc,
        ip=ip)


class SequenceOp:

  @overload
  def __init__(self, resultsOrRoot: Sequence[Type],
               optionalRoot: Optional[Union[Operation, Value]]):
    ...

  @overload
  def __init__(self, resultsOrRoot: Optional[Union[Operation, Value]],
               optionalRoot: NoneType):
    ...

  def __init__(self, resultsOrRoot=None, optionalRoot=None):
    results = resultsOrRoot if isinstance(resultsOrRoot, Sequence) else []
    root = (
        resultsOrRoot
        if not isinstance(resultsOrRoot, Sequence) else optionalRoot)
    root = _get_op_result_or_value(root) if root else None
    super().__init__(results_=results, root=root)
    self.regions[0].blocks.append(pdl.OperationType.get())

  @property
  def body(self) -> Block:
    return self.regions[0].blocks[0]

  @property
  def bodyTarget(self) -> Value:
    return self.body.arguments[0]


class WithPDLPatternsOp:

  def __init__(self,
               target: Optional[Union[Operation, Value]] = None,
               *,
               loc=None,
               ip=None):
    super().__init__(
        root=_get_op_result_or_value(target) if target else None,
        loc=loc,
        ip=ip)
    self.regions[0].blocks.append(pdl.OperationType.get())

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
