#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from ..ir import *
  from ._ods_common import get_op_result_or_value as _get_op_result_or_value
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, Union


def _get_int64_attr(arg: Optional[Union[int, IntegerAttr]],
                    default_value: int = None):
  if isinstance(arg, IntegerAttr):
    return arg

  if arg is None:
    assert default_value is not None, "must provide default value"
    arg = default_value

  return IntegerAttr.get(IntegerType.get_signless(64), arg)


class GetParentForOp:
  """Extension for GetParentForOp."""

  def __init__(self,
               result_type: Type,
               target: Union[Operation, Value],
               *,
               num_loops: int = 1,
               ip=None,
               loc=None):
    super().__init__(
        result_type,
        _get_op_result_or_value(target),
        num_loops=_get_int64_attr(num_loops, default_value=1),
        ip=ip,
        loc=loc)


class LoopOutlineOp:
  """Extension for LoopOutlineOp."""

  def __init__(self,
               result_type: Type,
               target: Union[Operation, Value],
               *,
               func_name: Union[str, StringAttr],
               ip=None,
               loc=None):
    super().__init__(
        result_type,
        _get_op_result_or_value(target),
        func_name=(func_name if isinstance(func_name, StringAttr) else
                   StringAttr.get(func_name)),
        ip=ip,
        loc=loc)


class LoopPeelOp:
  """Extension for LoopPeelOp."""

  def __init__(self,
               result_type: Type,
               target: Union[Operation, Value],
               *,
               fail_if_already_divisible: Union[bool, BoolAttr] = False,
               ip=None,
               loc=None):
    super().__init__(
        result_type,
        _get_op_result_or_value(target),
        fail_if_already_divisible=(fail_if_already_divisible if isinstance(
            fail_if_already_divisible, BoolAttr) else
                                   BoolAttr.get(fail_if_already_divisible)),
        ip=ip,
        loc=loc)


class LoopPipelineOp:
  """Extension for LoopPipelineOp."""

  def __init__(self,
               result_type: Type,
               target: Union[Operation, Value],
               *,
               iteration_interval: Optional[Union[int, IntegerAttr]] = None,
               read_latency: Optional[Union[int, IntegerAttr]] = None,
               ip=None,
               loc=None):
    super().__init__(
        result_type,
        _get_op_result_or_value(target),
        iteration_interval=_get_int64_attr(iteration_interval, default_value=1),
        read_latency=_get_int64_attr(read_latency, default_value=10),
        ip=ip,
        loc=loc)


class LoopUnrollOp:
  """Extension for LoopUnrollOp."""

  def __init__(self,
               target: Union[Operation, Value],
               *,
               factor: Union[int, IntegerAttr],
               ip=None,
               loc=None):
    super().__init__(
        _get_op_result_or_value(target),
        factor=_get_int64_attr(factor),
        ip=ip,
        loc=loc)
