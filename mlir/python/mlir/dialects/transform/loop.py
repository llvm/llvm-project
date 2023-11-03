#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._loop_transform_ops_gen import *
from .._loop_transform_ops_gen import _Dialect

try:
    from ...ir import *
    from .._ods_common import (
        get_op_result_or_value as _get_op_result_or_value,
        _cext as _ods_cext,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, Union


@_ods_cext.register_operation(_Dialect, replace=True)
class LoopOutlineOp(LoopOutlineOp):
    """Extension for LoopOutlineOp."""

    def __init__(
        self,
        function_type: Type,
        call_type: Type,
        target: Union[Operation, Value],
        *,
        func_name: Union[str, StringAttr],
        ip=None,
        loc=None,
    ):
        super().__init__(
            function_type,
            call_type,
            _get_op_result_or_value(target),
            func_name=(
                func_name
                if isinstance(func_name, StringAttr)
                else StringAttr.get(func_name)
            ),
            ip=ip,
            loc=loc,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class LoopPeelOp(LoopPeelOp):
    """Extension for LoopPeelOp."""

    def __init__(
        self,
        main_loop_type: Type,
        remainder_loop_type: Type,
        target: Union[Operation, Value],
        *,
        fail_if_already_divisible: Union[bool, BoolAttr] = False,
        ip=None,
        loc=None,
    ):
        super().__init__(
            main_loop_type,
            remainder_loop_type,
            _get_op_result_or_value(target),
            fail_if_already_divisible=(
                fail_if_already_divisible
                if isinstance(fail_if_already_divisible, BoolAttr)
                else BoolAttr.get(fail_if_already_divisible)
            ),
            ip=ip,
            loc=loc,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class LoopPipelineOp(LoopPipelineOp):
    """Extension for LoopPipelineOp."""

    def __init__(
        self,
        result_type: Type,
        target: Union[Operation, Value],
        *,
        iteration_interval: Optional[Union[int, IntegerAttr]] = None,
        read_latency: Optional[Union[int, IntegerAttr]] = None,
        ip=None,
        loc=None,
    ):
        if iteration_interval is None:
            iteration_interval = 1
        if read_latency is None:
            read_latency = 10
        super().__init__(
            result_type,
            _get_op_result_or_value(target),
            iteration_interval=iteration_interval,
            read_latency=read_latency,
            ip=ip,
            loc=loc,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class LoopUnrollOp(LoopUnrollOp):
    """Extension for LoopUnrollOp."""

    def __init__(
        self,
        target: Union[Operation, Value],
        *,
        factor: Union[int, IntegerAttr],
        ip=None,
        loc=None,
    ):
        super().__init__(
            _get_op_result_or_value(target),
            factor=factor,
            ip=ip,
            loc=loc,
        )
