#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._tensor_transform_ops_gen import *
from .._tensor_transform_ops_gen import _Dialect

try:
    from ...ir import *
    from ...dialects import transform
    from .._ods_common import _cext as _ods_cext
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, overload, Union


@_ods_cext.register_operation(_Dialect, replace=True)
class MakeLoopIndependentOp(MakeLoopIndependentOp):
    """Specialization for MakeLoopIndependentOp class."""

    @overload
    def __init__(
        self,
        transformed_type: Type,
        target: Union[Operation, OpView, Value],
        num_loops: Union[int, IntegerAttr],
        *,
        loc=None,
        ip=None,
    ):
        ...

    @overload
    def __init__(
        self,
        target: Union[Operation, OpView, Value],
        num_loops: Union[int, IntegerAttr],
        *,
        loc=None,
        ip=None,
    ):
        ...

    def __init__(
        self,
        transformed_type_or_target: Type,
        target_or_num_loops: Union[int, IntegerAttr, Operation, OpView, Value] = None,
        num_loops_or_none: Optional[Union[int, IntegerAttr]] = None,
        *,
        loc=None,
        ip=None,
    ):
        if isinstance(transformed_type_or_target, Type):
            transformed_type = transformed_type_or_target
            target = target_or_num_loops
            num_loops = num_loops_or_none
        else:
            transformed_type = transform.AnyOpType.get()
            target = transformed_type_or_target
            num_loops = target_or_num_loops

        super().__init__(
            transformed_type,
            target,
            num_loops,
            loc=loc,
            ip=ip,
        )
