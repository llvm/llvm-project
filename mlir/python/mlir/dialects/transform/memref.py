#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._memref_transform_ops_gen import *
from .._memref_transform_ops_gen import _Dialect

try:
    from ...ir import *
    from ...dialects import transform
    from .._ods_common import _cext as _ods_cext
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, overload, Union


@_ods_cext.register_operation(_Dialect, replace=True)
class MemRefAllocaToGlobalOp(MemRefAllocaToGlobalOp):
    """Specialization for MemRefAllocaToGlobalOp class."""

    @overload
    def __init__(
        self,
        get_global_type: Type,
        global_type: Type,
        alloca: Union[Operation, OpView, Value],
        *,
        loc=None,
        ip=None,
    ):
        ...

    @overload
    def __init__(self, alloca: Union[Operation, OpView, Value], *, loc=None, ip=None):
        ...

    def __init__(
        self,
        get_global_type_or_alloca: Union[Operation, OpView, Type, Value],
        global_type_or_none: Optional[Type] = None,
        alloca_or_none: Optional[Union[Operation, OpView, Value]] = None,
        *,
        loc=None,
        ip=None,
    ):
        if isinstance(get_global_type_or_alloca, Type):
            get_global_type = get_global_type_or_alloca
            global_type = global_type_or_none
            alloca = alloca_or_none
        else:
            get_global_type = transform.AnyOpType.get()
            global_type = transform.AnyOpType.get()
            alloca = get_global_type_or_alloca

        super().__init__(
            get_global_type,
            global_type,
            alloca,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class MemRefMultiBufferOp(MemRefMultiBufferOp):
    """Specialization for MemRefMultiBufferOp class."""

    @overload
    def __init__(
        self,
        transformed_type: Type,
        target: Union[Operation, OpView, Value],
        factor: Union[int, IntegerAttr],
        *,
        skip_analysis: Optional[bool] = None,
        loc=None,
        ip=None,
    ):
        ...

    @overload
    def __init__(
        self,
        target: Union[Operation, OpView, Value],
        factor: Union[int, IntegerAttr],
        *,
        skip_analysis: Optional[bool] = None,
        loc=None,
        ip=None,
    ):
        ...

    def __init__(
        self,
        transformed_type_or_target: Type,
        target_or_factor: Union[int, IntegerAttr, Operation, OpView, Value] = None,
        factor_or_none: Optional[Union[int, IntegerAttr]] = None,
        *,
        skip_analysis: Optional[bool] = None,
        loc=None,
        ip=None,
    ):
        if isinstance(transformed_type_or_target, Type):
            transformed_type = transformed_type_or_target
            target = target_or_factor
            factor = factor_or_none
        else:
            transformed_type = transform.AnyOpType.get()
            target = transformed_type_or_target
            factor = target_or_factor

        super().__init__(
            transformed_type,
            target,
            factor,
            skip_analysis=skip_analysis,
            loc=loc,
            ip=ip,
        )
