#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._transform_enum_gen import *
from .._transform_ops_gen import *
from .._transform_ops_gen import _Dialect
from ..._mlir_libs._mlirDialectsTransform import *
from ..._mlir_libs._mlirDialectsTransform import AnyOpType, OperationType

try:
    from ...ir import *
    from .._ods_common import (
        get_op_result_or_value as _get_op_result_or_value,
        get_op_results_or_values as _get_op_results_or_values,
        _cext as _ods_cext,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, Sequence, Union, NewType


@_ods_cext.register_operation(_Dialect, replace=True)
class CastOp(CastOp):
    def __init__(
        self,
        result_type: Type,
        target: Union[Operation, Value],
        *,
        loc=None,
        ip=None,
    ):
        super().__init__(result_type, _get_op_result_or_value(target), loc=loc, ip=ip)


@_ods_cext.register_operation(_Dialect, replace=True)
class ApplyPatternsOp(ApplyPatternsOp):
    def __init__(
        self,
        target: Union[Operation, Value, OpView],
        *,
        loc=None,
        ip=None,
    ):
        super().__init__(target, loc=loc, ip=ip)
        self.regions[0].blocks.append()

    @property
    def patterns(self) -> Block:
        return self.regions[0].blocks[0]


@_ods_cext.register_operation(_Dialect, replace=True)
class GetParentOp(GetParentOp):
    def __init__(
        self,
        result_type: Type,
        target: Union[Operation, Value],
        *,
        isolated_from_above: bool = False,
        op_name: Optional[str] = None,
        deduplicate: bool = False,
        nth_parent: int = 1,
        loc=None,
        ip=None,
    ):
        super().__init__(
            result_type,
            _get_op_result_or_value(target),
            isolated_from_above=isolated_from_above,
            op_name=op_name,
            deduplicate=deduplicate,
            nth_parent=nth_parent,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class MergeHandlesOp(MergeHandlesOp):
    def __init__(
        self,
        handles: Sequence[Union[Operation, Value]],
        *,
        deduplicate: bool = False,
        loc=None,
        ip=None,
    ):
        super().__init__(
            [_get_op_result_or_value(h) for h in handles],
            deduplicate=deduplicate,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class ReplicateOp(ReplicateOp):
    def __init__(
        self,
        pattern: Union[Operation, Value],
        handles: Sequence[Union[Operation, Value]],
        *,
        loc=None,
        ip=None,
    ):
        super().__init__(
            [_get_op_result_or_value(h).type for h in handles],
            _get_op_result_or_value(pattern),
            [_get_op_result_or_value(h) for h in handles],
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class SequenceOp(SequenceOp):
    def __init__(
        self,
        failure_propagation_mode,
        results: Sequence[Type],
        target: Union[Operation, Value, Type],
        extra_bindings: Optional[
            Union[Sequence[Value], Sequence[Type], Operation, OpView]
        ] = None,
    ):
        root = (
            _get_op_result_or_value(target)
            if isinstance(target, (Operation, Value))
            else None
        )
        root_type = root.type if not isinstance(target, Type) else target

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

        super().__init__(
            results_=results,
            failure_propagation_mode=failure_propagation_mode,
            root=root,
            extra_bindings=extra_bindings,
        )
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


@_ods_cext.register_operation(_Dialect, replace=True)
class NamedSequenceOp(NamedSequenceOp):
    def __init__(
        self,
        sym_name,
        input_types: Sequence[Type],
        result_types: Sequence[Type],
        sym_visibility=None,
        arg_attrs=None,
        res_attrs=None,
    ):
        function_type = FunctionType.get(input_types, result_types)
        super().__init__(
            sym_name=sym_name,
            function_type=TypeAttr.get(function_type),
            sym_visibility=sym_visibility,
            arg_attrs=arg_attrs,
            res_attrs=res_attrs,
        )
        self.regions[0].blocks.append(*input_types)

    @property
    def body(self) -> Block:
        return self.regions[0].blocks[0]

    @property
    def bodyTarget(self) -> Value:
        return self.body.arguments[0]

    @property
    def bodyExtraArgs(self) -> BlockArgumentList:
        return self.body.arguments[1:]


@_ods_cext.register_operation(_Dialect, replace=True)
class YieldOp(YieldOp):
    def __init__(
        self,
        operands: Optional[Union[Operation, Sequence[Value]]] = None,
        *,
        loc=None,
        ip=None,
    ):
        if operands is None:
            operands = []
        super().__init__(_get_op_results_or_values(operands), loc=loc, ip=ip)


AnyOpTypeT = NewType("AnyOpType", AnyOpType)


def any_op_t() -> AnyOpTypeT:
    return AnyOpTypeT(AnyOpType.get())
