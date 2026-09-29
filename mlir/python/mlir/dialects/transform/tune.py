#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Sequence

from ...ir import (
    Type,
    Value,
    Operation,
    OpView,
    Attribute,
    ArrayAttr,
    StringAttr,
    F64Type,
    IntegerType,
    IntegerAttr,
    FloatAttr,
    BoolAttr,
)
from .._transform_tune_extension_ops_gen import *
from .._transform_tune_extension_ops_gen import _Dialect

try:
    from .._ods_common import (
        get_op_result_or_value as _get_op_result_or_value,
        _cext as _ods_cext,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Union


@_ods_cext.register_operation(_Dialect, replace=True)
class KnobOp(KnobOp):
    def __init__(
        self,
        result: Type,  # !transform.any_param or !transform.param<Type>
        name: Union[StringAttr, str],
        options: Union[
            ArrayAttr, Sequence[Union[Attribute, bool, int, float, str]], Attribute
        ],
        *,
        selected: Optional[Union[Attribute, bool, int, float, str]] = None,
        loc=None,
        ip=None,
    ):
        if isinstance(name, str):
            name = StringAttr.get(name)

        def map_to_attr(value):
            if isinstance(value, bool):
                return BoolAttr.get(value)
            if isinstance(value, int):
                return IntegerAttr.get(IntegerType.get_signless(64), value)
            if isinstance(value, float):
                return FloatAttr.get(F64Type.get(), value)
            if isinstance(value, str):
                return StringAttr.get(value)
            assert isinstance(value, Attribute)
            return value

        if isinstance(options, Sequence) and not isinstance(options, ArrayAttr):
            options = ArrayAttr.get([map_to_attr(opt) for opt in options])

        super().__init__(
            result,
            name,
            options,
            selected=selected and map_to_attr(selected),
            loc=loc,
            ip=ip,
        )


def knob(
    result: Type,  # !transform.any_param or !transform.param<Type>
    name: Union[StringAttr, str],
    options: Union[
        ArrayAttr, Sequence[Union[Attribute, bool, int, float, str]], Attribute
    ],
    *,
    selected: Optional[Union[Attribute, bool, int, float, str]] = None,
    loc=None,
    ip=None,
):
    return KnobOp(result, name, options, selected=selected, loc=loc, ip=ip)


@_ods_cext.register_operation(_Dialect, replace=True)
class AlternativesOp(AlternativesOp):
    def __init__(
        self,
        results: Sequence[Type],
        name: Union[StringAttr, str],
        num_alternatives: int,
        *,
        selected_region: Optional[
            Union[int, IntegerAttr, Value, Operation, OpView]
        ] = None,
        loc=None,
        ip=None,
    ):
        if isinstance(name, str):
            name = StringAttr.get(name)

        selected_region_attr = selected_region_param = None
        if isinstance(selected_region, IntegerAttr):
            selected_region_attr = selected_region
        elif isinstance(selected_region, int):
            selected_region_attr = IntegerAttr.get(
                IntegerType.get_signless(32), selected_region
            )
        elif isinstance(selected_region, (Value, Operation, OpView)):
            selected_region_param = _get_op_result_or_value(selected_region)

        super().__init__(
            results,
            name,
            num_alternatives,
            selected_region_attr=selected_region_attr,
            selected_region_param=selected_region_param,
            loc=loc,
            ip=ip,
        )
        for region in self.regions:
            region.blocks.append()


def alternatives(
    results: Sequence[Type],
    name: Union[StringAttr, str],
    num_alternatives: int,
    *,
    selected_region: Optional[Union[int, IntegerAttr, Value, Operation, OpView]] = None,
    loc=None,
    ip=None,
):
    return AlternativesOp(
        results, name, num_alternatives, selected_region=selected_region, loc=loc, ip=ip
    )
