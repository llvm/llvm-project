#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._xegpu_transform_ops_gen import *
from .._xegpu_transform_ops_gen import _Dialect

try:
    from ...ir import *
    from ...dialects import transform
    from .._ods_common import _cext as _ods_cext
    from .._ods_common import (
        MixedValues,
        MixedInt,
        get_op_result_or_value as _get_op_result_or_value,
        _dispatch_dynamic_index_list,
    )

except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Union, Optional


@_ods_cext.register_operation(_Dialect, replace=True)
class GetDescOp(GetDescOp):
    """Specialization for GetDescOp class."""

    def __init__(
        self,
        target: Value,
        *,
        loc=None,
        ip=None,
    ):
        desc_type = transform.AnyOpType.get()
        super().__init__(
            desc_type,
            target,
            loc=loc,
            ip=ip,
        )


def get_desc_op(
    target: Value,
    *,
    loc=None,
    ip=None,
) -> OpResult:
    return GetDescOp(target, loc=loc, ip=ip).result


@_ods_cext.register_operation(_Dialect, replace=True)
class SetDescLayoutOp(SetDescLayoutOp):
    """Specialization for SetDescLayoutOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        sg_layout: MixedValues,
        sg_data: MixedValues,
        *,
        inst_data: Optional[MixedValues] = None,
        loc=None,
        ip=None,
    ):
        target_handle = _get_op_result_or_value(target)
        inst_data = [] if inst_data is None else inst_data
        (
            dynamic_sg_layout,
            static_sg_layout,
            _,
        ) = _dispatch_dynamic_index_list(sg_layout)
        (
            dynamic_sg_data,
            static_sg_data,
            _,
        ) = _dispatch_dynamic_index_list(sg_data)
        (
            dynamic_inst_data,
            static_inst_data,
            _,
        ) = _dispatch_dynamic_index_list(inst_data)

        super().__init__(
            target_handle.type,
            target_handle,
            dynamic_sg_layout,
            dynamic_sg_data,
            dynamic_inst_data,
            static_sg_layout=static_sg_layout,
            static_sg_data=static_sg_data,
            static_inst_data=static_inst_data,
            loc=loc,
            ip=ip,
        )


def set_desc_layout(
    target: Union[Operation, Value],
    sg_layout: MixedValues,
    sg_data: MixedValues,
    *,
    inst_data: Optional[MixedValues] = None,
    loc=None,
    ip=None,
) -> OpResult:
    return SetDescLayoutOp(
        target,
        sg_layout,
        sg_data,
        inst_data=inst_data,
        loc=loc,
        ip=ip,
    ).result


@_ods_cext.register_operation(_Dialect, replace=True)
class SetOpLayoutAttrOp(SetOpLayoutAttrOp):
    """Specialization for SetOpLayoutAttrOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        sg_layout: MixedValues,
        sg_data: MixedValues,
        *,
        inst_data: Optional[MixedValues] = None,
        index: Optional[Union[int, Attribute]] = None,
        result: Optional[Union[bool, Attribute]] = None,
        loc=None,
        ip=None,
    ):
        inst_data = [] if inst_data is None else inst_data
        (
            dynamic_sg_layout,
            static_sg_layout,
            _,
        ) = _dispatch_dynamic_index_list(sg_layout)
        (
            dynamic_sg_data,
            static_sg_data,
            _,
        ) = _dispatch_dynamic_index_list(sg_data)
        (
            dynamic_inst_data,
            static_inst_data,
            _,
        ) = _dispatch_dynamic_index_list(inst_data)
        super().__init__(
            _get_op_result_or_value(target),
            dynamic_sg_layout,
            dynamic_sg_data,
            dynamic_inst_data,
            static_sg_layout=static_sg_layout,
            static_sg_data=static_sg_data,
            static_inst_data=static_inst_data,
            index=index,
            result=result,
            loc=loc,
            ip=ip,
        )


def set_op_layout_attr(
    target: Union[Operation, Value],
    sg_layout: MixedValues,
    sg_data: MixedValues,
    *,
    inst_data: Optional[MixedValues] = None,
    index: Optional[Union[int, Attribute]] = None,
    result: Optional[Union[bool, Attribute]] = None,
    loc=None,
    ip=None,
) -> SetOpLayoutAttrOp:
    return SetOpLayoutAttrOp(
        target,
        sg_layout,
        sg_data,
        inst_data=inst_data,
        index=index,
        result=result,
        loc=loc,
        ip=ip,
    )


@_ods_cext.register_operation(_Dialect, replace=True)
class SetGPULaunchThreadsOp(SetGPULaunchThreadsOp):
    """Specialization for SetGPULaunchThreadsOp class."""

    def __init__(
        self,
        launch_op: Union[Operation, Value],
        threads: MixedValues,
        *,
        loc=None,
        ip=None,
    ):
        (
            dynamic_threads,
            static_threads,
            _,
        ) = _dispatch_dynamic_index_list(threads)

        super().__init__(
            _get_op_result_or_value(launch_op),
            dynamic_threads,
            static_threads=static_threads,
            loc=loc,
            ip=ip,
        )


def set_gpu_launch_threads(
    launch_op: Union[Operation, Value],
    threads: MixedValues,
    *,
    loc=None,
    ip=None,
) -> SetGPULaunchThreadsOp:
    return SetGPULaunchThreadsOp(launch_op, threads, loc=loc, ip=ip)


@_ods_cext.register_operation(_Dialect, replace=True)
class InsertPrefetchOp(InsertPrefetchOp):
    """Specialization for InsertPrefetchOp class."""

    def __init__(
        self,
        target: Value,
        *,
        nb_prefetch: Optional[MixedInt] = 1,
        loc=None,
        ip=None,
    ):
        static_nb_prefetch = 1
        dynamic_nb_prefetch = None
        if isinstance(nb_prefetch, int):
            static_nb_prefetch = nb_prefetch
        elif isinstance(nb_prefetch, IntegerAttr):
            static_nb_prefetch = nb_prefetch.value  # pytype: disable=attribute-error
        elif isinstance(nb_prefetch, (Operation, Value, OpView)):
            dynamic_nb_prefetch = nb_prefetch

        super().__init__(
            transform.AnyOpType.get(),
            target,
            dynamic_nb_prefetch=dynamic_nb_prefetch,
            static_nb_prefetch=static_nb_prefetch,
            loc=loc,
            ip=ip,
        )


def insert_prefetch(
    target: Value,
    *,
    nb_prefetch: Optional[MixedInt] = 1,
    loc=None,
    ip=None,
) -> OpResult:
    return InsertPrefetchOp(target, nb_prefetch=nb_prefetch, loc=loc, ip=ip).result


@_ods_cext.register_operation(_Dialect, replace=True)
class ConvertLayoutOp(ConvertLayoutOp):
    """Specialization for ConvertLayoutOp class."""

    def __init__(
        self,
        target: Value,
        input_sg_layout: MixedValues,
        input_sg_data: MixedValues,
        target_sg_layout: MixedValues,
        target_sg_data: MixedValues,
        *,
        input_inst_data: Optional[MixedValues] = None,
        target_inst_data: Optional[MixedValues] = None,
        loc=None,
        ip=None,
    ):
        input_inst_data = [] if input_inst_data is None else input_inst_data
        target_inst_data = [] if target_inst_data is None else target_inst_data
        (
            dynamic_input_sg_layout,
            static_input_sg_layout,
            _,
        ) = _dispatch_dynamic_index_list(input_sg_layout)
        (
            dynamic_input_sg_data,
            static_input_sg_data,
            _,
        ) = _dispatch_dynamic_index_list(input_sg_data)
        (
            dynamic_input_inst_data,
            static_input_inst_data,
            _,
        ) = _dispatch_dynamic_index_list(input_inst_data)
        (
            dynamic_target_sg_layout,
            static_target_sg_layout,
            _,
        ) = _dispatch_dynamic_index_list(target_sg_layout)
        (
            dynamic_target_sg_data,
            static_target_sg_data,
            _,
        ) = _dispatch_dynamic_index_list(target_sg_data)
        (
            dynamic_target_inst_data,
            static_target_inst_data,
            _,
        ) = _dispatch_dynamic_index_list(target_inst_data)
        super().__init__(
            transform.AnyOpType.get(),
            target,
            dynamic_input_sg_layout,
            dynamic_input_sg_data,
            dynamic_input_inst_data,
            dynamic_target_sg_layout,
            dynamic_target_sg_data,
            dynamic_target_inst_data,
            static_input_sg_layout=static_input_sg_layout,
            static_input_sg_data=static_input_sg_data,
            static_input_inst_data=static_input_inst_data,
            static_target_sg_layout=static_target_sg_layout,
            static_target_sg_data=static_target_sg_data,
            static_target_inst_data=static_target_inst_data,
            loc=loc,
            ip=ip,
        )


def convert_layout(
    target: Value,
    input_sg_layout: MixedValues,
    input_sg_data: MixedValues,
    target_sg_layout: MixedValues,
    target_sg_data: MixedValues,
    *,
    input_inst_data: Optional[MixedValues] = None,
    target_inst_data: Optional[MixedValues] = None,
    loc=None,
    ip=None,
) -> ConvertLayoutOp:
    return ConvertLayoutOp(
        target,
        input_sg_layout,
        input_sg_data,
        target_sg_layout,
        target_sg_data,
        input_inst_data=input_inst_data,
        target_inst_data=target_inst_data,
        loc=loc,
        ip=ip,
    ).result
