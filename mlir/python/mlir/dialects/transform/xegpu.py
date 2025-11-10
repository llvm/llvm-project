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
