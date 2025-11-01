#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._xegpu_transform_ops_gen import *
from .._xegpu_transform_ops_gen import _Dialect

try:
    from ...ir import *
    from .._ods_common import _cext as _ods_cext
    from .._ods_common import (
        MixedValues,
        get_op_result_or_value as _get_op_result_or_value,
        _dispatch_dynamic_index_list,
    )

except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Union


@_ods_cext.register_operation(_Dialect, replace=True)
class SetDescLayoutOp(SetDescLayoutOp):
    """Specialization for SetDescLayoutOp class."""

    def __init__(
        self,
        target: Union[Operation, Value],
        sg_layout: MixedValues,
        *,
        sg_data: MixedValues = None,
        inst_data: MixedValues = None,
        loc=None,
        ip=None,
    ):
        target_value = _get_op_result_or_value(target)
        sg_data = [] if sg_data is None else sg_data
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
            target_value.type,
            target_value,
            dynamic_sg_layout,
            dynamic_sg_data,
            dynamic_inst_data,
            static_sg_layout=static_sg_layout,
            static_sg_data=static_sg_data,
            static_inst_data=static_inst_data,
            loc=loc,
            ip=ip,
        )
