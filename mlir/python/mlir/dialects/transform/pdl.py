#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._transform_pdl_extension_ops_gen import *
from .._transform_pdl_extension_ops_gen import _Dialect

try:
    from ...ir import *
    from .._ods_common import (
        get_op_result_or_value as _get_op_result_or_value,
        get_op_results_or_values as _get_op_results_or_values,
        _cext as _ods_cext,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Union


@_ods_cext.register_operation(_Dialect, replace=True)
class PDLMatchOp(PDLMatchOp):
    def __init__(
        self,
        result_type: Type,
        target: Union[Operation, Value],
        pattern_name: Union[Attribute, str],
        *,
        loc=None,
        ip=None,
    ):
        super().__init__(
            result_type,
            _get_op_result_or_value(target),
            pattern_name,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class WithPDLPatternsOp(WithPDLPatternsOp):
    def __init__(self, target: Union[Operation, Value, Type], *, loc=None, ip=None):
        root = _get_op_result_or_value(target) if not isinstance(target, Type) else None
        root_type = target if isinstance(target, Type) else root.type
        super().__init__(root=root, loc=loc, ip=ip)
        self.regions[0].blocks.append(root_type)

    @property
    def body(self) -> Block:
        return self.regions[0].blocks[0]

    @property
    def bodyTarget(self) -> Value:
        return self.body.arguments[0]
