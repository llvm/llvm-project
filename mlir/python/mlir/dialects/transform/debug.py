#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

from ...ir import Attribute, Operation, Value, StringAttr
from .._transform_debug_extension_ops_gen import *
from .._transform_pdl_extension_ops_gen import _Dialect

try:
    from .._ods_common import _cext as _ods_cext
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Union


@_ods_cext.register_operation(_Dialect, replace=True)
class EmitParamAsRemarkOp(EmitParamAsRemarkOp):
    def __init__(
        self,
        param: Attribute,
        *,
        anchor: Optional[Operation] = None,
        message: Optional[Union[StringAttr, str]] = None,
        loc=None,
        ip=None,
    ):
        if isinstance(message, str):
            message = StringAttr.get(message)

        super().__init__(
            param,
            anchor=anchor,
            message=message,
            loc=loc,
            ip=ip,
        )


def emit_param_as_remark(
    param: Attribute,
    *,
    anchor: Optional[Operation] = None,
    message: Optional[Union[StringAttr, str]] = None,
    loc=None,
    ip=None,
):
    return EmitParamAsRemarkOp(param, anchor=anchor, message=message, loc=loc, ip=ip)


@_ods_cext.register_operation(_Dialect, replace=True)
class EmitRemarkAtOp(EmitRemarkAtOp):
    def __init__(
        self,
        at: Union[Operation, Value],
        message: Optional[Union[StringAttr, str]] = None,
        *,
        loc=None,
        ip=None,
    ):
        if isinstance(message, str):
            message = StringAttr.get(message)

        super().__init__(
            at,
            message,
            loc=loc,
            ip=ip,
        )


def emit_remark_at(
    at: Union[Operation, Value],
    message: Optional[Union[StringAttr, str]] = None,
    *,
    loc=None,
    ip=None,
):
    return EmitRemarkAtOp(at, message, loc=loc, ip=ip)
