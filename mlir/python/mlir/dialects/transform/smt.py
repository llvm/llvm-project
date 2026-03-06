#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence

from ...ir import Type, Block
from .._transform_smt_extension_ops_gen import *
from .._transform_smt_extension_ops_gen import _Dialect
from ...dialects import transform

try:
    from .._ods_common import _cext as _ods_cext
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


@_ods_cext.register_operation(_Dialect, replace=True)
class ConstrainParamsOp(ConstrainParamsOp):
    def __init__(
        self,
        results: Sequence[Type],
        params: Sequence[transform.AnyParamType],
        arg_types: Sequence[Type],
        loc=None,
        ip=None,
    ):
        if len(params) != len(arg_types):
            raise ValueError(f"{params=} not same length as {arg_types=}")
        super().__init__(
            results,
            params,
            loc=loc,
            ip=ip,
        )
        self.regions[0].blocks.append(*arg_types)

    @property
    def body(self) -> Block:
        return self.regions[0].blocks[0]


def constrain_params(
    results: Sequence[Type],
    params: Sequence[transform.AnyParamType],
    arg_types: Sequence[Type],
    loc=None,
    ip=None,
):
    return ConstrainParamsOp(results, params, arg_types, loc=loc, ip=ip)
