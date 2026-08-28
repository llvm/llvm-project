#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._llvm_ops_gen import *
from ._llvm_ops_gen import _Dialect
from ._llvm_enum_gen import *
from .._mlir_libs._mlirDialectsLLVM import *
from ..ir import Value, IntegerType, IntegerAttr
from ._ods_common import get_op_result_or_op_results as _get_op_result_or_op_results


def mlir_constant(value, *, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(
        ConstantOp(res=value.type, value=value, loc=loc, ip=ip)
    )


def md_const(val, *, width=32, context=None):
    if not isinstance(val, int):
        raise NotImplementedError(
            f"{val=} not supported; only integers currently supported."
        )
    i_type = IntegerType.get_signless(width, context=context)
    return MDConstantAttr.get(IntegerAttr.get(i_type, val), context=context)


def md_str(s, *, context=None):
    return MDStringAttr.get(s, context=context)
