#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._irdl_ops_gen import *
from ._irdl_ops_gen import _Dialect
from ._irdl_enum_gen import *
from .._mlir_libs._mlirDialectsIRDL import *
from ..ir import register_attribute_builder
from ._ods_common import (
    get_op_result_or_value as _get_value,
    get_op_results_or_values as _get_values,
    _cext as _ods_cext,
)
from ..extras.meta import region_op


@_ods_cext.register_operation(_Dialect, replace=True)
class DialectOp(DialectOp):
    """Specialization for the dialect op class."""

    def __init__(self, sym_name, *, loc=None, ip=None):
        super().__init__(sym_name, loc=loc, ip=ip)
        self.regions[0].blocks.append()

    @property
    def body(self):
        return self.regions[0].blocks[0]


@_ods_cext.register_operation(_Dialect, replace=True)
class OperationOp(OperationOp):
    """Specialization for the operation op class."""

    def __init__(self, sym_name, *, loc=None, ip=None):
        super().__init__(sym_name, loc=loc, ip=ip)
        self.regions[0].blocks.append()

    @property
    def body(self):
        return self.regions[0].blocks[0]


@register_attribute_builder("VariadicityArrayAttr")
def _variadicity_array_attr(x, context):
    return _ods_cext.ir.Attribute.parse(
        f"#irdl<variadicity_array [{', '.join(str(i) for i in x)}]>"
    )
