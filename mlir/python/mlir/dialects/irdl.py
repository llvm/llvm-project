#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._irdl_ops_gen import *
from ._irdl_ops_gen import _Dialect
from ._irdl_enum_gen import *
from .._mlir_libs._mlirDialectsIRDL import *
from ..ir import register_attribute_builder
from ._ods_common import _cext as _ods_cext
from typing import Union, Sequence

_ods_ir = _ods_cext.ir


@_ods_cext.register_operation(_Dialect, replace=True)
class DialectOp(DialectOp):
    __doc__ = DialectOp.__doc__

    def __init__(self, sym_name: Union[str, _ods_ir.Attribute], *, loc=None, ip=None):
        super().__init__(sym_name, loc=loc, ip=ip)
        self.regions[0].blocks.append()

    @property
    def body(self) -> _ods_ir.Block:
        return self.regions[0].blocks[0]


def dialect(sym_name: Union[str, _ods_ir.Attribute], *, loc=None, ip=None) -> DialectOp:
    return DialectOp(sym_name=sym_name, loc=loc, ip=ip)


@_ods_cext.register_operation(_Dialect, replace=True)
class OperationOp(OperationOp):
    __doc__ = OperationOp.__doc__

    def __init__(self, sym_name: Union[str, _ods_ir.Attribute], *, loc=None, ip=None):
        super().__init__(sym_name, loc=loc, ip=ip)
        self.regions[0].blocks.append()

    @property
    def body(self) -> _ods_ir.Block:
        return self.regions[0].blocks[0]


def operation_(
    sym_name: Union[str, _ods_ir.Attribute], *, loc=None, ip=None
) -> OperationOp:
    return OperationOp(sym_name=sym_name, loc=loc, ip=ip)


@_ods_cext.register_operation(_Dialect, replace=True)
class TypeOp(TypeOp):
    __doc__ = TypeOp.__doc__

    def __init__(self, sym_name: Union[str, _ods_ir.Attribute], *, loc=None, ip=None):
        super().__init__(sym_name, loc=loc, ip=ip)
        self.regions[0].blocks.append()

    @property
    def body(self) -> _ods_ir.Block:
        return self.regions[0].blocks[0]


def type_(sym_name: Union[str, _ods_ir.Attribute], *, loc=None, ip=None) -> TypeOp:
    return TypeOp(sym_name=sym_name, loc=loc, ip=ip)


@_ods_cext.register_operation(_Dialect, replace=True)
class AttributeOp(AttributeOp):
    __doc__ = AttributeOp.__doc__

    def __init__(self, sym_name: Union[str, _ods_ir.Attribute], *, loc=None, ip=None):
        super().__init__(sym_name, loc=loc, ip=ip)
        self.regions[0].blocks.append()

    @property
    def body(self) -> _ods_ir.Block:
        return self.regions[0].blocks[0]


def attribute(
    sym_name: Union[str, _ods_ir.Attribute], *, loc=None, ip=None
) -> AttributeOp:
    return AttributeOp(sym_name=sym_name, loc=loc, ip=ip)


@register_attribute_builder("VariadicityArrayAttr")
def _variadicity_array_attr(x: Sequence[Variadicity], context) -> _ods_ir.Attribute:
    return _ods_ir.Attribute.parse(
        f"#irdl<variadicity_array [{', '.join(str(i) for i in x)}]>", context
    )
