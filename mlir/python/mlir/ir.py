#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._mlir_libs._mlir.ir import *
from ._mlir_libs._mlir.ir import _GlobalDebug
from ._mlir_libs._mlir import register_type_caster


# Convenience decorator for registering user-friendly Attribute builders.
def register_attribute_builder(kind):
    def decorator_builder(func):
        AttrBuilder.insert(kind, func)
        return func

    return decorator_builder


@register_attribute_builder("BoolAttr")
def _boolAttr(x, context):
    return BoolAttr.get(x, context=context)


@register_attribute_builder("IndexAttr")
def _indexAttr(x, context):
    return IntegerAttr.get(IndexType.get(context=context), x)


@register_attribute_builder("I16Attr")
def _i16Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(16, context=context), x)


@register_attribute_builder("I32Attr")
def _i32Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), x)


@register_attribute_builder("I64Attr")
def _i64Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), x)


@register_attribute_builder("SI16Attr")
def _si16Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signed(16, context=context), x)


@register_attribute_builder("SI32Attr")
def _si32Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signed(32, context=context), x)


@register_attribute_builder("F32Attr")
def _f32Attr(x, context):
    return FloatAttr.get_f32(x, context=context)


@register_attribute_builder("F64Attr")
def _f64Attr(x, context):
    return FloatAttr.get_f64(x, context=context)


@register_attribute_builder("StrAttr")
def _stringAttr(x, context):
    return StringAttr.get(x, context=context)


@register_attribute_builder("SymbolNameAttr")
def _symbolNameAttr(x, context):
    return StringAttr.get(x, context=context)


@register_attribute_builder("SymbolRefAttr")
def _symbolRefAttr(x, context):
    if isinstance(x, list):
        return SymbolRefAttr.get(x, context=context)
    else:
        return FlatSymbolRefAttr.get(x, context=context)


@register_attribute_builder("FlatSymbolRefAttr")
def _flatSymbolRefAttr(x, context):
    return FlatSymbolRefAttr.get(x, context=context)


@register_attribute_builder("ArrayAttr")
def _arrayAttr(x, context):
    return ArrayAttr.get(x, context=context)


@register_attribute_builder("I32ArrayAttr")
def _i32ArrayAttr(x, context):
    return ArrayAttr.get([_i32Attr(v, context) for v in x])


@register_attribute_builder("I64ArrayAttr")
def _i64ArrayAttr(x, context):
    return ArrayAttr.get([_i64Attr(v, context) for v in x])


@register_attribute_builder("F32ArrayAttr")
def _f32ArrayAttr(x, context):
    return ArrayAttr.get([_f32Attr(v, context) for v in x])


@register_attribute_builder("F64ArrayAttr")
def _f64ArrayAttr(x, context):
    return ArrayAttr.get([_f64Attr(v, context) for v in x])


@register_attribute_builder("DenseI64ArrayAttr")
def _denseI64ArrayAttr(x, context):
    return DenseI64ArrayAttr.get(x, context=context)


@register_attribute_builder("DenseBoolArrayAttr")
def _denseBoolArrayAttr(x, context):
    return DenseBoolArrayAttr.get(x, context=context)


@register_attribute_builder("TypeAttr")
def _typeAttr(x, context):
    return TypeAttr.get(x, context=context)


@register_attribute_builder("TypeArrayAttr")
def _typeArrayAttr(x, context):
    return _arrayAttr([TypeAttr.get(t, context=context) for t in x], context)


try:
    import numpy as np

    @register_attribute_builder("IndexElementsAttr")
    def _indexElementsAttr(x, context):
        return DenseElementsAttr.get(
            np.array(x, dtype=np.int64),
            type=IndexType.get(context=context),
            context=context,
        )

except ImportError:
    pass
