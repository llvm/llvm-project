#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._mlir_libs._mlir.ir import *
from ._mlir_libs._mlir.ir import _GlobalDebug


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

@register_attribute_builder("I32Attr")
def _i32Attr(x, context):
  return IntegerAttr.get(
      IntegerType.get_signless(32, context=context), x)

@register_attribute_builder("I64Attr")
def _i64Attr(x, context):
  return IntegerAttr.get(
      IntegerType.get_signless(64, context=context), x)

@register_attribute_builder("StrAttr")
def _stringAttr(x, context):
  return StringAttr.get(x, context=context)

@register_attribute_builder("SymbolNameAttr")
def _symbolNameAttr(x, context):
  return StringAttr.get(x, context=context)

try:
  import numpy as np
  @register_attribute_builder("IndexElementsAttr")
  def _indexElementsAttr(x, context):
    return DenseElementsAttr.get(
        np.array(x, dtype=np.int64), type=IndexType.get(context=context),
        context=context)
except ImportError:
  pass
