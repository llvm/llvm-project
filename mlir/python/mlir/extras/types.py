#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import partial
from typing import Optional, List

from ..ir import (
    Attribute,
    BF16Type,
    ComplexType,
    F16Type,
    F32Type,
    F64Type,
    Float8E4M3B11FNUZType,
    Float8E4M3FNType,
    Float8E5M2Type,
    FunctionType,
    IndexType,
    IntegerType,
    MemRefType,
    NoneType,
    OpaqueType,
    RankedTensorType,
    StridedLayoutAttr,
    StringAttr,
    TupleType,
    Type,
    UnrankedMemRefType,
    UnrankedTensorType,
    VectorType,
)

index = lambda: IndexType.get()


def i(width):
    return IntegerType.get_signless(width)


def si(width):
    return IntegerType.get_signed(width)


def ui(width):
    return IntegerType.get_unsigned(width)


bool = lambda: i(1)
i8 = lambda: i(8)
i16 = lambda: i(16)
i32 = lambda: i(32)
i64 = lambda: i(64)

si8 = lambda: si(8)
si16 = lambda: si(16)
si32 = lambda: si(32)
si64 = lambda: si(64)

ui8 = lambda: ui(8)
ui16 = lambda: ui(16)
ui32 = lambda: ui(32)
ui64 = lambda: ui(64)

f16 = lambda: F16Type.get()
f32 = lambda: F32Type.get()
f64 = lambda: F64Type.get()
bf16 = lambda: BF16Type.get()

f8E5M2 = lambda: Float8E5M2Type.get()
f8E4M3 = lambda: Float8E4M3FNType.get()
f8E4M3B11FNUZ = lambda: Float8E4M3B11FNUZType.get()

none = lambda: NoneType.get()


def complex(type):
    return ComplexType.get(type)


def opaque(dialect_namespace, type_data):
    return OpaqueType.get(dialect_namespace, type_data)


def _shaped(*shape, element_type: Type = None, type_constructor=None):
    if type_constructor is None:
        raise ValueError("shaped is an abstract base class - cannot be constructed.")
    if (element_type is None and shape and not isinstance(shape[-1], Type)) or (
        shape and isinstance(shape[-1], Type) and element_type is not None
    ):
        raise ValueError(
            f"Either element_type must be provided explicitly XOR last arg to tensor type constructor must be the element type."
        )
    if element_type is not None:
        type = element_type
        sizes = shape
    else:
        type = shape[-1]
        sizes = shape[:-1]
    if sizes:
        return type_constructor(sizes, type)
    else:
        return type_constructor(type)


def vector(
    *shape,
    element_type: Type = None,
    scalable: Optional[List[bool]] = None,
    scalable_dims: Optional[List[int]] = None,
):
    return _shaped(
        *shape,
        element_type=element_type,
        type_constructor=partial(
            VectorType.get, scalable=scalable, scalable_dims=scalable_dims
        ),
    )


def tensor(*shape, element_type: Type = None, encoding: Optional[str] = None):
    if encoding is not None:
        encoding = StringAttr.get(encoding)
    if not shape or (len(shape) == 1 and isinstance(shape[-1], Type)):
        if encoding is not None:
            raise ValueError("UnrankedTensorType does not support encoding.")
        return _shaped(
            *shape, element_type=element_type, type_constructor=UnrankedTensorType.get
        )
    return _shaped(
        *shape,
        element_type=element_type,
        type_constructor=partial(RankedTensorType.get, encoding=encoding),
    )


def memref(
    *shape,
    element_type: Type = None,
    memory_space: Optional[int] = None,
    layout: Optional[StridedLayoutAttr] = None,
):
    if memory_space is not None:
        memory_space = Attribute.parse(str(memory_space))
    if not shape or (len(shape) == 1 and isinstance(shape[-1], Type)):
        return _shaped(
            *shape,
            element_type=element_type,
            type_constructor=partial(UnrankedMemRefType.get, memory_space=memory_space),
        )
    return _shaped(
        *shape,
        element_type=element_type,
        type_constructor=partial(
            MemRefType.get, memory_space=memory_space, layout=layout
        ),
    )


def tuple(*elements):
    return TupleType.get_tuple(elements)


def function(*, inputs, results):
    return FunctionType.get(inputs, results)
