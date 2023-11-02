#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import partial
from typing import Optional, List

from .ir import (
    Attribute,
    BF16Type,
    ComplexType,
    Context,
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

__all__ = []

_index = lambda: IndexType.get()
_bool = lambda: IntegerType.get_signless(1)

_i8 = lambda: IntegerType.get_signless(8)
_i16 = lambda: IntegerType.get_signless(16)
_i32 = lambda: IntegerType.get_signless(32)
_i64 = lambda: IntegerType.get_signless(64)

_si8 = lambda: IntegerType.get_signed(8)
_si16 = lambda: IntegerType.get_signed(16)
_si32 = lambda: IntegerType.get_signed(32)
_si64 = lambda: IntegerType.get_signed(64)

_ui8 = lambda: IntegerType.get_unsigned(8)
_ui16 = lambda: IntegerType.get_unsigned(16)
_ui32 = lambda: IntegerType.get_unsigned(32)
_ui64 = lambda: IntegerType.get_unsigned(64)

_f16 = lambda: F16Type.get()
_f32 = lambda: F32Type.get()
_f64 = lambda: F64Type.get()
_bf16 = lambda: BF16Type.get()

_f8e5m2 = lambda: Float8E5M2Type.get()
_f8e4m3 = lambda: Float8E4M3FNType.get()
_f8e4m3b11fnuz = lambda: Float8E4M3B11FNUZType.get()

_none = lambda: NoneType.get()


def _i(width):
    return IntegerType.get_signless(width)


def _si(width):
    return IntegerType.get_signed(width)


def _ui(width):
    return IntegerType.get_unsigned(width)


def _complex(type):
    return ComplexType.get(type)


def _opaque(dialect_namespace, type_data):
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


def _vector(
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


def _tensor(*shape, element_type: Type = None, encoding: Optional[str] = None):
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


def _memref(
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


def _tuple(*elements):
    return TupleType.get_tuple(elements)


def _function(*, inputs, results):
    return FunctionType.get(inputs, results)


def __getattr__(name):
    if name == "__path__":
        # https://docs.python.org/3/reference/import.html#path__
        # If a module is a package (either regular or namespace), the module object’s __path__ attribute must be set.
        # This module is NOT a package and so this must be None (rather than throw the RuntimeError below).
        return None
    try:
        Context.current
    except ValueError:
        raise RuntimeError("Types can only be instantiated under an active context.")

    if f"_{name}" in globals():
        builder = globals()[f"_{name}"]
        if (
            isinstance(builder, type(lambda: None))
            and builder.__name__ == (lambda: None).__name__
        ):
            return builder()
        return builder
    raise RuntimeError(f"{name} is not a legal type.")
