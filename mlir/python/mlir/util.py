from .ir import (
    BF16Type,
    ComplexType,
    F16Type,
    F32Type,
    F64Type,
    IndexType,
    IntegerType,
    Type,
)


def is_complex_type(t: Type) -> bool:
    return ComplexType.isinstance(t)


def is_float_type(t: Type) -> bool:
    # TODO: Create a FloatType in the Python API and implement the switch
    # there.
    return (
        F64Type.isinstance(t)
        or F32Type.isinstance(t)
        or F16Type.isinstance(t)
        or BF16Type.isinstance(t)
    )


def is_integer_type(t: Type) -> bool:
    return IntegerType.isinstance(t)


def is_index_type(t: Type) -> bool:
    return IndexType.isinstance(t)


def is_integer_like_type(t: Type) -> bool:
    return is_integer_type(t) or is_index_type(t)


def get_floating_point_width(t: Type) -> int:
    # TODO: Create a FloatType in the Python API and implement the switch
    # there.
    if F64Type.isinstance(t):
        return 64
    if F32Type.isinstance(t):
        return 32
    if F16Type.isinstance(t):
        return 16
    if BF16Type.isinstance(t):
        return 16
    raise NotImplementedError(f"Unhandled floating point type switch {t}")
