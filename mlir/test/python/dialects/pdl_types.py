# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import pdl


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: test_attribute_type
@run
def test_attribute_type():
    with Context():
        parsedType = Type.parse("!pdl.attribute")
        constructedType = pdl.AttributeType.get()

        assert isinstance(parsedType, pdl.AttributeType)
        assert not isinstance(parsedType, pdl.OperationType)
        assert not isinstance(parsedType, pdl.RangeType)
        assert not isinstance(parsedType, pdl.TypeType)
        assert not isinstance(parsedType, pdl.ValueType)

        assert isinstance(constructedType, pdl.AttributeType)
        assert not isinstance(constructedType, pdl.OperationType)
        assert not isinstance(constructedType, pdl.RangeType)
        assert not isinstance(constructedType, pdl.TypeType)
        assert not isinstance(constructedType, pdl.ValueType)

        assert parsedType == constructedType

        # CHECK: !pdl.attribute
        print(parsedType)
        # CHECK: !pdl.attribute
        print(constructedType)


# CHECK-LABEL: TEST: test_operation_type
@run
def test_operation_type():
    with Context():
        parsedType = Type.parse("!pdl.operation")
        constructedType = pdl.OperationType.get()

        assert not isinstance(parsedType, pdl.AttributeType)
        assert isinstance(parsedType, pdl.OperationType)
        assert not isinstance(parsedType, pdl.RangeType)
        assert not isinstance(parsedType, pdl.TypeType)
        assert not isinstance(parsedType, pdl.ValueType)

        assert not isinstance(constructedType, pdl.AttributeType)
        assert isinstance(constructedType, pdl.OperationType)
        assert not isinstance(constructedType, pdl.RangeType)
        assert not isinstance(constructedType, pdl.TypeType)
        assert not isinstance(constructedType, pdl.ValueType)

        assert parsedType == constructedType

        # CHECK: !pdl.operation
        print(parsedType)
        # CHECK: !pdl.operation
        print(constructedType)


# CHECK-LABEL: TEST: test_range_type
@run
def test_range_type():
    with Context():
        typeType = Type.parse("!pdl.type")
        parsedType = Type.parse("!pdl.range<type>")
        constructedType = pdl.RangeType.get(typeType)
        elementType = constructedType.element_type

        assert not isinstance(parsedType, pdl.AttributeType)
        assert not isinstance(parsedType, pdl.OperationType)
        assert isinstance(parsedType, pdl.RangeType)
        assert not isinstance(parsedType, pdl.TypeType)
        assert not isinstance(parsedType, pdl.ValueType)

        assert not isinstance(constructedType, pdl.AttributeType)
        assert not isinstance(constructedType, pdl.OperationType)
        assert isinstance(constructedType, pdl.RangeType)
        assert not isinstance(constructedType, pdl.TypeType)
        assert not isinstance(constructedType, pdl.ValueType)

        assert parsedType == constructedType
        assert elementType == typeType

        # CHECK: !pdl.range<type>
        print(parsedType)
        # CHECK: !pdl.range<type>
        print(constructedType)
        # CHECK: !pdl.type
        print(elementType)


# CHECK-LABEL: TEST: test_type_type
@run
def test_type_type():
    with Context():
        parsedType = Type.parse("!pdl.type")
        constructedType = pdl.TypeType.get()

        assert not isinstance(parsedType, pdl.AttributeType)
        assert not isinstance(parsedType, pdl.OperationType)
        assert not isinstance(parsedType, pdl.RangeType)
        assert isinstance(parsedType, pdl.TypeType)
        assert not isinstance(parsedType, pdl.ValueType)

        assert not isinstance(constructedType, pdl.AttributeType)
        assert not isinstance(constructedType, pdl.OperationType)
        assert not isinstance(constructedType, pdl.RangeType)
        assert isinstance(constructedType, pdl.TypeType)
        assert not isinstance(constructedType, pdl.ValueType)

        assert parsedType == constructedType

        # CHECK: !pdl.type
        print(parsedType)
        # CHECK: !pdl.type
        print(constructedType)


# CHECK-LABEL: TEST: test_value_type
@run
def test_value_type():
    with Context():
        parsedType = Type.parse("!pdl.value")
        constructedType = pdl.ValueType.get()

        assert not isinstance(parsedType, pdl.AttributeType)
        assert not isinstance(parsedType, pdl.OperationType)
        assert not isinstance(parsedType, pdl.RangeType)
        assert not isinstance(parsedType, pdl.TypeType)
        assert isinstance(parsedType, pdl.ValueType)

        assert not isinstance(constructedType, pdl.AttributeType)
        assert not isinstance(constructedType, pdl.OperationType)
        assert not isinstance(constructedType, pdl.RangeType)
        assert not isinstance(constructedType, pdl.TypeType)
        assert isinstance(constructedType, pdl.ValueType)

        assert parsedType == constructedType

        # CHECK: !pdl.value
        print(parsedType)
        # CHECK: !pdl.value
        print(constructedType)


# CHECK-LABEL: TEST: test_type_without_context
@run
def test_type_without_context():
    # Constructing a type without the surrounding ir.Context context manager
    # should raise an exception but not crash.
    try:
        constructedType = pdl.ValueType.get()
    except RuntimeError as e:
        assert (
            "An MLIR function requires a Context but none was provided in the call or from the surrounding environment"
            in e.args[0]
        )
    else:
        assert False, "Expected TypeError to be raised."
