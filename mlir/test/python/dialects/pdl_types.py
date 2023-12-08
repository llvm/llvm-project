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

    assert pdl.AttributeType.isinstance(parsedType)
    assert not pdl.OperationType.isinstance(parsedType)
    assert not pdl.RangeType.isinstance(parsedType)
    assert not pdl.TypeType.isinstance(parsedType)
    assert not pdl.ValueType.isinstance(parsedType)

    assert pdl.AttributeType.isinstance(constructedType)
    assert not pdl.OperationType.isinstance(constructedType)
    assert not pdl.RangeType.isinstance(constructedType)
    assert not pdl.TypeType.isinstance(constructedType)
    assert not pdl.ValueType.isinstance(constructedType)

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

    assert not pdl.AttributeType.isinstance(parsedType)
    assert pdl.OperationType.isinstance(parsedType)
    assert not pdl.RangeType.isinstance(parsedType)
    assert not pdl.TypeType.isinstance(parsedType)
    assert not pdl.ValueType.isinstance(parsedType)

    assert not pdl.AttributeType.isinstance(constructedType)
    assert pdl.OperationType.isinstance(constructedType)
    assert not pdl.RangeType.isinstance(constructedType)
    assert not pdl.TypeType.isinstance(constructedType)
    assert not pdl.ValueType.isinstance(constructedType)

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

    assert not pdl.AttributeType.isinstance(parsedType)
    assert not pdl.OperationType.isinstance(parsedType)
    assert pdl.RangeType.isinstance(parsedType)
    assert not pdl.TypeType.isinstance(parsedType)
    assert not pdl.ValueType.isinstance(parsedType)

    assert not pdl.AttributeType.isinstance(constructedType)
    assert not pdl.OperationType.isinstance(constructedType)
    assert pdl.RangeType.isinstance(constructedType)
    assert not pdl.TypeType.isinstance(constructedType)
    assert not pdl.ValueType.isinstance(constructedType)

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

    assert not pdl.AttributeType.isinstance(parsedType)
    assert not pdl.OperationType.isinstance(parsedType)
    assert not pdl.RangeType.isinstance(parsedType)
    assert pdl.TypeType.isinstance(parsedType)
    assert not pdl.ValueType.isinstance(parsedType)

    assert not pdl.AttributeType.isinstance(constructedType)
    assert not pdl.OperationType.isinstance(constructedType)
    assert not pdl.RangeType.isinstance(constructedType)
    assert pdl.TypeType.isinstance(constructedType)
    assert not pdl.ValueType.isinstance(constructedType)

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

    assert not pdl.AttributeType.isinstance(parsedType)
    assert not pdl.OperationType.isinstance(parsedType)
    assert not pdl.RangeType.isinstance(parsedType)
    assert not pdl.TypeType.isinstance(parsedType)
    assert pdl.ValueType.isinstance(parsedType)

    assert not pdl.AttributeType.isinstance(constructedType)
    assert not pdl.OperationType.isinstance(constructedType)
    assert not pdl.RangeType.isinstance(constructedType)
    assert not pdl.TypeType.isinstance(constructedType)
    assert pdl.ValueType.isinstance(constructedType)

    assert parsedType == constructedType

    # CHECK: !pdl.value
    print(parsedType)
    # CHECK: !pdl.value
    print(constructedType)
