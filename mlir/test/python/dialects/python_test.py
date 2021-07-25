# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.python_test as test

def run(f):
  print("\nTEST:", f.__name__)
  f()

# CHECK-LABEL: TEST: testAttributes
def testAttributes():
  with Context() as ctx, Location.unknown():
    ctx.allow_unregistered_dialects = True

    #
    # Check op construction with attributes.
    #

    i32 = IntegerType.get_signless(32)
    one = IntegerAttr.get(i32, 1)
    two = IntegerAttr.get(i32, 2)
    unit = UnitAttr.get()

    # CHECK: "python_test.attributed_op"() {
    # CHECK-DAG: mandatory_i32 = 1 : i32
    # CHECK-DAG: optional_i32 = 2 : i32
    # CHECK-DAG: unit
    # CHECK: }
    op = test.AttributedOp(one, two, unit)
    print(f"{op}")

    # CHECK: "python_test.attributed_op"() {
    # CHECK: mandatory_i32 = 2 : i32
    # CHECK: }
    op2 = test.AttributedOp(two, None, None)
    print(f"{op2}")

    #
    # Check generic "attributes" access and mutation.
    #

    assert "additional" not in op.attributes

    # CHECK: "python_test.attributed_op"() {
    # CHECK-DAG: additional = 1 : i32
    # CHECK-DAG: mandatory_i32 = 2 : i32
    # CHECK: }
    op2.attributes["additional"] = one
    print(f"{op2}")

    # CHECK: "python_test.attributed_op"() {
    # CHECK-DAG: additional = 2 : i32
    # CHECK-DAG: mandatory_i32 = 2 : i32
    # CHECK: }
    op2.attributes["additional"] = two
    print(f"{op2}")

    # CHECK: "python_test.attributed_op"() {
    # CHECK-NOT: additional = 2 : i32
    # CHECK:     mandatory_i32 = 2 : i32
    # CHECK: }
    del op2.attributes["additional"]
    print(f"{op2}")

    try:
      print(op.attributes["additional"])
    except KeyError:
      pass
    else:
      assert False, "expected KeyError on unknown attribute key"

    #
    # Check accessors to defined attributes.
    #

    # CHECK: Mandatory: 1
    # CHECK: Optional: 2
    # CHECK: Unit: True
    print(f"Mandatory: {op.mandatory_i32.value}")
    print(f"Optional: {op.optional_i32.value}")
    print(f"Unit: {op.unit}")

    # CHECK: Mandatory: 2
    # CHECK: Optional: None
    # CHECK: Unit: False
    print(f"Mandatory: {op2.mandatory_i32.value}")
    print(f"Optional: {op2.optional_i32}")
    print(f"Unit: {op2.unit}")

    # CHECK: Mandatory: 2
    # CHECK: Optional: None
    # CHECK: Unit: False
    op.mandatory_i32 = two
    op.optional_i32 = None
    op.unit = False
    print(f"Mandatory: {op.mandatory_i32.value}")
    print(f"Optional: {op.optional_i32}")
    print(f"Unit: {op.unit}")
    assert "optional_i32" not in op.attributes
    assert "unit" not in op.attributes

    try:
      op.mandatory_i32 = None
    except ValueError:
      pass
    else:
      assert False, "expected ValueError on setting a mandatory attribute to None"

    # CHECK: Optional: 2
    op.optional_i32 = two
    print(f"Optional: {op.optional_i32.value}")

    # CHECK: Optional: None
    del op.optional_i32
    print(f"Optional: {op.optional_i32}")

    # CHECK: Unit: False
    op.unit = None
    print(f"Unit: {op.unit}")
    assert "unit" not in op.attributes

    # CHECK: Unit: True
    op.unit = True
    print(f"Unit: {op.unit}")

    # CHECK: Unit: False
    del op.unit
    print(f"Unit: {op.unit}")

run(testAttributes)
