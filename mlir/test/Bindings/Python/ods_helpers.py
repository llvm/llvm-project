# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0


def add_dummy_value():
  return Operation.create(
      "custom.value",
      results=[IntegerType.get_signless(32)]).result


def testOdsBuildDefaultImplicitRegions():

  class TestFixedRegionsOp(OpView):
    OPERATION_NAME = "custom.test_op"
    _ODS_REGIONS = (2, True)

  class TestVariadicRegionsOp(OpView):
    OPERATION_NAME = "custom.test_any_regions_op"
    _ODS_REGIONS = (2, False)

  with Context() as ctx, Location.unknown():
    ctx.allow_unregistered_dialects = True
    m = Module.create()
    with InsertionPoint.at_block_terminator(m.body):
      op = TestFixedRegionsOp.build_generic(results=[], operands=[])
      # CHECK: NUM_REGIONS: 2
      print(f"NUM_REGIONS: {len(op.regions)}")
      # Including a regions= that matches should be fine.
      op = TestFixedRegionsOp.build_generic(results=[], operands=[], regions=2)
      print(f"NUM_REGIONS: {len(op.regions)}")
      # Reject greater than.
      try:
        op = TestFixedRegionsOp.build_generic(results=[], operands=[], regions=3)
      except ValueError as e:
        # CHECK: ERROR:Operation "custom.test_op" requires a maximum of 2 regions but was built with regions=3
        print(f"ERROR:{e}")
      # Reject less than.
      try:
        op = TestFixedRegionsOp.build_generic(results=[], operands=[], regions=1)
      except ValueError as e:
        # CHECK: ERROR:Operation "custom.test_op" requires a minimum of 2 regions but was built with regions=1
        print(f"ERROR:{e}")

      # If no regions specified for a variadic region op, build the minimum.
      op = TestVariadicRegionsOp.build_generic(results=[], operands=[])
      # CHECK: DEFAULT_NUM_REGIONS: 2
      print(f"DEFAULT_NUM_REGIONS: {len(op.regions)}")
      # Should also accept an explicit regions= that matches the minimum.
      op = TestVariadicRegionsOp.build_generic(
          results=[], operands=[], regions=2)
      # CHECK: EQ_NUM_REGIONS: 2
      print(f"EQ_NUM_REGIONS: {len(op.regions)}")
      # And accept greater than minimum.
      # Should also accept an explicit regions= that matches the minimum.
      op = TestVariadicRegionsOp.build_generic(
          results=[], operands=[], regions=3)
      # CHECK: GT_NUM_REGIONS: 3
      print(f"GT_NUM_REGIONS: {len(op.regions)}")
      # Should reject less than minimum.
      try:
        op = TestVariadicRegionsOp.build_generic(results=[], operands=[], regions=1)
      except ValueError as e:
        # CHECK: ERROR:Operation "custom.test_any_regions_op" requires a minimum of 2 regions but was built with regions=1
        print(f"ERROR:{e}")



run(testOdsBuildDefaultImplicitRegions)


def testOdsBuildDefaultNonVariadic():

  class TestOp(OpView):
    OPERATION_NAME = "custom.test_op"

  with Context() as ctx, Location.unknown():
    ctx.allow_unregistered_dialects = True
    m = Module.create()
    with InsertionPoint.at_block_terminator(m.body):
      v0 = add_dummy_value()
      v1 = add_dummy_value()
      t0 = IntegerType.get_signless(8)
      t1 = IntegerType.get_signless(16)
      op = TestOp.build_generic(results=[t0, t1], operands=[v0, v1])
      # CHECK: %[[V0:.+]] = "custom.value"
      # CHECK: %[[V1:.+]] = "custom.value"
      # CHECK: "custom.test_op"(%[[V0]], %[[V1]])
      # CHECK-NOT: operand_segment_sizes
      # CHECK-NOT: result_segment_sizes
      # CHECK-SAME: : (i32, i32) -> (i8, i16)
      print(m)

run(testOdsBuildDefaultNonVariadic)


def testOdsBuildDefaultSizedVariadic():

  class TestOp(OpView):
    OPERATION_NAME = "custom.test_op"
    _ODS_OPERAND_SEGMENTS = [1, -1, 0]
    _ODS_RESULT_SEGMENTS = [-1, 0, 1]

  with Context() as ctx, Location.unknown():
    ctx.allow_unregistered_dialects = True
    m = Module.create()
    with InsertionPoint.at_block_terminator(m.body):
      v0 = add_dummy_value()
      v1 = add_dummy_value()
      v2 = add_dummy_value()
      v3 = add_dummy_value()
      t0 = IntegerType.get_signless(8)
      t1 = IntegerType.get_signless(16)
      t2 = IntegerType.get_signless(32)
      t3 = IntegerType.get_signless(64)
      # CHECK: %[[V0:.+]] = "custom.value"
      # CHECK: %[[V1:.+]] = "custom.value"
      # CHECK: %[[V2:.+]] = "custom.value"
      # CHECK: %[[V3:.+]] = "custom.value"
      # CHECK: "custom.test_op"(%[[V0]], %[[V1]], %[[V2]], %[[V3]])
      # CHECK-SAME: operand_segment_sizes = dense<[1, 2, 1]> : vector<3xi32>
      # CHECK-SAME: result_segment_sizes = dense<[2, 1, 1]> : vector<3xi32>
      # CHECK-SAME: : (i32, i32, i32, i32) -> (i8, i16, i32, i64)
      op = TestOp.build_generic(
          results=[[t0, t1], t2, t3],
          operands=[v0, [v1, v2], v3])

      # Now test with optional omitted.
      # CHECK: "custom.test_op"(%[[V0]])
      # CHECK-SAME: operand_segment_sizes = dense<[1, 0, 0]>
      # CHECK-SAME: result_segment_sizes = dense<[0, 0, 1]>
      # CHECK-SAME: (i32) -> i64
      op = TestOp.build_generic(
          results=[None, None, t3],
          operands=[v0, None, None])
      print(m)

      # And verify that errors are raised for None in a required operand.
      try:
        op = TestOp.build_generic(
            results=[None, None, t3],
            operands=[None, None, None])
      except ValueError as e:
        # CHECK: OPERAND_CAST_ERROR:Operand 0 of operation "custom.test_op" must be a Value (was None and operand is not optional)
        print(f"OPERAND_CAST_ERROR:{e}")

      # And verify that errors are raised for None in a required result.
      try:
        op = TestOp.build_generic(
            results=[None, None, None],
            operands=[v0, None, None])
      except ValueError as e:
        # CHECK: RESULT_CAST_ERROR:Result 2 of operation "custom.test_op" must be a Type (was None and result is not optional)
        print(f"RESULT_CAST_ERROR:{e}")

      # Variadic lists with None elements should reject.
      try:
        op = TestOp.build_generic(
            results=[None, None, t3],
            operands=[v0, [None], None])
      except ValueError as e:
        # CHECK: OPERAND_LIST_CAST_ERROR:Operand 1 of operation "custom.test_op" must be a Sequence of Values (contained a None item)
        print(f"OPERAND_LIST_CAST_ERROR:{e}")
      try:
        op = TestOp.build_generic(
            results=[[None], None, t3],
            operands=[v0, None, None])
      except ValueError as e:
        # CHECK: RESULT_LIST_CAST_ERROR:Result 0 of operation "custom.test_op" must be a Sequence of Types (contained a None item)
        print(f"RESULT_LIST_CAST_ERROR:{e}")

run(testOdsBuildDefaultSizedVariadic)


def testOdsBuildDefaultCastError():

  class TestOp(OpView):
    OPERATION_NAME = "custom.test_op"

  with Context() as ctx, Location.unknown():
    ctx.allow_unregistered_dialects = True
    m = Module.create()
    with InsertionPoint.at_block_terminator(m.body):
      v0 = add_dummy_value()
      v1 = add_dummy_value()
      t0 = IntegerType.get_signless(8)
      t1 = IntegerType.get_signless(16)
      try:
        op = TestOp.build_generic(
            results=[t0, t1],
            operands=[None, v1])
      except ValueError as e:
        # CHECK: ERROR: Operand 0 of operation "custom.test_op" must be a Value
        print(f"ERROR: {e}")
      try:
        op = TestOp.build_generic(
            results=[t0, None],
            operands=[v0, v1])
      except ValueError as e:
        # CHECK: Result 1 of operation "custom.test_op" must be a Type
        print(f"ERROR: {e}")

run(testOdsBuildDefaultCastError)
