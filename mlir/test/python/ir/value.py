# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *
from mlir.dialects import func


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


# CHECK-LABEL: TEST: testCapsuleConversions
@run
def testCapsuleConversions():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        i32 = IntegerType.get_signless(32)
        value = Operation.create("custom.op1", results=[i32]).result
        value_capsule = value._CAPIPtr
        assert '"mlir.ir.Value._CAPIPtr"' in repr(value_capsule)
        value2 = Value._CAPICreate(value_capsule)
        assert value2 == value


# CHECK-LABEL: TEST: testOpResultOwner
@run
def testOpResultOwner():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        i32 = IntegerType.get_signless(32)
        op = Operation.create("custom.op1", results=[i32])
        assert op.result.owner == op


# CHECK-LABEL: TEST: testBlockArgOwner
@run
def testBlockArgOwner():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    module = Module.parse(
        r"""
    func.func @foo(%arg0: f32) {
      return
    }""",
        ctx,
    )
    func = module.body.operations[0]
    block = func.regions[0].blocks[0]
    assert block.arguments[0].owner == block


# CHECK-LABEL: TEST: testValueIsInstance
@run
def testValueIsInstance():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    module = Module.parse(
        r"""
    func.func @foo(%arg0: f32) {
      %0 = "some_dialect.some_op"() : () -> f64
      return
    }""",
        ctx,
    )
    func = module.body.operations[0]
    assert BlockArgument.isinstance(func.regions[0].blocks[0].arguments[0])
    assert not OpResult.isinstance(func.regions[0].blocks[0].arguments[0])

    op = func.regions[0].blocks[0].operations[0]
    assert not BlockArgument.isinstance(op.results[0])
    assert OpResult.isinstance(op.results[0])


# CHECK-LABEL: TEST: testValueHash
@run
def testValueHash():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    module = Module.parse(
        r"""
    func.func @foo(%arg0: f32) -> f32 {
      %0 = "some_dialect.some_op"(%arg0) : (f32) -> f32
      return %0 : f32
    }""",
        ctx,
    )

    [func] = module.body.operations
    block = func.entry_block
    op, ret = block.operations
    assert hash(block.arguments[0]) == hash(op.operands[0])
    assert hash(op.result) == hash(ret.operands[0])


# CHECK-LABEL: TEST: testValueUses
@run
def testValueUses():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        i32 = IntegerType.get_signless(32)
        module = Module.create()
        with InsertionPoint(module.body):
            value = Operation.create("custom.op1", results=[i32]).results[0]
            op1 = Operation.create("custom.op2", operands=[value])
            op2 = Operation.create("custom.op2", operands=[value])

    # CHECK: Use owner: "custom.op2"
    # CHECK: Use operand_number: 0
    # CHECK: Use owner: "custom.op2"
    # CHECK: Use operand_number: 0
    for use in value.uses:
        assert use.owner in [op1, op2]
        print(f"Use owner: {use.owner}")
        print(f"Use operand_number: {use.operand_number}")


# CHECK-LABEL: TEST: testValueReplaceAllUsesWith
@run
def testValueReplaceAllUsesWith():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        i32 = IntegerType.get_signless(32)
        module = Module.create()
        with InsertionPoint(module.body):
            value = Operation.create("custom.op1", results=[i32]).results[0]
            op1 = Operation.create("custom.op2", operands=[value])
            op2 = Operation.create("custom.op2", operands=[value])
            value2 = Operation.create("custom.op3", results=[i32]).results[0]
            value.replace_all_uses_with(value2)

    assert len(list(value.uses)) == 0

    # CHECK: Use owner: "custom.op2"
    # CHECK: Use operand_number: 0
    # CHECK: Use owner: "custom.op2"
    # CHECK: Use operand_number: 0
    for use in value2.uses:
        assert use.owner in [op1, op2]
        print(f"Use owner: {use.owner}")
        print(f"Use operand_number: {use.operand_number}")


# CHECK-LABEL: TEST: testValuePrintAsOperand
@run
def testValuePrintAsOperand():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        i32 = IntegerType.get_signless(32)
        module = Module.create()
        with InsertionPoint(module.body):
            value = Operation.create("custom.op1", results=[i32]).results[0]
            # CHECK: Value(%[[VAL1:.*]] = "custom.op1"() : () -> i32)
            print(value)

            value2 = Operation.create("custom.op2", results=[i32]).results[0]
            # CHECK: Value(%[[VAL2:.*]] = "custom.op2"() : () -> i32)
            print(value2)

            f = func.FuncOp("test", ([i32, i32], []))
            entry_block1 = Block.create_at_start(f.operation.regions[0], [i32, i32])

            with InsertionPoint(entry_block1):
                value3 = Operation.create("custom.op3", results=[i32]).results[0]
                # CHECK: Value(%[[VAL3:.*]] = "custom.op3"() : () -> i32)
                print(value3)
                value4 = Operation.create("custom.op4", results=[i32]).results[0]
                # CHECK: Value(%[[VAL4:.*]] = "custom.op4"() : () -> i32)
                print(value4)

                f = func.FuncOp("test", ([i32, i32], []))
                entry_block2 = Block.create_at_start(f.operation.regions[0], [i32, i32])
                with InsertionPoint(entry_block2):
                    value5 = Operation.create("custom.op5", results=[i32]).results[0]
                    # CHECK: Value(%[[VAL5:.*]] = "custom.op5"() : () -> i32)
                    print(value5)
                    value6 = Operation.create("custom.op6", results=[i32]).results[0]
                    # CHECK: Value(%[[VAL6:.*]] = "custom.op6"() : () -> i32)
                    print(value6)

                    func.ReturnOp([])

                func.ReturnOp([])

        # CHECK: %[[VAL1]]
        print(value.get_name())
        # CHECK: %[[VAL2]]
        print(value2.get_name())
        # CHECK: %[[VAL3]]
        print(value3.get_name())
        # CHECK: %[[VAL4]]
        print(value4.get_name())

        # CHECK: %0
        print(value3.get_name(use_local_scope=True))
        # CHECK: %1
        print(value4.get_name(use_local_scope=True))

        # CHECK: %[[VAL5]]
        print(value5.get_name())
        # CHECK: %[[VAL6]]
        print(value6.get_name())

        # CHECK: %[[ARG0:.*]]
        print(entry_block1.arguments[0].get_name())
        # CHECK: %[[ARG1:.*]]
        print(entry_block1.arguments[1].get_name())

        # CHECK: %[[ARG2:.*]]
        print(entry_block2.arguments[0].get_name())
        # CHECK: %[[ARG3:.*]]
        print(entry_block2.arguments[1].get_name())

        # CHECK: module {
        # CHECK:   %[[VAL1]] = "custom.op1"() : () -> i32
        # CHECK:   %[[VAL2]] = "custom.op2"() : () -> i32
        # CHECK:   func.func @test(%[[ARG0]]: i32, %[[ARG1]]: i32) {
        # CHECK:     %[[VAL3]] = "custom.op3"() : () -> i32
        # CHECK:     %[[VAL4]] = "custom.op4"() : () -> i32
        # CHECK:     func @test(%[[ARG2]]: i32, %[[ARG3]]: i32) {
        # CHECK:       %[[VAL5]] = "custom.op5"() : () -> i32
        # CHECK:       %[[VAL6]] = "custom.op6"() : () -> i32
        # CHECK:       return
        # CHECK:     }
        # CHECK:     return
        # CHECK:   }
        # CHECK: }
        print(module)

        value2.owner.detach_from_parent()
        # CHECK: %0
        print(value2.get_name())
