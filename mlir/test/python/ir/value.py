# RUN: %PYTHON %s | FileCheck %s --enable-var-scope=false

import gc
from mlir.ir import *
from mlir.dialects import func
from mlir.dialects._ods_common import SubClassValueT


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

            topFn = func.FuncOp("test", ([i32, i32], []))
            entry_block = Block.create_at_start(topFn.operation.regions[0], [i32, i32])

            with InsertionPoint(entry_block):
                value3 = Operation.create("custom.op3", results=[i32]).results[0]
                # CHECK: Value(%[[VAL3:.*]] = "custom.op3"() : () -> i32)
                print(value3)
                value4 = Operation.create("custom.op4", results=[i32]).results[0]
                # CHECK: Value(%[[VAL4:.*]] = "custom.op4"() : () -> i32)
                print(value4)
                func.ReturnOp([])

        # CHECK: %[[VAL1]]
        print(value.get_name())
        # CHECK: %[[VAL2]]
        print(value2.get_name())
        # CHECK: %[[VAL3]]
        print(value3.get_name())
        # CHECK: %[[VAL4]]
        print(value4.get_name())

        print("With AsmState")
        # CHECK-LABEL: With AsmState
        state = AsmState(topFn.operation, use_local_scope=True)
        # CHECK: %0
        print(value3.get_name(state=state))
        # CHECK: %1
        print(value4.get_name(state=state))

        print("With use_local_scope")
        # CHECK-LABEL: With use_local_scope
        # CHECK: %0
        print(value3.get_name(use_local_scope=True))
        # CHECK: %1
        print(value4.get_name(use_local_scope=True))

        # CHECK: %[[ARG0:.*]]
        print(entry_block.arguments[0].get_name())
        # CHECK: %[[ARG1:.*]]
        print(entry_block.arguments[1].get_name())

        # CHECK: module {
        # CHECK:   %[[VAL1]] = "custom.op1"() : () -> i32
        # CHECK:   %[[VAL2]] = "custom.op2"() : () -> i32
        # CHECK:   func.func @test(%[[ARG0]]: i32, %[[ARG1]]: i32) {
        # CHECK:     %[[VAL3]] = "custom.op3"() : () -> i32
        # CHECK:     %[[VAL4]] = "custom.op4"() : () -> i32
        # CHECK:     return
        # CHECK:   }
        # CHECK: }
        print(module)

        value2.owner.detach_from_parent()
        # CHECK: %0
        print(value2.get_name())


# CHECK-LABEL: TEST: testValueSetType
@run
def testValueSetType():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        i32 = IntegerType.get_signless(32)
        i64 = IntegerType.get_signless(64)
        module = Module.create()
        with InsertionPoint(module.body):
            value = Operation.create("custom.op1", results=[i32]).results[0]
            # CHECK: Value(%[[VAL1:.*]] = "custom.op1"() : () -> i32)
            print(value)

            value.set_type(i64)
            # CHECK: Value(%[[VAL1]] = "custom.op1"() : () -> i64)
            print(value)

            # CHECK: %[[VAL1]] = "custom.op1"() : () -> i64
            print(value.owner)


# CHECK-LABEL: TEST: testValueCasters
@run
def testValueCasters():
    class NOPResult(OpResult):
        def __init__(self, v):
            super().__init__(v)

        def __str__(self):
            return super().__str__().replace(Value.__name__, NOPResult.__name__)

    class NOPValue(Value):
        def __init__(self, v):
            super().__init__(v)

        def __str__(self):
            return super().__str__().replace(Value.__name__, NOPValue.__name__)

    class NOPBlockArg(BlockArgument):
        def __init__(self, v):
            super().__init__(v)

        def __str__(self):
            return super().__str__().replace(Value.__name__, NOPBlockArg.__name__)

    @register_value_caster(IntegerType.static_typeid)
    def cast_int(v) -> SubClassValueT:
        print("in caster", v.__class__.__name__)
        if isinstance(v, OpResult):
            return NOPResult(v)
        if isinstance(v, BlockArgument):
            return NOPBlockArg(v)
        elif isinstance(v, Value):
            return NOPValue(v)

    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        i32 = IntegerType.get_signless(32)
        module = Module.create()
        with InsertionPoint(module.body):
            values = Operation.create("custom.op1", results=[i32, i32]).results
            # CHECK: in caster OpResult
            # CHECK: result 0 NOPResult(%0:2 = "custom.op1"() : () -> (i32, i32))
            print("result", values[0].result_number, values[0])
            # CHECK: in caster OpResult
            # CHECK: result 1 NOPResult(%0:2 = "custom.op1"() : () -> (i32, i32))
            print("result", values[1].result_number, values[1])

            # CHECK: results slice 0 NOPResult(%0:2 = "custom.op1"() : () -> (i32, i32))
            print("results slice", values[:1][0].result_number, values[:1][0])

            value0, value1 = values
            # CHECK: in caster OpResult
            # CHECK: result 0 NOPResult(%0:2 = "custom.op1"() : () -> (i32, i32))
            print("result", value0.result_number, values[0])
            # CHECK: in caster OpResult
            # CHECK: result 1 NOPResult(%0:2 = "custom.op1"() : () -> (i32, i32))
            print("result", value1.result_number, values[1])

            op1 = Operation.create("custom.op2", operands=[value0, value1])
            # CHECK: "custom.op2"(%0#0, %0#1) : (i32, i32) -> ()
            print(op1)

            # CHECK: in caster Value
            # CHECK: operand 0 NOPValue(%0:2 = "custom.op1"() : () -> (i32, i32))
            print("operand 0", op1.operands[0])
            # CHECK: in caster Value
            # CHECK: operand 1 NOPValue(%0:2 = "custom.op1"() : () -> (i32, i32))
            print("operand 1", op1.operands[1])

            # CHECK: in caster BlockArgument
            # CHECK: in caster BlockArgument
            @func.FuncOp.from_py_func(i32, i32)
            def reduction(arg0, arg1):
                # CHECK: as func arg 0 NOPBlockArg
                print("as func arg", arg0.arg_number, arg0.__class__.__name__)
                # CHECK: as func arg 1 NOPBlockArg
                print("as func arg", arg1.arg_number, arg1.__class__.__name__)

            # CHECK: args slice 0 NOPBlockArg(<block argument> of type 'i32' at index: 0)
            print(
                "args slice",
                reduction.func_op.arguments[:1][0].arg_number,
                reduction.func_op.arguments[:1][0],
            )

    try:

        @register_value_caster(IntegerType.static_typeid)
        def dont_cast_int_shouldnt_register(v):
            ...

    except RuntimeError as e:
        # CHECK: Value caster is already registered: {{.*}}cast_int
        print(e)

    @register_value_caster(IntegerType.static_typeid, replace=True)
    def dont_cast_int(v) -> OpResult:
        assert isinstance(v, OpResult)
        print("don't cast", v.result_number, v)
        return v

    with Location.unknown(ctx):
        i32 = IntegerType.get_signless(32)
        module = Module.create()
        with InsertionPoint(module.body):
            # CHECK: don't cast 0 Value(%0 = "custom.op1"() : () -> i32)
            new_value = Operation.create("custom.op1", results=[i32]).result
            # CHECK: result 0 Value(%0 = "custom.op1"() : () -> i32)
            print("result", new_value.result_number, new_value)

            # CHECK: don't cast 0 Value(%1 = "custom.op2"() : () -> i32)
            new_value = Operation.create("custom.op2", results=[i32]).results[0]
            # CHECK: result 0 Value(%1 = "custom.op2"() : () -> i32)
            print("result", new_value.result_number, new_value)
