# RUN: env PYTHONUNBUFFERED=1 %PYTHON %s 2>&1 | FileCheck %s

from contextlib import contextmanager
from typing import Any

from mlir import ir
from mlir.dialects import arith, ext, func, scf
from mlir.passmanager import PassManager


class MemoryEffectsTest(ext.Dialect, name="memory_effects_test"):
    pass


class NoEffectModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op):
        return []


class ReadModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op):
        return [
            ir.MemoryEffectInstance(
                ir.MemoryEffect.Read,
                op.op_operands[0],
                parameters=ir.StringAttr.get("read parameter"),
                stage=1,
                effect_on_full_region=True,
                resource=ir.SideEffectResource.Default,
            )
        ]


class ReadDeadModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op):
        return [ir.MemoryEffectInstance(ir.MemoryEffect.Read)]


class WriteModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op):
        return [ir.MemoryEffectInstance(ir.MemoryEffect.Write)]


class FreeModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op):
        return [ir.MemoryEffectInstance(ir.MemoryEffect.Free)]


class AllocateModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op):
        return [ir.MemoryEffectInstance(ir.MemoryEffect.Allocate)]


class AllocateResultModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op):
        return [ir.MemoryEffectInstance(ir.MemoryEffect.Allocate, op.results[0])]


class BlockArgumentTargetModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op):
        return [
            ir.MemoryEffectInstance(
                ir.MemoryEffect.Read, op.regions[0].blocks[0].arguments[0]
            )
        ]


class SymbolTargetModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op):
        try:
            ir.MemoryEffectInstance(
                ir.MemoryEffect.Read, ir.StringAttr.get("not a symbol")
            )
        except TypeError as error:
            print("invalid symbol target:", error)
        try:
            ir.MemoryEffectInstance(ir.MemoryEffect.Read, parameters=42)
        except TypeError as error:
            print("invalid parameters:", error)
        try:
            ir.MemoryEffectInstance(ir.MemoryEffect.Read, 42)
        except TypeError as error:
            print("invalid target:", error)
        return [
            ir.MemoryEffectInstance(
                ir.MemoryEffect.Read,
                ir.FlatSymbolRefAttr.get("global"),
                parameters=ir.StringAttr.get("symbol parameter"),
                stage=2,
                effect_on_full_region=True,
            )
        ]


class ReadOp(MemoryEffectsTest.Operation, name="read", traits=[ReadModel]):
    operand: ext.Operand[Any]
    result: ext.Result[Any]


class WriteOp(MemoryEffectsTest.Operation, name="write", traits=[WriteModel]):
    operand: ext.Operand[Any]
    result: ext.Result[Any]


class WriteBarrierOp(
    MemoryEffectsTest.Operation, name="write_barrier", traits=[WriteModel]
):
    operand: ext.Operand[Any]


class NoEffectOp(MemoryEffectsTest.Operation, name="no_effect", traits=[NoEffectModel]):
    pass


class ReadDeadOp(MemoryEffectsTest.Operation, name="read_dead", traits=[ReadDeadModel]):
    pass


class WriteDeadOp(MemoryEffectsTest.Operation, name="write_dead", traits=[WriteModel]):
    pass


class FreeDeadOp(MemoryEffectsTest.Operation, name="free_dead", traits=[FreeModel]):
    pass


class AllocateDeadOp(
    MemoryEffectsTest.Operation, name="allocate_dead", traits=[AllocateModel]
):
    pass


class AllocateResultOp(
    MemoryEffectsTest.Operation,
    name="allocate_result",
    traits=[AllocateResultModel],
):
    result: ext.Result[Any]


class BlockArgumentTargetOp(
    MemoryEffectsTest.Operation,
    name="block_argument_target",
    traits=[ir.NoTerminatorTrait, BlockArgumentTargetModel],
):
    body: ext.Region


class SymbolTargetOp(
    MemoryEffectsTest.Operation, name="symbol_target", traits=[SymbolTargetModel]
):
    pass


class RegionOp(
    MemoryEffectsTest.Operation,
    name="region",
    traits=[ir.NoTerminatorTrait],
):
    result: ext.Result[Any]
    body: ext.Region


class RecursiveRegionOp(
    MemoryEffectsTest.Operation,
    name="recursive_region",
    traits=[ir.NoTerminatorTrait, ir.RecursiveMemoryEffectsTrait],
):
    result: ext.Result[Any]
    body: ext.Region


def run(test):
    print("\nTEST:", test.__name__)
    with ir.Context(), ir.Location.unknown():
        MemoryEffectsTest.load()
        test()


@contextmanager
def function_module(inputs=(), results=()):
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        function = func.FuncOp("test", (inputs, results))
        with ir.InsertionPoint(function.add_entry_block()):
            yield module, function


def run_pass(module, pipeline):
    PassManager.parse(pipeline).run(module.operation)


def count_ops(module, op_type):
    return len(ir.get_ops_of_type(module, op_type))


# CHECK-LABEL: TEST: testRecursiveMemoryEffectsTraits
@run
def testRecursiveMemoryEffectsTraits():
    # CHECK: recursive memory effects traits: False True True False
    print(
        "recursive memory effects traits:",
        RegionOp.has_trait(ir.RecursiveMemoryEffectsTrait),
        RecursiveRegionOp.has_trait(ir.RecursiveMemoryEffectsTrait),
        scf.IfOp.has_trait(ir.RecursiveMemoryEffectsTrait),
        arith.AddIOp.has_trait(ir.RecursiveMemoryEffectsTrait),
    )


# CHECK-LABEL: TEST: testMemoryEffectProperties
@run
def testMemoryEffectProperties():
    # CHECK: memory effect properties: True True True True
    print(
        "memory effect properties:",
        isinstance(ir.MemoryEffect.Allocate, ir.MemoryEffect),
        isinstance(ir.MemoryEffect.Free, ir.MemoryEffect),
        isinstance(ir.MemoryEffect.Read, ir.MemoryEffect),
        isinstance(ir.MemoryEffect.Write, ir.MemoryEffect),
    )
    # CHECK: memory effect equality: True True True True False False
    print(
        "memory effect equality:",
        ir.MemoryEffect.Allocate == ir.MemoryEffect.Allocate,
        ir.MemoryEffect.Free == ir.MemoryEffect.Free,
        ir.MemoryEffect.Read == ir.MemoryEffect.Read,
        ir.MemoryEffect.Write == ir.MemoryEffect.Write,
        ir.MemoryEffect.Read == ir.MemoryEffect.Write,
        ir.MemoryEffect.Read == 42,
    )
    # CHECK: default resource property: True
    print(
        "default resource property:",
        isinstance(ir.SideEffectResource.Default, ir.SideEffectResource),
    )


# CHECK-LABEL: TEST: testQueryMemoryEffects
@run
def testQueryMemoryEffects():
    i32 = ir.IntegerType.get_signless(32)
    with function_module(inputs=[i32], results=[i32]) as (_, function):
        read_op = ReadOp(function.arguments[0], i32)
        func.ReturnOp([read_op.result])

    read_effects = ir.MemoryEffectsOpInterface(read_op).get_effects()
    read_effect = read_effects[0]
    # CHECK: queried effects: True 1 True True 1 True True True
    print(
        "queried effects:",
        isinstance(read_effects, list),
        len(read_effects),
        isinstance(read_effect, ir.MemoryEffectInstance),
        read_effect.effect == ir.MemoryEffect.Read,
        read_effect.stage,
        read_effect.effect_on_full_region,
        isinstance(read_effect.resource, ir.SideEffectResource),
        read_effect.value == read_op.operands[0],
    )
    # CHECK: queried optional properties: "read parameter" True
    print(
        "queried optional properties:",
        read_effect.parameters,
        read_effect.symbol_ref is None,
    )


# CHECK-LABEL: TEST: testSymbolEffectProperties
@run
def testSymbolEffectProperties():
    symbol_effect = ir.MemoryEffectInstance(
        ir.MemoryEffect.Read, ir.FlatSymbolRefAttr.get("global")
    )
    # CHECK: symbol effect properties: True True True
    print(
        "symbol effect properties:",
        isinstance(symbol_effect.symbol_ref, ir.FlatSymbolRefAttr),
        symbol_effect.value is None,
        symbol_effect.parameters is None,
    )


# CHECK-LABEL: TEST: testMemoryEffectsCSE
@run
def testMemoryEffectsCSE():
    i32 = ir.IntegerType.get_signless(32)

    with function_module(inputs=[i32], results=[i32, i32]) as (
        read_cse,
        function,
    ):
        read0 = ReadOp(function.arguments[0], i32)
        read1 = ReadOp(function.arguments[0], i32)
        func.ReturnOp([read0.result, read1.result])
    run_pass(read_cse, "builtin.module(func.func(cse))")

    # A single Read effect remains CSE-eligible.
    # CHECK: CSE read count: 1
    print("CSE read count:", count_ops(read_cse, ReadOp))

    with function_module(inputs=[i32], results=[i32, i32]) as (
        write_cse,
        function,
    ):
        write0 = WriteOp(function.arguments[0], i32)
        write1 = WriteOp(function.arguments[0], i32)
        func.ReturnOp([write0.result, write1.result])
    run_pass(write_cse, "builtin.module(func.func(cse))")

    # Writes cannot be CSE'd.
    # CHECK: CSE write count: 2
    print("CSE write count:", count_ops(write_cse, WriteOp))

    with function_module(inputs=[i32], results=[i32, i32]) as (
        read_across_write,
        function,
    ):
        read0 = ReadOp(function.arguments[0], i32)
        WriteBarrierOp(function.arguments[0])
        read1 = ReadOp(function.arguments[0], i32)
        func.ReturnOp([read0.result, read1.result])
    run_pass(read_across_write, "builtin.module(func.func(cse))")

    # A potentially-aliasing Write on the default resource blocks Read CSE.
    # CHECK: CSE read across write count: 2
    print(
        "CSE read across write count:",
        count_ops(read_across_write, ReadOp),
    )


# CHECK-LABEL: TEST: testRecursiveMemoryEffectsCSE
@run
def testRecursiveMemoryEffectsCSE():
    i32 = ir.IntegerType.get_signless(32)
    with function_module(results=[i32, i32, i32, i32]) as (
        recursive_cse,
        _,
    ):
        region_ops = [
            RegionOp(i32),
            RegionOp(i32),
            RecursiveRegionOp(i32),
            RecursiveRegionOp(i32),
        ]
        for region_op in region_ops:
            region_op.body.blocks.append()
            with ir.InsertionPoint(region_op.body.blocks[0]):
                NoEffectOp()
        func.ReturnOp([region_op.result for region_op in region_ops])
    run_pass(recursive_cse, "builtin.module(func.func(cse))")

    # An op without RecursiveMemoryEffects has unknown effects and cannot be
    # CSE'd. The trait makes the other op's empty nested effects visible.
    # CHECK: CSE non-recursive region count: 2
    # CHECK: CSE recursive region count: 1
    print(
        "CSE non-recursive region count:",
        count_ops(recursive_cse, RegionOp),
    )
    print(
        "CSE recursive region count:",
        count_ops(recursive_cse, RecursiveRegionOp),
    )


# CHECK-LABEL: TEST: testMemoryEffectsDCE
@run
def testMemoryEffectsDCE():
    i32 = ir.IntegerType.get_signless(32)
    with function_module() as (dead_code, _):
        NoEffectOp()
        ReadDeadOp()
        WriteDeadOp()
        FreeDeadOp()
        AllocateDeadOp()
        AllocateResultOp(i32)
        func.ReturnOp([])
    run_pass(dead_code, "builtin.module(func.func(trivial-dce))")

    # Empty and Read-only effect lists are dead. Write, Free and untargeted
    # Allocate effects are observable. An Allocate targeting its own unused
    # result is dead.
    # CHECK: DCE no effect count: 0
    # CHECK: DCE read count: 0
    # CHECK: DCE write count: 1
    # CHECK: DCE free count: 1
    # CHECK: DCE untargeted allocate count: 1
    # CHECK: DCE result allocate count: 0
    print("DCE no effect count:", count_ops(dead_code, NoEffectOp))
    print("DCE read count:", count_ops(dead_code, ReadDeadOp))
    print("DCE write count:", count_ops(dead_code, WriteDeadOp))
    print("DCE free count:", count_ops(dead_code, FreeDeadOp))
    print(
        "DCE untargeted allocate count:",
        count_ops(dead_code, AllocateDeadOp),
    )
    print(
        "DCE result allocate count:",
        count_ops(dead_code, AllocateResultOp),
    )


# CHECK-LABEL: TEST: testRecursiveReadDCE
@run
def testRecursiveReadDCE():
    i32 = ir.IntegerType.get_signless(32)
    with function_module() as (recursive_read_dce, _):
        region_op = RegionOp(i32)
        region_op.body.blocks.append()
        with ir.InsertionPoint(region_op.body.blocks[0]):
            ReadDeadOp()

        recursive_region_op = RecursiveRegionOp(i32)
        recursive_region_op.body.blocks.append()
        with ir.InsertionPoint(recursive_region_op.body.blocks[0]):
            ReadDeadOp()
        func.ReturnOp([])
    run_pass(recursive_read_dce, "builtin.module(func.func(trivial-dce))")

    # The non-recursive op has unknown effects and remains. The recursive op is
    # removable because all nested effects are reads.
    # CHECK: DCE non-recursive read region count: 1
    # CHECK: DCE recursive read region count: 0
    print(
        "DCE non-recursive read region count:",
        count_ops(recursive_read_dce, RegionOp),
    )
    print(
        "DCE recursive read region count:",
        count_ops(recursive_read_dce, RecursiveRegionOp),
    )


# CHECK-LABEL: TEST: testRecursiveWriteDCE
@run
def testRecursiveWriteDCE():
    i32 = ir.IntegerType.get_signless(32)
    with function_module() as (recursive_write_dce, _):
        recursive_region_op = RecursiveRegionOp(i32)
        recursive_region_op.body.blocks.append()
        with ir.InsertionPoint(recursive_region_op.body.blocks[0]):
            WriteDeadOp()
        func.ReturnOp([])
    run_pass(recursive_write_dce, "builtin.module(func.func(trivial-dce))")

    # A nested Write remains observable through RecursiveMemoryEffects.
    # CHECK: DCE recursive write region count: 1
    # CHECK: DCE nested write count: 1
    print(
        "DCE recursive write region count:",
        count_ops(recursive_write_dce, RecursiveRegionOp),
    )
    print(
        "DCE nested write count:",
        count_ops(recursive_write_dce, WriteDeadOp),
    )


# CHECK-LABEL: TEST: testMemoryEffectTargets
@run
def testMemoryEffectTargets():
    i32 = ir.IntegerType.get_signless(32)
    with function_module() as (target_variants, _):
        block_argument_target = BlockArgumentTargetOp()
        block_argument_target.body.blocks.append(i32)
        SymbolTargetOp()
        func.ReturnOp([])
    run_pass(target_variants, "builtin.module(func.func(trivial-dce))")

    # These Read effects exercise BlockArgument and SymbolRefAttr targets and
    # remain removable by trivial-dce.
    # CHECK: invalid symbol target: target Attribute must be a SymbolRefAttr
    # CHECK: invalid parameters: parameters must be an Attribute or None
    # CHECK: invalid target: target must be an OpOperand, OpResult, BlockArgument, SymbolRefAttr, or None
    # CHECK: DCE block argument target count: 0
    # CHECK: DCE symbol target count: 0
    print(
        "DCE block argument target count:",
        count_ops(target_variants, BlockArgumentTargetOp),
    )
    print(
        "DCE symbol target count:",
        count_ops(target_variants, SymbolTargetOp),
    )
