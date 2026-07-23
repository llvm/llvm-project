# RUN: env PYTHONUNBUFFERED=1 %PYTHON %s 2>&1 | FileCheck %s

from typing import Any

from mlir import ir
from mlir.dialects import ext, func
from mlir.passmanager import PassManager


class MemoryEffectsTest(ext.Dialect, name="memory_effects_test"):
    pass


class NoEffectModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op, effects):
        pass


class ReadModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op, effects):
        effects.append(
            ir.MemoryEffect.Read,
            op.op_operands[0],
            parameters=ir.StringAttr.get("read parameter"),
            stage=1,
            effect_on_full_region=True,
            resource=ir.SideEffectResource.Default,
        )


class ReadDeadModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op, effects):
        effects.append(ir.MemoryEffect.Read)


class WriteModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op, effects):
        effects.append(ir.MemoryEffect.Write)


class FreeModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op, effects):
        effects.append(ir.MemoryEffect.Free)


class AllocateModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op, effects):
        effects.append(ir.MemoryEffect.Allocate)


class AllocateResultModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op, effects):
        effects.append(ir.MemoryEffect.Allocate, op.results[0])


class BlockArgumentTargetModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op, effects):
        effects.append(ir.MemoryEffect.Read, op.regions[0].blocks[0].arguments[0])


class SymbolTargetModel(ir.MemoryEffectsOpInterface):
    @staticmethod
    def get_effects(op, effects):
        try:
            effects.append(ir.MemoryEffect.Read, ir.StringAttr.get("not a symbol"))
        except TypeError as error:
            print("invalid symbol target:", error)
        try:
            effects.append(ir.MemoryEffect.Read, parameters=42)
        except TypeError as error:
            print("invalid parameters:", error)
        try:
            effects.append(ir.MemoryEffect.Read, 42)
        except TypeError as error:
            print("invalid target:", error)
        effects.append(
            ir.MemoryEffect.Read,
            ir.FlatSymbolRefAttr.get("global"),
            parameters=ir.StringAttr.get("symbol parameter"),
            stage=2,
            effect_on_full_region=True,
        )


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


def run_pass(source, pipeline):
    module = ir.Module.parse(source)
    PassManager.parse(pipeline).run(module.operation)
    return str(module)


with ir.Context(), ir.Location.unknown():
    MemoryEffectsTest.load()

    # CHECK: memory effect properties: True True True True
    print(
        "memory effect properties:",
        isinstance(ir.MemoryEffect.Allocate, ir.MemoryEffect),
        isinstance(ir.MemoryEffect.Free, ir.MemoryEffect),
        isinstance(ir.MemoryEffect.Read, ir.MemoryEffect),
        isinstance(ir.MemoryEffect.Write, ir.MemoryEffect),
    )
    # CHECK: default resource property: True
    print(
        "default resource property:",
        isinstance(ir.SideEffectResource.Default, ir.SideEffectResource),
    )

    read_cse = run_pass(
        """
        module {
          func.func @test(%arg0: i32) -> (i32, i32) {
            %0 = "memory_effects_test.read"(%arg0) : (i32) -> i32
            %1 = "memory_effects_test.read"(%arg0) : (i32) -> i32
            return %0, %1 : i32, i32
          }
        }
        """,
        "builtin.module(func.func(cse))",
    )
    # A single Read effect remains CSE-eligible.
    # CHECK: CSE read count: 1
    print("CSE read count:", read_cse.count('"memory_effects_test.read"'))

    write_cse = run_pass(
        """
        module {
          func.func @test(%arg0: i32) -> (i32, i32) {
            %0 = "memory_effects_test.write"(%arg0) : (i32) -> i32
            %1 = "memory_effects_test.write"(%arg0) : (i32) -> i32
            return %0, %1 : i32, i32
          }
        }
        """,
        "builtin.module(func.func(cse))",
    )
    # Writes cannot be CSE'd.
    # CHECK: CSE write count: 2
    print("CSE write count:", write_cse.count('"memory_effects_test.write"'))

    read_across_write = run_pass(
        """
        module {
          func.func @test(%arg0: i32) -> (i32, i32) {
            %0 = "memory_effects_test.read"(%arg0) : (i32) -> i32
            "memory_effects_test.write_barrier"(%arg0) : (i32) -> ()
            %1 = "memory_effects_test.read"(%arg0) : (i32) -> i32
            return %0, %1 : i32, i32
          }
        }
        """,
        "builtin.module(func.func(cse))",
    )
    # A potentially-aliasing Write on the default resource blocks Read CSE.
    # CHECK: CSE read across write count: 2
    print(
        "CSE read across write count:",
        read_across_write.count('"memory_effects_test.read"'),
    )

    dead_code = run_pass(
        """
        module {
          func.func @test() {
            "memory_effects_test.no_effect"() : () -> ()
            "memory_effects_test.read_dead"() : () -> ()
            "memory_effects_test.write_dead"() : () -> ()
            "memory_effects_test.free_dead"() : () -> ()
            "memory_effects_test.allocate_dead"() : () -> ()
            %0 = "memory_effects_test.allocate_result"() : () -> i32
            return
          }
        }
        """,
        "builtin.module(func.func(trivial-dce))",
    )
    # Empty and Read-only effect lists are dead. Write, Free and untargeted
    # Allocate effects are observable. An Allocate targeting its own unused
    # result is dead.
    # CHECK: DCE no effect count: 0
    # CHECK: DCE read count: 0
    # CHECK: DCE write count: 1
    # CHECK: DCE free count: 1
    # CHECK: DCE untargeted allocate count: 1
    # CHECK: DCE result allocate count: 0
    print("DCE no effect count:", dead_code.count('"memory_effects_test.no_effect"'))
    print("DCE read count:", dead_code.count('"memory_effects_test.read_dead"'))
    print("DCE write count:", dead_code.count('"memory_effects_test.write_dead"'))
    print("DCE free count:", dead_code.count('"memory_effects_test.free_dead"'))
    print(
        "DCE untargeted allocate count:",
        dead_code.count('"memory_effects_test.allocate_dead"'),
    )
    print(
        "DCE result allocate count:",
        dead_code.count('"memory_effects_test.allocate_result"'),
    )

    target_variants = run_pass(
        """
        module {
          func.func @test() {
            "memory_effects_test.block_argument_target"() ({
            ^bb0(%arg0: i32):
            }) : () -> ()
            "memory_effects_test.symbol_target"() : () -> ()
            return
          }
        }
        """,
        "builtin.module(func.func(trivial-dce))",
    )
    # These Read effects exercise BlockArgument and SymbolRefAttr targets and
    # remain removable by trivial-dce.
    # CHECK: invalid symbol target: target Attribute must be a SymbolRefAttr
    # CHECK: invalid parameters: parameters must be an Attribute or None
    # CHECK: invalid target: target must be an OpOperand, OpResult, BlockArgument, SymbolRefAttr, or None
    # CHECK: DCE block argument target count: 0
    # CHECK: DCE symbol target count: 0
    print(
        "DCE block argument target count:",
        target_variants.count('"memory_effects_test.block_argument_target"'),
    )
    print(
        "DCE symbol target count:",
        target_variants.count('"memory_effects_test.symbol_target"'),
    )
