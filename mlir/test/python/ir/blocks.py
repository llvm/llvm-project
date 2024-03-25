# RUN: %PYTHON %s | FileCheck %s

import gc
import io
import itertools
from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import cf
from mlir.dialects import func


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


# CHECK-LABEL: TEST: testBlockCreation
# CHECK: func @test(%[[ARG0:.*]]: i32 loc("arg0"), %[[ARG1:.*]]: i16 loc("arg1"))
# CHECK:   cf.br ^bb1(%[[ARG1]] : i16)
# CHECK: ^bb1(%[[PHI0:.*]]: i16 loc("middle")):
# CHECK:   cf.br ^bb2(%[[ARG0]] : i32)
# CHECK: ^bb2(%[[PHI1:.*]]: i32 loc("successor")):
# CHECK:   return
@run
def testBlockCreation():
    with Context() as ctx, Location.unknown():
        module = builtin.ModuleOp()
        with InsertionPoint(module.body):
            f_type = FunctionType.get(
                [IntegerType.get_signless(32), IntegerType.get_signless(16)], []
            )
            f_op = func.FuncOp("test", f_type)
            entry_block = f_op.add_entry_block(
                [Location.name("arg0"), Location.name("arg1")]
            )
            i32_arg, i16_arg = entry_block.arguments
            successor_block = entry_block.create_after(
                i32_arg.type, arg_locs=[Location.name("successor")]
            )
            with InsertionPoint(successor_block) as successor_ip:
                assert successor_ip.block == successor_block
                func.ReturnOp([])
            middle_block = successor_block.create_before(
                i16_arg.type, arg_locs=[Location.name("middle")]
            )

            with InsertionPoint(entry_block) as entry_ip:
                assert entry_ip.block == entry_block
                cf.BranchOp([i16_arg], dest=middle_block)

            with InsertionPoint(middle_block) as middle_ip:
                assert middle_ip.block == middle_block
                cf.BranchOp([i32_arg], dest=successor_block)
        module.print(enable_debug_info=True)
        # Ensure region back references are coherent.
        assert entry_block.region == middle_block.region == successor_block.region


# CHECK-LABEL: TEST: testBlockCreationArgLocs
@run
def testBlockCreationArgLocs():
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        f32 = F32Type.get()
        op = Operation.create("test", regions=1, loc=Location.unknown())
        blocks = op.regions[0].blocks

        with Location.name("default_loc"):
            blocks.append(f32)
        blocks.append()
        # CHECK:      ^bb0(%{{.+}}: f32 loc("default_loc")):
        # CHECK-NEXT: ^bb1:
        op.print(enable_debug_info=True)

        try:
            blocks.append(f32)
        except RuntimeError as err:
            # CHECK: Missing loc: An MLIR function requires a Location but none was provided
            print("Missing loc:", err)

        try:
            blocks.append(f32, f32, arg_locs=[Location.unknown()])
        except ValueError as err:
            # CHECK: Wrong loc count: Expected 2 locations, got: 1
            print("Wrong loc count:", err)


# CHECK-LABEL: TEST: testFirstBlockCreation
# CHECK: func @test(%{{.*}}: f32 loc("arg_loc"))
# CHECK:   return
@run
def testFirstBlockCreation():
    with Context() as ctx, Location.unknown():
        module = builtin.ModuleOp()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            f = func.FuncOp("test", ([f32], []))
            entry_block = Block.create_at_start(
                f.operation.regions[0], [f32], [Location.name("arg_loc")]
            )
            with InsertionPoint(entry_block):
                func.ReturnOp([])

        module.print(enable_debug_info=True)
        assert module.verify()
        assert f.body.blocks[0] == entry_block


# CHECK-LABEL: TEST: testBlockMove
# CHECK:  %0 = "realop"() ({
# CHECK:  ^bb0([[ARG0:%.+]]: f32):
# CHECK:    "ret"([[ARG0]]) : (f32) -> ()
# CHECK:  }) : () -> f32
@run
def testBlockMove():
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            dummy = Operation.create("dummy", regions=1)
            block = Block.create_at_start(dummy.operation.regions[0], [f32])
            with InsertionPoint(block):
                ret_op = Operation.create("ret", operands=[block.arguments[0]])
            realop = Operation.create(
                "realop", results=[r.type for r in ret_op.operands], regions=1
            )
            block.append_to(realop.operation.regions[0])
            dummy.operation.erase()
        print(module)


# CHECK-LABEL: TEST: testBlockHash
@run
def testBlockHash():
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            dummy = Operation.create("dummy", regions=1)
            block1 = Block.create_at_start(dummy.operation.regions[0], [f32])
            block2 = Block.create_at_start(dummy.operation.regions[0], [f32])
            assert hash(block1) != hash(block2)
