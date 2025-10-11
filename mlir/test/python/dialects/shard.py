# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import shard
from mlir.dialects import func


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


# CHECK-LABEL: TEST: testShardGrid
@constructAndPrintInModule
def testShardGrid():
    # Test creating shard grids with different shapes
    grid2d = shard.GridOp("grid_2d", [2, 2])
    grid1d = shard.GridOp("grid_1d", [4])
    grid_dynamic = shard.GridOp("grid_dynamic", [2, -1])  # -1 for dynamic dimension

    # CHECK: "shard.grid"() <{shape = array<i64: 2, 2>, sym_name = "grid_2d"}> : () -> ()
    # CHECK: "shard.grid"() <{shape = array<i64: 4>, sym_name = "grid_1d"}> : () -> ()
    # CHECK: "shard.grid"() <{shape = array<i64: 2, -1>, sym_name = "grid_dynamic"}> : () -> ()


# CHECK-LABEL: TEST: testCollectiveOperations
@constructAndPrintInModule
def testCollectiveOperations():
    # Create grid and types
    grid = shard.GridOp("grid_2x2", [2, 2])
    i32 = IntegerType.get_signless(32)
    input_type = RankedTensorType.get([4, 2], i32)
    gather_result_type = RankedTensorType.get([4, 4], i32)

    # Create a function to hold the operations
    func_type = FunctionType.get([input_type], [input_type])
    test_func = func.FuncOp("test_collectives", func_type)

    with InsertionPoint(test_func.add_entry_block()):
        arg = test_func.entry_block.arguments[0]

        gather_op = shard.AllGatherOp(
            input=arg,
            grid=FlatSymbolRefAttr.get("grid_2x2"),
            grid_axes=ArrayAttr.get([IntegerAttr.get(i32, 1)]),
            gather_axis=IntegerAttr.get(i32, 1),
            result=gather_result_type,
        )

        reduce_op = shard.AllReduceOp(
            input=arg,
            grid=FlatSymbolRefAttr.get("grid_2x2"),
            reduction=shard.ReductionKind.Sum,
            result=input_type,
        )

        func.ReturnOp([reduce_op])

    # CHECK: "shard.grid"() <{shape = array<i64: 2, 2>, sym_name = "grid_2x2"}> : () -> ()
    # CHECK: "func.func"() <{function_type = (tensor<4x2xi32>) -> tensor<4x2xi32>, sym_name = "test_collectives"}>
    # CHECK: "shard.all_gather"({{.*}}) <{gather_axis = 1 : i32, grid = @grid_2x2}> : (tensor<4x2xi32>) -> tensor<4x4xi32>
    # CHECK: "shard.all_reduce"({{.*}}) <{grid = @grid_2x2, {{.*}} reduction = #shard<partial sum>}> : (tensor<4x2xi32>) -> tensor<4x2xi32>
