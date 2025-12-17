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
    module.operation.verify()
    return f


# CHECK-LABEL: TEST: testShardGrid
@constructAndPrintInModule
def testShardGrid():
    # Test creating shard grids with different shapes
    grid2d = shard.GridOp("grid_2d", [2, 2])
    grid1d = shard.GridOp("grid_1d", [4])

    # CHECK: shard.grid @grid_2d(shape = 2x2)
    # CHECK: shard.grid @grid_1d(shape = 4)


# CHECK-LABEL: TEST: testCollectiveOperations
@constructAndPrintInModule
def testCollectiveOperations():
    # Create grid and types
    grid_op = shard.GridOp("grid_2x2", [2, 2])
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()
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
            grid_axes=DenseI16ArrayAttr.get([1]),
            gather_axis=IntegerAttr.get(index_type, 1),
            result=gather_result_type,
        )

        reduce_op = shard.AllReduceOp(
            input=arg,
            grid=FlatSymbolRefAttr.get("grid_2x2"),
            reduction=shard.ReductionKind.Sum,
            result=input_type,
        )

        func.ReturnOp([reduce_op])

    # CHECK: shard.grid @grid_2x2(shape = 2x2)
    # CHECK: func.func @test_collectives(%arg0: tensor<4x2xi32>) -> tensor<4x2xi32>
    # CHECK: %all_gather = shard.all_gather %arg0 on @grid_2x2 grid_axes = [1] gather_axis = 1 : tensor<4x2xi32> -> tensor<4x4xi32>
    # CHECK: %all_reduce = shard.all_reduce %arg0 on @grid_2x2 : tensor<4x2xi32> -> tensor<4x2xi32>
