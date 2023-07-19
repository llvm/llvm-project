# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import gpu


def run(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f()
        print(module)
    return f


@run
def testMapForallToBlocksCompact():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        gpu.MapForallToBlocks(sequence.bodyTarget)
        transform.YieldOp()
    # CHECK-LABEL: TEST: testMapForallToBlocksCompact
    # CHECK: = transform.gpu.map_forall_to_blocks
    # CHECK-NOT: grid_dims
    # CHECK-SAME: (!transform.any_op) -> !transform.any_op
    # CHECK-NOT: grid_dims


@run
def testMapForallToBlocksTyped():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        gpu.MapForallToBlocks(
            transform.OperationType.get("test.dummy"), sequence.bodyTarget
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testMapForallToBlocksTyped
    # CHECK: = transform.gpu.map_forall_to_blocks
    # CHECK-SAME: (!transform.any_op) -> !transform.op<"test.dummy">


@run
def testMapForallToBlocksGridDims():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.PROPAGATE, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        gpu.MapForallToBlocks(sequence.bodyTarget, grid_dims=[4, 2])
        transform.YieldOp()
    # CHECK-LABEL: TEST: testMapForallToBlocksGridDims
    # CHECK: = transform.gpu.map_forall_to_blocks
    # CHECK-SAME: grid_dims = [4, 2]
    # CHECK-SAME: (!transform.any_op) -> !transform.any_op
