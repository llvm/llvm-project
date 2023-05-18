# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import bufferization
from mlir.dialects.bufferization import LayoutMapOption


def run(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f()
        print(module)
    return f


@run
def testEmptyTensorToAllocTensorOpCompact():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("tensor.empty"),
    )
    with InsertionPoint(sequence.body):
        bufferization.EmptyTensorToAllocTensorOp(sequence.bodyTarget)
        transform.YieldOp()
    # CHECK-LABEL: TEST: testEmptyTensorToAllocTensorOpCompact
    # CHECK: = transform.bufferization.empty_tensor_to_alloc_tensor
    # CHECK-SAME: (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">


@run
def testEmptyTensorToAllocTensorOpTyped():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("tensor.empty"),
    )
    with InsertionPoint(sequence.body):
        bufferization.EmptyTensorToAllocTensorOp(
            transform.OperationType.get("bufferization.alloc_tensor"),
            sequence.bodyTarget,
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testEmptyTensorToAllocTensorOpTyped
    # CHECK: = transform.bufferization.empty_tensor_to_alloc_tensor
    # CHECK-SAME: (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">


@run
def testOneShotBufferizeOpCompact():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        bufferization.OneShotBufferizeOp(sequence.bodyTarget)
        transform.YieldOp()
    # CHECK-LABEL: TEST: testOneShotBufferizeOpCompact
    # CHECK: = transform.bufferization.one_shot_bufferize
    # CHECK-SAME: (!transform.any_op) -> !transform.any_op


@run
def testOneShotBufferizeOpTyped():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        bufferization.OneShotBufferizeOp(
            transform.OperationType.get("test.dummy"),
            sequence.bodyTarget,
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testOneShotBufferizeOpTyped
    # CHECK: = transform.bufferization.one_shot_bufferize
    # CHECK-SAME: (!transform.any_op) -> !transform.op<"test.dummy">


@run
def testOneShotBufferizeOpAttributes():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        bufferization.OneShotBufferizeOp(
            sequence.bodyTarget,
            allow_return_allocs_from_loops=True,
            allow_unknown_ops=True,
            bufferize_function_boundaries=True,
            function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
            memcpy_op="linalg.copy",
            print_conflicts=True,
            test_analysis_only=True,
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testOneShotBufferizeOpAttributes
    # CHECK: = transform.bufferization.one_shot_bufferize
    # CHECK-SAME: layout{IdentityLayoutMap}
    # CHECK-SAME: allow_return_allocs_from_loops = true
    # CHECK-SAME: allow_unknown_ops = true
    # CHECK-SAME: bufferize_function_boundaries = true
    # CHECK-SAME: memcpy_op = "linalg.copy"
    # CHECK-SAME: print_conflicts = true
    # CHECK-SAME: test_analysis_only = true
    # CHECK-SAME: (!transform.any_op) -> !transform.any_op
