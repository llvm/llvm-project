# RUN: %PYTHON %s | FileCheck %s


from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import memref


def run(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f()
        print(module)
    return f


@run
def testMemRefMultiBufferOpCompact():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("memref.alloc"),
    )
    with InsertionPoint(sequence.body):
        memref.MemRefMultiBufferOp(sequence.bodyTarget, 4)
        transform.YieldOp()
    # CHECK-LABEL: TEST: testMemRefMultiBufferOpCompact
    # CHECK: = transform.memref.multibuffer
    # CHECK-SAME: factor = 4 : i64
    # CHECK-SAME: (!transform.op<"memref.alloc">) -> !transform.any_op


@run
def testMemRefMultiBufferOpTyped():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("memref.alloc"),
    )
    with InsertionPoint(sequence.body):
        memref.MemRefMultiBufferOp(
            transform.OperationType.get("memref.alloc"), sequence.bodyTarget, 4
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testMemRefMultiBufferOpTyped
    # CHECK: = transform.memref.multibuffer
    # CHECK-SAME: factor = 4 : i64
    # CHECK-SAME: (!transform.op<"memref.alloc">) -> !transform.op<"memref.alloc">


@run
def testMemRefMultiBufferOpAttributes():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("memref.alloc"),
    )
    with InsertionPoint(sequence.body):
        memref.MemRefMultiBufferOp(sequence.bodyTarget, 4, skip_analysis=True)
        transform.YieldOp()
    # CHECK-LABEL: TEST: testMemRefMultiBufferOpAttributes
    # CHECK: = transform.memref.multibuffer
    # CHECK-SAME: factor = 4 : i64
    # CHECK-SAME: skip_analysis
    # CHECK-SAME: (!transform.op<"memref.alloc">) -> !transform.any_op
