# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects import pdl
from mlir.dialects.transform import loop


def run(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f()
        print(module)
    return f


@run
def loopOutline():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("scf.for"),
    )
    with InsertionPoint(sequence.body):
        loop.LoopOutlineOp(
            transform.AnyOpType.get(),
            transform.AnyOpType.get(),
            sequence.bodyTarget,
            func_name="foo",
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: loopOutline
    # CHECK: = transform.loop.outline %
    # CHECK: func_name = "foo"


@run
def loopPeel():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("scf.for"),
    )
    with InsertionPoint(sequence.body):
        loop.LoopPeelOp(transform.AnyOpType.get(), transform.AnyOpType.get(), sequence.bodyTarget)
        transform.YieldOp()
    # CHECK-LABEL: TEST: loopPeel
    # CHECK: = transform.loop.peel %

@run
def loopPeel_peel_front():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("scf.for"),
    )
    with InsertionPoint(sequence.body):
        loop.LoopPeelOp(
            transform.AnyOpType.get(),
            transform.AnyOpType.get(),
            sequence.bodyTarget,
            peel_front=True,
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: loopPeel_peel_front
    # CHECK: = transform.loop.peel %[[ARG0:.*]] {peel_front = true}


@run
def loopPipeline():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("scf.for"),
    )
    with InsertionPoint(sequence.body):
        loop.LoopPipelineOp(
            pdl.OperationType.get(), sequence.bodyTarget, iteration_interval=3
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: loopPipeline
    # CHECK: = transform.loop.pipeline %
    # CHECK-DAG: iteration_interval = 3
    # (read_latency has default value and is not printed)


@run
def loopUnroll():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.OperationType.get("scf.for"),
    )
    with InsertionPoint(sequence.body):
        loop.LoopUnrollOp(sequence.bodyTarget, factor=42)
        transform.YieldOp()
    # CHECK-LABEL: TEST: loopUnroll
    # CHECK: transform.loop.unroll %
    # CHECK: factor = 42
