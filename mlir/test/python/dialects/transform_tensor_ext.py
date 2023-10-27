# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import tensor


def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            sequence = transform.SequenceOp(
                transform.FailurePropagationMode.Propagate,
                [],
                transform.AnyOpType.get(),
            )
            with InsertionPoint(sequence.body):
                f(sequence.bodyTarget)
                transform.YieldOp()
        print(module)
    return f


@run
def testMakeLoopIndependentOpCompact(target):
    tensor.MakeLoopIndependentOp(target, 4)
    # CHECK-LABEL: TEST: testMakeLoopIndependentOpCompact
    # CHECK: = transform.tensor.make_loop_independent
    # CHECK-SAME: num_loops = 4 : i64
    # CHECK-SAME: (!transform.any_op) -> !transform.any_op


@run
def testMakeLoopIndependentOpTyped(target):
    tensor.MakeLoopIndependentOp(transform.OperationType.get("test.dummy"), target, 4)
    # CHECK-LABEL: TEST: testMakeLoopIndependentOpTyped
    # CHECK: = transform.tensor.make_loop_independent
    # CHECK-SAME: num_loops = 4 : i64
    # CHECK-SAME: (!transform.any_op) -> !transform.op<"test.dummy">
