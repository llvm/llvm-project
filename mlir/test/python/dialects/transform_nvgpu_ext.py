# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import nvgpu


def run(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f()
        print(module)
    return f


@run
def testCreateAsyncGroups():
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        nvgpu.CreateAsyncGroupsOp(transform.AnyOpType.get(), sequence.bodyTarget)
        transform.YieldOp()
    # CHECK-LABEL: TEST: testCreateAsyncGroups
    # CHECK: transform.nvgpu.create_async_groups
