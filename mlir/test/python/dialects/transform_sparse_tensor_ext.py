# RUN: %PYTHON %s | FileCheck %s
# RUN: %PYTHON -m mypy %s --config-file %mlir_src_root/test/python/mypy.ini

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import sparse_tensor


def run(f):
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
        print("\nTEST:", f.__name__)
        print(module)
    return f


@run
def testMatchSparseInOut(target):
    sparse_tensor.MatchSparseInOut(transform.AnyOpType.get(), target)
    # CHECK-LABEL: TEST: testMatchSparseInOut
    # CHECK:       transform.sequence
    # CHECK-NEXT:  ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
    # CHECK-NEXT:    transform.sparse_tensor.match.sparse_inout %[[ARG0]]
