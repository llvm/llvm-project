# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import debug


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
def testDebugEmitParamAsRemark(target):
    i0 = IntegerAttr.get(IntegerType.get_signless(32), 0)
    i0_param = transform.ParamConstantOp(transform.AnyParamType.get(), i0)
    debug.emit_param_as_remark(i0_param)
    debug.emit_param_as_remark(i0_param, anchor=target, message="some text")
    # CHECK-LABEL: TEST: testDebugEmitParamAsRemark
    # CHECK: ^{{.*}}(%[[ARG0:.+]]: !transform.any_op):
    # CHECK: %[[PARAM:.*]] = transform.param.constant
    # CHECK: transform.debug.emit_param_as_remark %[[PARAM]]
    # CHECK: transform.debug.emit_param_as_remark %[[PARAM]]
    # CHECK-SAME: "some text"
    # CHECK-SAME: at %[[ARG0]]


@run
def testDebugEmitRemarkAtOp(target):
    debug.emit_remark_at(target, "some text")
    # CHECK-LABEL: TEST: testDebugEmitRemarkAtOp
    # CHECK: ^{{.*}}(%[[ARG0:.+]]: !transform.any_op):
    # CHECK: transform.debug.emit_remark_at %[[ARG0]], "some text"
