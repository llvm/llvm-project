# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform, smt
from mlir.dialects.transform import smt as transform_smt


def run(f):
    print("\nTEST:", f.__name__)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            sequence = transform.SequenceOp(
                transform.FailurePropagationMode.Propagate,
                [],
                transform.AnyOpType.get(),
            )
            with ir.InsertionPoint(sequence.body):
                f(sequence.bodyTarget)
                transform.YieldOp()
        print(module)
    return f


# CHECK-LABEL: TEST: testConstrainParamsOp
@run
def testConstrainParamsOp(target):
    dummy_value = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 42)
    # CHECK: %[[PARAM_AS_PARAM:.*]] = transform.param.constant
    symbolic_value = transform.ParamConstantOp(
        transform.AnyParamType.get(), dummy_value
    )
    # CHECK: transform.smt.constrain_params(%[[PARAM_AS_PARAM]])
    constrain_params = transform_smt.ConstrainParamsOp(
        [symbolic_value], [smt.IntType.get()]
    )
    # CHECK-NEXT: ^bb{{.*}}(%[[PARAM_AS_SMT_SYMB:.*]]: !smt.int):
    with ir.InsertionPoint(constrain_params.body):
        # CHECK: %[[C0:.*]] = smt.int.constant 0
        c0 = smt.IntConstantOp(ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0))
        # CHECK: %[[C43:.*]] = smt.int.constant 43
        c43 = smt.IntConstantOp(ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 43))
        # CHECK: %[[LB:.*]] = smt.int.cmp le %[[C0]], %[[PARAM_AS_SMT_SYMB]]
        lb = smt.IntCmpOp(smt.IntPredicate.le, c0, constrain_params.body.arguments[0])
        # CHECK: %[[UB:.*]] = smt.int.cmp le %[[PARAM_AS_SMT_SYMB]], %[[C43]]
        ub = smt.IntCmpOp(smt.IntPredicate.le, constrain_params.body.arguments[0], c43)
        # CHECK: %[[BOUNDED:.*]] = smt.and %[[LB]], %[[UB]]
        bounded = smt.AndOp([lb, ub])
        # CHECK: smt.assert %[[BOUNDED:.*]]
        smt.AssertOp(bounded)
