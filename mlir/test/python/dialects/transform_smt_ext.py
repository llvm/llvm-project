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
    c42_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 42)
    # CHECK: %[[PARAM_AS_PARAM:.*]] = transform.param.constant
    symbolic_value_as_param = transform.ParamConstantOp(
        transform.AnyParamType.get(), c42_attr
    )
    # CHECK: transform.smt.constrain_params(%[[PARAM_AS_PARAM]])
    constrain_params = transform_smt.ConstrainParamsOp(
        [], [symbolic_value_as_param], [smt.IntType.get()]
    )
    # CHECK-NEXT: ^bb{{.*}}(%[[PARAM_AS_SMT_SYMB:.*]]: !smt.int):
    with ir.InsertionPoint(constrain_params.body):
        symbolic_value_as_smt_var = constrain_params.body.arguments[0]
        # CHECK: %[[C0:.*]] = smt.int.constant 0
        c0 = smt.IntConstantOp(ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0))
        # CHECK: %[[C43:.*]] = smt.int.constant 43
        c43 = smt.IntConstantOp(ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 43))
        # CHECK: %[[LB:.*]] = smt.int.cmp le %[[C0]], %[[PARAM_AS_SMT_SYMB]]
        lb = smt.IntCmpOp(smt.IntPredicate.le, c0, symbolic_value_as_smt_var)
        # CHECK: %[[UB:.*]] = smt.int.cmp le %[[PARAM_AS_SMT_SYMB]], %[[C43]]
        ub = smt.IntCmpOp(smt.IntPredicate.le, symbolic_value_as_smt_var, c43)
        # CHECK: %[[BOUNDED:.*]] = smt.and %[[LB]], %[[UB]]
        bounded = smt.AndOp([lb, ub])
        # CHECK: smt.assert %[[BOUNDED:.*]]
        smt.AssertOp(bounded)
        smt.YieldOp([])

    # CHECK: transform.smt.constrain_params(%[[PARAM_AS_PARAM]])
    compute_with_params = transform_smt.ConstrainParamsOp(
        [transform.ParamType.get(ir.IntegerType.get_signless(32))],
        [symbolic_value_as_param],
        [smt.IntType.get()],
    )
    # CHECK-NEXT: ^bb{{.*}}(%[[SMT_SYMB:.*]]: !smt.int):
    with ir.InsertionPoint(compute_with_params.body):
        symbolic_value_as_smt_var = compute_with_params.body.arguments[0]
        # CHECK: %[[TWICE:.*]] = smt.int.add %[[SMT_SYMB]], %[[SMT_SYMB]]
        twice_symb = smt.IntAddOp(
            [symbolic_value_as_smt_var, symbolic_value_as_smt_var]
        )
        # CHECK: smt.yield %[[TWICE]]
        smt.YieldOp([twice_symb])
