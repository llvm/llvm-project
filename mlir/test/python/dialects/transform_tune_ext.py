# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import tune, debug


def run(f):
    print("\n// TEST:", f.__name__)
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


# CHECK-LABEL: TEST: testKnobOp
@run
def testKnobOp(target):
    any_param = transform.AnyParamType.get()

    # CHECK: %[[HEADS_OR_TAILS:.*]] = transform.tune.knob<"coin"> options = [true, false] -> !transform.any_param
    heads_or_tails = tune.KnobOp(
        result=any_param, name=ir.StringAttr.get("coin"), options=[True, False]
    )
    # CHECK: transform.tune.knob<"animal"> options = ["cat", "dog", unit] -> !transform.any_param
    tune.KnobOp(any_param, name="animal", options=["cat", "dog", ir.UnitAttr.get()])
    # CHECK: transform.tune.knob<"tile_size"> options = [2, 4, 8, 16, 24, 32] -> !transform.any_param
    tune.KnobOp(any_param, "tile_size", [2, 4, 8, 16, 24, 32])
    # CHECK: transform.tune.knob<"magic_value"> options = [2.000000e+00, 2.250000e+00, 2.500000e+00, 2.750000e+00, 3.000000e+00] -> !transform.any_param
    tune.knob(any_param, "magic_value", [2.0, 2.25, 2.5, 2.75, 3.0])

    # CHECK: transform.debug.emit_param_as_remark %[[HEADS_OR_TAILS]]
    debug.emit_param_as_remark(heads_or_tails)

    # CHECK: %[[HEADS:.*]] = transform.tune.knob<"coin"> = true from options = [true, false] -> !transform.any_param
    heads = tune.KnobOp(any_param, "coin", options=[True, False], selected=True)
    # CHECK: transform.tune.knob<"animal"> = "dog" from options = ["cat", "dog", unit] -> !transform.any_param
    tune.KnobOp(
        any_param,
        name="animal",
        options=["cat", "dog", ir.UnitAttr.get()],
        selected="dog",
    )
    # CHECK: transform.tune.knob<"tile_size"> = 8 : i64 from options = [2, 4, 8, 16, 24, 32] -> !transform.any_param
    tune.KnobOp(any_param, "tile_size", [2, 4, 8, 16, 24, 32], selected=8)
    # CHECK: transform.tune.knob<"magic_value"> = 2.500000e+00 : f64 from options = [2.000000e+00, 2.250000e+00, 2.500000e+00, 2.750000e+00, 3.000000e+00] -> !transform.any_param
    tune.knob(any_param, "magic_value", [2.0, 2.25, 2.5, 2.75, 3.0], selected=2.5)

    # CHECK: transform.debug.emit_param_as_remark %[[HEADS]]
    debug.emit_param_as_remark(heads)

    # CHECK: transform.tune.knob<"range_as_a_dict"> = 4 : i64 from options = {start = 2 : i64, step = 2 : i64, stop = 16 : i64} -> !transform.any_param
    # NB: Membership of `selected` in non-ArrayAttr `options` is _not_ verified.
    i64 = ir.IntegerType.get_signless(64)
    tune.knob(
        any_param,
        "range_as_a_dict",
        ir.DictAttr.get(
            {
                "start": ir.IntegerAttr.get(i64, 2),
                "stop": ir.IntegerAttr.get(i64, 16),
                "step": ir.IntegerAttr.get(i64, 2),
            }
        ),
        selected=4,
    )


# CHECK-LABEL: TEST: testAlternativesOp
@run
def testAlternativesOp(target):
    any_param = transform.AnyParamType.get()

    # CHECK: %[[LEFT_OR_RIGHT_OUTCOME:.*]] = transform.tune.alternatives<"left_or_right"> -> !transform.any_param {
    left_or_right = tune.AlternativesOp(
        [transform.AnyParamType.get()], "left_or_right", 2
    )
    idx_for_left, idx_for_right = 0, 1
    with ir.InsertionPoint(left_or_right.alternatives[idx_for_left].blocks[0]):
        # CHECK: %[[C0:.*]] = transform.param.constant 0
        i32_0 = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
        c0 = transform.ParamConstantOp(transform.AnyParamType.get(), i32_0)
        # CHECK: transform.yield %[[C0]]
        transform.yield_(c0)
    # CHECK-NEXT: }, {
    with ir.InsertionPoint(left_or_right.alternatives[idx_for_right].blocks[0]):
        # CHECK: %[[C1:.*]] = transform.param.constant 1
        i32_1 = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 1)
        c1 = transform.ParamConstantOp(transform.AnyParamType.get(), i32_1)
        # CHECK: transform.yield %[[C1]]
        transform.yield_(c1)
    # CHECK-NEXT: }
    outcome_of_left_or_right_decision = left_or_right.results[0]

    # CHECK: transform.tune.alternatives<"fork_in_the_road"> selected_region = 0 -> !transform.any_param {
    fork_in_the_road = tune.AlternativesOp(
        [transform.AnyParamType.get()], "fork_in_the_road", 2, selected_region=0
    )
    with ir.InsertionPoint(fork_in_the_road.alternatives[idx_for_left].blocks[0]):
        # CHECK: %[[C0:.*]] = transform.param.constant 0
        i32_0 = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
        c0 = transform.ParamConstantOp(transform.AnyParamType.get(), i32_0)
        # CHECK: transform.yield %[[C0]]
        transform.yield_(c0)
    # CHECK-NEXT: }, {
    with ir.InsertionPoint(fork_in_the_road.alternatives[idx_for_right].blocks[0]):
        # CHECK: %[[C1:.*]] = transform.param.constant 1
        i32_1 = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 1)
        c1 = transform.ParamConstantOp(transform.AnyParamType.get(), i32_1)
        # CHECK: transform.yield %[[C1]]
        transform.yield_(c1)
    # CHECK-NEXT: }

    # CHECK: transform.tune.alternatives<"left_or_right_as_before"> selected_region = %[[LEFT_OR_RIGHT_OUTCOME]] : !transform.any_param {
    left_or_right_as_before = tune.AlternativesOp(
        [],
        "left_or_right_as_before",
        2,
        selected_region=outcome_of_left_or_right_decision,
    )
    with ir.InsertionPoint(
        left_or_right_as_before.alternatives[idx_for_left].blocks[0]
    ):
        # CHECK: transform.param.constant 1337
        i32_1337 = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 1337)
        c1337 = transform.ParamConstantOp(transform.AnyParamType.get(), i32_1337)
        # CHECK: transform.debug.emit_param_as_remark
        debug.emit_param_as_remark(c1337)
        transform.yield_([])
    # CHECK-NEXT: }, {
    with ir.InsertionPoint(
        left_or_right_as_before.alternatives[idx_for_right].blocks[0]
    ):
        # CHECK: transform.param.constant 42
        i32_42 = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 42)
        c42 = transform.ParamConstantOp(transform.AnyParamType.get(), i32_42)
        # CHECK: transform.debug.emit_param_as_remark
        debug.emit_param_as_remark(c42)
        transform.yield_([])
    # CHECK-NEXT: }
