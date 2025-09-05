# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import tune, debug


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


# CHECK-LABEL: TEST: testKnobOp
@run
def testKnobOp(target):
    any_param = transform.AnyParamType.get()

    # CHECK: %[[HEADS_OR_TAILS:.*]] = transform.tune.knob<"coin"> options = [true, false] -> !transform.any_param
    heads_or_tails = tune.KnobOp(
        result=any_param, name=StringAttr.get("coin"), options=[True, False]
    )
    # CHECK: transform.tune.knob<"animal"> options = ["cat", "dog", unit] -> !transform.any_param
    tune.KnobOp(any_param, name="animal", options=["cat", "dog", UnitAttr.get()])
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
        any_param, name="animal", options=["cat", "dog", UnitAttr.get()], selected="dog"
    )
    # CHECK: transform.tune.knob<"tile_size"> = 8 : i64 from options = [2, 4, 8, 16, 24, 32] -> !transform.any_param
    tune.KnobOp(any_param, "tile_size", [2, 4, 8, 16, 24, 32], selected=8)
    # CHECK: transform.tune.knob<"magic_value"> = 2.500000e+00 : f64 from options = [2.000000e+00, 2.250000e+00, 2.500000e+00, 2.750000e+00, 3.000000e+00] -> !transform.any_param
    tune.knob(any_param, "magic_value", [2.0, 2.25, 2.5, 2.75, 3.0], selected=2.5)

    # CHECK: transform.debug.emit_param_as_remark %[[HEADS]]
    debug.emit_param_as_remark(heads)

    # CHECK: transform.tune.knob<"range_as_a_dict"> = 4 : i64 from options = {start = 2 : i64, step = 2 : i64, stop = 16 : i64} -> !transform.any_param
    # NB: Membership of `selected` in non-ArrayAttr `options` is _not_ verified.
    i64 = IntegerType.get_signless(64)
    tune.knob(
        any_param,
        "range_as_a_dict",
        DictAttr.get(
            {
                "start": IntegerAttr.get(i64, 2),
                "stop": IntegerAttr.get(i64, 16),
                "step": IntegerAttr.get(i64, 2),
            }
        ),
        selected=4,
    )
