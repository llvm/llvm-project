# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import pdl as transform_pdl


def run(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f(module)
        print(module)
    return f


@run
def testTypes(module: Module):
    # CHECK-LABEL: TEST: testTypes
    # CHECK: !transform.any_op
    any_op = transform.AnyOpType.get()
    print(any_op)

    # CHECK: !transform.any_param
    any_param = transform.AnyParamType.get()
    print(any_param)

    # CHECK: !transform.any_value
    any_value = transform.AnyValueType.get()
    print(any_value)

    # CHECK: !transform.op<"foo.bar">
    # CHECK: foo.bar
    concrete_op = transform.OperationType.get("foo.bar")
    print(concrete_op)
    print(concrete_op.operation_name)

    # CHECK: !transform.param<i32>
    # CHECK: i32
    param = transform.ParamType.get(IntegerType.get_signless(32))
    print(param)
    print(param.type)


@run
def testSequenceOp(module: Module):
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [transform.AnyOpType.get()],
        transform.AnyOpType.get(),
    )
    with InsertionPoint(sequence.body):
        transform.YieldOp([sequence.bodyTarget])
    # CHECK-LABEL: TEST: testSequenceOp
    # CHECK: = transform.sequence -> !transform.any_op failures(propagate) {
    # CHECK: ^{{.*}}(%[[ARG0:.+]]: !transform.any_op):
    # CHECK:   yield %[[ARG0]] : !transform.any_op
    # CHECK: }

@run
def testNestedSequenceOp(module: Module):
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        nested = transform.SequenceOp(
            transform.FailurePropagationMode.Propagate, [], sequence.bodyTarget
        )
        with InsertionPoint(nested.body):
            doubly_nested = transform.SequenceOp(
                transform.FailurePropagationMode.Propagate,
                [transform.AnyOpType.get()],
                nested.bodyTarget,
            )
            with InsertionPoint(doubly_nested.body):
                transform.YieldOp([doubly_nested.bodyTarget])
            transform.YieldOp()
        transform.YieldOp()
    # CHECK-LABEL: TEST: testNestedSequenceOp
    # CHECK: transform.sequence failures(propagate) {
    # CHECK: ^{{.*}}(%[[ARG0:.+]]: !transform.any_op):
    # CHECK:   sequence %[[ARG0]] : !transform.any_op failures(propagate) {
    # CHECK:   ^{{.*}}(%[[ARG1:.+]]: !transform.any_op):
    # CHECK:     = sequence %[[ARG1]] : !transform.any_op -> !transform.any_op failures(propagate) {
    # CHECK:     ^{{.*}}(%[[ARG2:.+]]: !transform.any_op):
    # CHECK:       yield %[[ARG2]] : !transform.any_op
    # CHECK:     }
    # CHECK:   }
    # CHECK: }


@run
def testSequenceOpWithExtras(module: Module):
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.AnyOpType.get(),
        [transform.AnyOpType.get(), transform.OperationType.get("foo.bar")],
    )
    with InsertionPoint(sequence.body):
        transform.YieldOp()
    # CHECK-LABEL: TEST: testSequenceOpWithExtras
    # CHECK: transform.sequence failures(propagate)
    # CHECK: ^{{.*}}(%{{.*}}: !transform.any_op, %{{.*}}: !transform.any_op, %{{.*}}: !transform.op<"foo.bar">):


@run
def testNestedSequenceOpWithExtras(module: Module):
  sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [],
        transform.AnyOpType.get(),
        [transform.AnyOpType.get(), transform.OperationType.get("foo.bar")],
    )
  with InsertionPoint(sequence.body):
    nested = transform.SequenceOp(
            transform.FailurePropagationMode.Propagate,
            [],
            sequence.bodyTarget,
            sequence.bodyExtraArgs,
        )
    with InsertionPoint(nested.body):
      transform.YieldOp()
    transform.YieldOp()
  # CHECK-LABEL: TEST: testNestedSequenceOpWithExtras
  # CHECK: transform.sequence failures(propagate)
  # CHECK: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op, %[[ARG1:.*]]: !transform.any_op, %[[ARG2:.*]]: !transform.op<"foo.bar">):
  # CHECK:   sequence %[[ARG0]], %[[ARG1]], %[[ARG2]] : (!transform.any_op, !transform.any_op, !transform.op<"foo.bar">)


@run
def testTransformPDLOps(module: Module):
  withPdl = transform_pdl.WithPDLPatternsOp(transform.AnyOpType.get())
  with InsertionPoint(withPdl.body):
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate,
        [transform.AnyOpType.get()],
        withPdl.bodyTarget,
    )
    with InsertionPoint(sequence.body):
      match = transform_pdl.PDLMatchOp(
          transform.AnyOpType.get(), sequence.bodyTarget, "pdl_matcher"
      )
      transform.YieldOp(match)
  # CHECK-LABEL: TEST: testTransformPDLOps
  # CHECK: transform.with_pdl_patterns {
  # CHECK: ^{{.*}}(%[[ARG0:.+]]: !transform.any_op):
  # CHECK:   = sequence %[[ARG0]] : !transform.any_op -> !transform.any_op failures(propagate) {
  # CHECK:   ^{{.*}}(%[[ARG1:.+]]: !transform.any_op):
  # CHECK:     %[[RES:.+]] = pdl_match @pdl_matcher in %[[ARG1]]
  # CHECK:     yield %[[RES]] : !transform.any_op
  # CHECK:   }
  # CHECK: }


@run
def testNamedSequenceOp(module: Module):
    module.operation.attributes["transform.with_named_sequence"] = UnitAttr.get()
    named_sequence = transform.NamedSequenceOp(
        "__transform_main",
        [transform.AnyOpType.get()],
        [transform.AnyOpType.get()],
        arg_attrs = [{"transform.consumed": UnitAttr.get()}])
    with InsertionPoint(named_sequence.body):
        transform.YieldOp([named_sequence.bodyTarget])
    # CHECK-LABEL: TEST: testNamedSequenceOp
    # CHECK: module attributes {transform.with_named_sequence} {
    # CHECK: transform.named_sequence @__transform_main(%[[ARG0:.+]]: !transform.any_op {transform.consumed}) -> !transform.any_op {
    # CHECK:   yield %[[ARG0]] : !transform.any_op


@run
def testGetParentOp(module: Module):
  sequence = transform.SequenceOp(
      transform.FailurePropagationMode.Propagate, [], transform.AnyOpType.get()
  )
  with InsertionPoint(sequence.body):
    transform.GetParentOp(
        transform.AnyOpType.get(),
        sequence.bodyTarget,
        isolated_from_above=True,
        nth_parent=2,
    )
    transform.YieldOp()
  # CHECK-LABEL: TEST: testGetParentOp
  # CHECK: transform.sequence
  # CHECK: ^{{.*}}(%[[ARG1:.+]]: !transform.any_op):
  # CHECK:   = get_parent_op %[[ARG1]] {isolated_from_above, nth_parent = 2 : i64}


@run
def testMergeHandlesOp(module: Module):
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        transform.MergeHandlesOp([sequence.bodyTarget])
        transform.YieldOp()
    # CHECK-LABEL: TEST: testMergeHandlesOp
    # CHECK: transform.sequence
    # CHECK: ^{{.*}}(%[[ARG1:.+]]: !transform.any_op):
    # CHECK:   = merge_handles %[[ARG1]]


@run
def testApplyPatternsOpCompact(module: Module):
  sequence = transform.SequenceOp(
      transform.FailurePropagationMode.Propagate, [], transform.AnyOpType.get()
  )
  with InsertionPoint(sequence.body):
    with InsertionPoint(transform.ApplyPatternsOp(sequence.bodyTarget).patterns):
      transform.ApplyCanonicalizationPatternsOp()
    transform.YieldOp()
    # CHECK-LABEL: TEST: testApplyPatternsOpCompact
    # CHECK: apply_patterns to
    # CHECK: transform.apply_patterns.canonicalization
    # CHECK: !transform.any_op


@run
def testApplyPatternsOpWithType(module: Module):
  sequence = transform.SequenceOp(
      transform.FailurePropagationMode.Propagate, [],
      transform.OperationType.get('test.dummy')
  )
  with InsertionPoint(sequence.body):
    with InsertionPoint(transform.ApplyPatternsOp(sequence.bodyTarget).patterns):
      transform.ApplyCanonicalizationPatternsOp()
    transform.YieldOp()
    # CHECK-LABEL: TEST: testApplyPatternsOp
    # CHECK: apply_patterns to
    # CHECK: transform.apply_patterns.canonicalization
    # CHECK: !transform.op<"test.dummy">


@run
def testReplicateOp(module: Module):
    with_pdl = transform_pdl.WithPDLPatternsOp(transform.AnyOpType.get())
    with InsertionPoint(with_pdl.body):
        sequence = transform.SequenceOp(
            transform.FailurePropagationMode.Propagate, [], with_pdl.bodyTarget
        )
        with InsertionPoint(sequence.body):
            m1 = transform_pdl.PDLMatchOp(
                transform.AnyOpType.get(), sequence.bodyTarget, "first"
            )
            m2 = transform_pdl.PDLMatchOp(
                transform.AnyOpType.get(), sequence.bodyTarget, "second"
            )
            transform.ReplicateOp(m1, [m2])
            transform.YieldOp()
    # CHECK-LABEL: TEST: testReplicateOp
    # CHECK: %[[FIRST:.+]] = pdl_match
    # CHECK: %[[SECOND:.+]] = pdl_match
    # CHECK: %{{.*}} = replicate num(%[[FIRST]]) %[[SECOND]]


@run
def testApplyRegisteredPassOp(module: Module):
    sequence = transform.SequenceOp(
        transform.FailurePropagationMode.Propagate, [], transform.AnyOpType.get()
    )
    with InsertionPoint(sequence.body):
        mod = transform.ApplyRegisteredPassOp(
            transform.AnyOpType.get(), sequence.bodyTarget, "canonicalize"
        )
        mod = transform.ApplyRegisteredPassOp(
            transform.AnyOpType.get(),
            mod.result,
            "canonicalize",
            options={"top-down": BoolAttr.get(False)},
        )
        max_iter = transform.param_constant(
            transform.AnyParamType.get(),
            IntegerAttr.get(IntegerType.get_signless(64), 10),
        )
        max_rewrites = transform.param_constant(
            transform.AnyParamType.get(),
            IntegerAttr.get(IntegerType.get_signless(64), 1),
        )
        transform.apply_registered_pass(
            transform.AnyOpType.get(),
            mod,
            "canonicalize",
            options={
                "top-down": BoolAttr.get(False),
                "max-iterations": max_iter,
                "test-convergence": True,
                "max-rewrites": max_rewrites,
            },
        )
        transform.YieldOp()
    # CHECK-LABEL: TEST: testApplyRegisteredPassOp
    # CHECK: transform.sequence
    # CHECK:   %{{.*}} = apply_registered_pass "canonicalize" to {{.*}} : (!transform.any_op) -> !transform.any_op
    # CHECK:   %{{.*}} = apply_registered_pass "canonicalize"
    # CHECK-SAME:    with options = {"top-down" = false}
    # CHECK-SAME:    to {{.*}} : (!transform.any_op) -> !transform.any_op
    # CHECK:   %[[MAX_ITER:.+]] = transform.param.constant
    # CHECK:   %[[MAX_REWRITE:.+]] = transform.param.constant
    # CHECK:   %{{.*}} = apply_registered_pass "canonicalize"
    # NB: MLIR has sorted the dict lexicographically by key:
    # CHECK-SAME:    with options = {"max-iterations" = %[[MAX_ITER]],
    # CHECK-SAME:                    "max-rewrites" =  %[[MAX_REWRITE]],
    # CHECK-SAME:                    "test-convergence" = true,
    # CHECK-SAME:                    "top-down" = false}
    # CHECK-SAME:    to %{{.*}} : (!transform.any_op, !transform.any_param, !transform.any_param) -> !transform.any_op
