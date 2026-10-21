// RUN: mlir-opt %s --transform-interpreter --verify-diagnostics \
// RUN:   -split-input-file | FileCheck %s

// `transform.test_fold` with `api = try_fold` folds each test op with
// `OpBuilder::tryFold`. With `api = materialize`, it also materializes the
// replacements with `OpBuilder::materializeFoldResults`. The `fold` attribute
// configures the fold of each test op, as in IR/fold-dispatch.mlir.

// CHECK-LABEL: func @try_fold
func.func @try_fold() {
  // `tryFold` creates no constant.
  // CHECK-NEXT: "test.fold_dispatch"()
  // CHECK-SAME: {fold = {replace = [#test.fold_ref<keep>, 42 : i32]}}
  // expected-remark @below {{fold: [keep, 42 : i32]}}
  %0:2 = "test.fold_dispatch"()
      {fold = {replace = [#test.fold_ref<keep>, 42 : i32]}}
      : () -> (i32, i32)
  // A fold that changes the op in place repeats, and the result keeps the
  // in-place bit.
  // CHECK-NEXT: "test.fold_dispatch"() {fold = {replace = [1 : i32, 1 : i32]}}
  // expected-remark @below {{fold: [1 : i32, 1 : i32] in place}}
  %1:2 = "test.fold_dispatch"()
      {fold = {in_place_steps = 1, replace = [1 : i32, 1 : i32]}}
      : () -> (i32, i32)
  // An in-place fold followed by a failure is an in-place fold.
  // CHECK-NEXT: "test.fold_dispatch"() : () -> (i32, i32)
  // expected-remark @below {{fold: in place}}
  %2:2 = "test.fold_dispatch"() {fold = {in_place}} : () -> (i32, i32)
  // After 64 in-place folds, `tryFold` stops with a failure.
  // CHECK-NEXT: "test.fold_dispatch"() {fold = {in_place_steps = 36 : i64}}
  // expected-remark @below {{fold: failure}}
  %3:2 = "test.fold_dispatch"() {fold = {in_place_steps = 100}}
      : () -> (i32, i32)
  // `tryFold` does not fold a constant.
  // CHECK-NEXT: "test.constant"
  // expected-remark @below {{fold: failure}}
  %4 = "test.constant"() <{value = 7 : i32}> : () -> i32
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match
        ops{["test.fold_dispatch", "test.constant"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.test_fold %ops api = try_fold : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @materialize
// CHECK-SAME: (%[[ARG0:.*]]: i32)
func.func @materialize(%arg0: i32) {
  // An attribute becomes a new constant before the op. A value is used
  // directly.
  // CHECK-NEXT: "test.constant"() <{value = 42 : i32}>
  // CHECK-NEXT: "test.fold_dispatch"(%[[ARG0]])
  // expected-remark @below {{fold: [42 : i32, operand 0]}}
  // expected-remark @below {{materialized: [test.constant, operand 0]}}
  %0:2 = "test.fold_dispatch"(%arg0)
      {fold = {replace = [42 : i32, #test.fold_ref<operand 0>]}}
      : (i32) -> (i32, i32)
  // A kept result gets no value.
  // CHECK-NEXT: "test.constant"() <{value = 43 : i32}>
  // CHECK-NEXT: "test.fold_dispatch"()
  // expected-remark @below {{fold: [keep, 43 : i32]}}
  // expected-remark @below {{materialized: [none, test.constant]}}
  %1:2 = "test.fold_dispatch"()
      {fold = {replace = [#test.fold_ref<keep>, 43 : i32]}}
      : () -> (i32, i32)
  // If a constant fails to materialize, no constant is inserted.
  // CHECK-NEXT: "test.fold_dispatch"()
  // CHECK-NEXT: return
  // expected-remark @below {{fold: [44 : i32, "unmaterializable"]}}
  // expected-remark @below {{materialized: failure}}
  %2:2 = "test.fold_dispatch"()
      {fold = {replace = [44 : i32, "unmaterializable"]}} : () -> (i32, i32)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match ops{["test.fold_dispatch"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.test_fold %ops api = materialize : !transform.any_op
    transform.yield
  }
}

// -----

// With `liveOnly`, a replaced result without uses gets no value and no
// constant.
// CHECK-LABEL: func @materialize_live_only
func.func @materialize_live_only() -> i32 {
  // CHECK-NEXT: %[[C:.*]] = "test.constant"() <{value = 2 : i32}>
  // CHECK-NEXT: "test.fold_dispatch"()
  // CHECK-NEXT: return
  // expected-remark @below {{fold: [1 : i32, 2 : i32]}}
  // expected-remark @below {{materialized: [none, test.constant]}}
  %0:2 = "test.fold_dispatch"() {fold = {replace = [1 : i32, 2 : i32]}}
      : () -> (i32, i32)
  return %0#1 : i32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match ops{["test.fold_dispatch"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.test_fold %ops api = materialize live_only
        : !transform.any_op
    transform.yield
  }
}
