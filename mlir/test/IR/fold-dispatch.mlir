// RUN: mlir-opt %s --transform-interpreter --verify-diagnostics

// `transform.test_fold` folds each test op with `Operation::fold` and reports
// the result as a remark. A dictionary attribute configures each fold of a
// test op:
//  - `replace = [...]` holds one replacement per result. A `#test.fold_ref`
//    names a part of the op: `<keep>` keeps the result, `<result I>` and
//    `<operand I>` are result or operand I, and `<operand_attr I>` is the
//    constant value of operand I. Any other attribute replaces the result
//    with that attribute.
//  - `in_place` makes the fold also change the op in place, once: the fold
//    drops the key.
//  - `in_place_steps = N` makes the next N folds only change the op in place:
//    each of them decrements N.
// Without the attribute, the fold fails.

// `test.fold_dispatch_fallback` has no fold and no fold trait, so only the
// fallback fold of the test dialect folds it, as `legacy_dialect_fold` says.
func.func @dialect_fallback(%arg0: i32) {
  // expected-remark @below {{fold: failure}}
  %0:2 = "test.fold_dispatch_fallback"() : () -> (i32, i32)
  // expected-remark @below {{fold: [1 : i32, 2 : i32]}}
  %1:2 = "test.fold_dispatch_fallback"()
      {legacy_dialect_fold = {replace = [1 : i32, 2 : i32]}}
      : () -> (i32, i32)
  // expected-remark @below {{fold: in place}}
  %2:2 = "test.fold_dispatch_fallback"() {legacy_dialect_fold = {in_place}}
      : () -> (i32, i32)
  // The result of the fallback is normalized, so the results of the op keep
  // the results.
  // expected-remark @below {{fold: failure}}
  %3:2 = "test.fold_dispatch_fallback"()
      {legacy_dialect_fold = {replace = [#test.fold_ref<result 0>,
                                         #test.fold_ref<result 1>]}}
      : () -> (i32, i32)
  // The fallback gets the constant operands.
  %c7 = "test.constant"() <{value = 7 : i32}> : () -> i32
  // expected-remark @below {{fold: [7 : i32, operand 1]}}
  %4:2 = "test.fold_dispatch_fallback"(%c7, %arg0)
      {legacy_dialect_fold = {replace = [#test.fold_ref<operand_attr 0>,
                                         #test.fold_ref<operand 1>]}}
      : (i32, i32) -> (i32, i32)
  return
}

// `test.fold_dispatch` has a fold in the `OpFoldResults` form, configured by
// `fold`, and a legacy fold trait, configured by `legacy_trait_fold`. The trait
// folds only when the fold of the op replaces no result.
func.func @own_fold_and_trait(%arg0: i32) {
  // A partial fold of the op skips the trait.
  // expected-remark @below {{fold: [keep, 42 : i32]}}
  %0:2 = "test.fold_dispatch"()
      {fold = {replace = [#test.fold_ref<keep>, 42 : i32]},
       legacy_trait_fold = {in_place}}
      : () -> (i32, i32)
  // After an in-place fold of the op, the trait replaces every result...
  // expected-remark @below {{fold: [1 : i32, 2 : i32] in place}}
  %1:2 = "test.fold_dispatch"()
      {fold = {in_place}, legacy_trait_fold = {replace = [1 : i32, 2 : i32]}}
      : () -> (i32, i32)
  // ... or some results.
  // expected-remark @below {{fold: [keep, 1 : i32] in place}}
  %2:2 = "test.fold_dispatch"()
      {fold = {in_place},
       legacy_trait_fold = {replace = [#test.fold_ref<result 0>, 1 : i32]}}
      : () -> (i32, i32)
  // After a failure of the op, the trait changes the op in place.
  // expected-remark @below {{fold: in place}}
  %3:2 = "test.fold_dispatch"() {legacy_trait_fold = {in_place}}
      : () -> (i32, i32)
  // The fold of the op keeps every result, which normalizes to a failure, so
  // the trait folds.
  // expected-remark @below {{fold: [1 : i32, 1 : i32]}}
  %4:2 = "test.fold_dispatch"()
      {fold = {replace = [#test.fold_ref<result 0>, #test.fold_ref<result 1>]},
       legacy_trait_fold = {replace = [1 : i32, 1 : i32]}} : () -> (i32, i32)
  // A partial fold of the op does not fall back on the dialect.
  // expected-remark @below {{fold: [42 : i32, keep]}}
  %5:2 = "test.fold_dispatch"()
      {fold = {replace = [42 : i32, #test.fold_ref<keep>]},
       legacy_dialect_fold = {replace = [7 : i32, 7 : i32]}}
      : () -> (i32, i32)
  // The fold of the op gets the constant operands.
  %c7 = "test.constant"() <{value = 7 : i32}> : () -> i32
  // expected-remark @below {{fold: [7 : i32, keep]}}
  %6:2 = "test.fold_dispatch"(%c7, %arg0)
      {fold = {replace = [#test.fold_ref<operand_attr 0>,
                          #test.fold_ref<operand_attr 1>]}}
      : (i32, i32) -> (i32, i32)
  return
}

// A replacement can name another result of the op only if the fold keeps that
// result. Otherwise the fold hook drops the replacements, and only the
// in-place change stays.
func.func @fold_naming_replaced_result() {
  // expected-remark @below {{fold: failure}}
  %0:2 = "test.fold_dispatch"()
      {fold = {replace = [#test.fold_ref<result 1>, 1 : i32]}}
      : () -> (i32, i32)
  // expected-remark @below {{fold: in place}}
  %1:2 = "test.fold_dispatch"()
      {fold = {replace = [#test.fold_ref<result 1>, 1 : i32], in_place}}
      : () -> (i32, i32)
  // expected-remark @below {{fold: [result 1, keep]}}
  %2:2 = "test.fold_dispatch"()
      {fold = {replace = [#test.fold_ref<result 1>, #test.fold_ref<keep>]}}
      : () -> (i32, i32)
  // An op can use its own results in a graph region, as in an unreachable
  // block.
  test.graph_region {
    // expected-remark @below {{fold: failure}}
    %3:2 = "test.fold_dispatch"(%3#1, %3#0)
        {fold = {replace = [#test.fold_ref<operand 0>,
                            #test.fold_ref<operand 1>]}}
        : (i32, i32) -> (i32, i32)
    "test.valid"(%3#0, %3#1) : (i32, i32) -> ()
  }
  return
}

// `test.fold_dispatch_traits` has no fold of its own and two fold traits: one
// in the `OpFoldResults` form, configured by `trait_fold`, then a legacy one.
// Folding stops at the first trait that does not fail.
func.func @trait_forms() {
  // A partial fold of the first trait skips the legacy trait and the dialect.
  // expected-remark @below {{fold: [keep, 1 : i32]}}
  %0:2 = "test.fold_dispatch_traits"()
      {trait_fold = {replace = [#test.fold_ref<keep>, 1 : i32]},
       legacy_trait_fold = {in_place},
       legacy_dialect_fold = {replace = [7 : i32, 7 : i32]}}
      : () -> (i32, i32)
  // An in-place fold of the first trait also skips the legacy trait.
  // expected-remark @below {{fold: in place}}
  %1:2 = "test.fold_dispatch_traits"()
      {trait_fold = {in_place},
       legacy_trait_fold = {replace = [1 : i32, 1 : i32]}} : () -> (i32, i32)
  // A partial fold keeps the in-place bit of the trait.
  // expected-remark @below {{fold: [1 : i32, keep] in place}}
  %2:2 = "test.fold_dispatch_traits"()
      {trait_fold = {replace = [1 : i32, #test.fold_ref<keep>], in_place},
       legacy_trait_fold = {replace = [7 : i32, 7 : i32]}} : () -> (i32, i32)
  // The first trait keeps every result, which normalizes to a failure, so the
  // legacy trait folds.
  // expected-remark @below {{fold: [1 : i32, 1 : i32]}}
  %3:2 = "test.fold_dispatch_traits"()
      {trait_fold = {replace = [#test.fold_ref<result 0>,
                                #test.fold_ref<result 1>]},
       legacy_trait_fold = {replace = [1 : i32, 1 : i32]}} : () -> (i32, i32)
  // After a failure of the first trait, the legacy trait changes the op in
  // place.
  // expected-remark @below {{fold: in place}}
  %4:2 = "test.fold_dispatch_traits"() {legacy_trait_fold = {in_place}}
      : () -> (i32, i32)
  // When both traits fail, the fold falls back on the dialect.
  // expected-remark @below {{fold: [3 : i32, 3 : i32]}}
  %5:2 = "test.fold_dispatch_traits"()
      {legacy_dialect_fold = {replace = [3 : i32, 3 : i32]}}
      : () -> (i32, i32)
  // A trait fold can name another result that it also replaces: a cast in a
  // graph region can return such a chain. The fold hook does not drop the
  // replacements of a trait fold.
  // expected-remark @below {{fold: [result 1, 1 : i32]}}
  %6:2 = "test.fold_dispatch_traits"()
      {trait_fold = {replace = [#test.fold_ref<result 1>, 1 : i32]}}
      : () -> (i32, i32)
  return
}

// The fallback fold of the test dialect also has the `OpFoldResults` form,
// configured by `dialect_fold`, which replaces its legacy method.
func.func @dialect_fallback_results_form() {
  // expected-remark @below {{fold: [1 : i32, keep]}}
  %0:2 = "test.fold_dispatch_fallback"()
      {dialect_fold = {replace = [1 : i32, #test.fold_ref<keep>]}}
      : () -> (i32, i32)
  // expected-remark @below {{fold: in place}}
  %1:2 = "test.fold_dispatch_fallback"() {dialect_fold = {in_place}}
      : () -> (i32, i32)
  // The legacy method does not run.
  // expected-remark @below {{fold: [2 : i32, 2 : i32]}}
  %2:2 = "test.fold_dispatch_fallback"()
      {dialect_fold = {replace = [2 : i32, 2 : i32]},
       legacy_dialect_fold = {replace = [7 : i32, 7 : i32]}}
      : () -> (i32, i32)
  // The result of the fallback is normalized.
  // expected-remark @below {{fold: failure}}
  %3:2 = "test.fold_dispatch_fallback"()
      {dialect_fold = {replace = [#test.fold_ref<result 0>,
                                    #test.fold_ref<result 1>]}}
      : () -> (i32, i32)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match
        ops{["test.fold_dispatch_fallback", "test.fold_dispatch",
             "test.fold_dispatch_traits"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.test_fold %ops : !transform.any_op
    transform.yield
  }
}
