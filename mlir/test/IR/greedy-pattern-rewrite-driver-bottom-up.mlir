// RUN: mlir-opt %s -test-patterns="max-iterations=1" \
// RUN:     -allow-unregistered-dialect --split-input-file | FileCheck %s

// CHECK-LABEL: func @add_to_worklist_after_inplace_update()
func.func @add_to_worklist_after_inplace_update() {
  // The following op is updated in-place and should be added back to the
  // worklist of the GreedyPatternRewriteDriver (regardless of the value of
  // config.max_iterations).

  // CHECK: "test.any_attr_of_i32_str"() <{attr = 3 : i32}> : () -> ()
  "test.any_attr_of_i32_str"() {attr = 0 : i32} : () -> ()
  return
}

// -----

// CHECK-LABEL: func @add_ancestors_to_worklist()
func.func @add_ancestors_to_worklist() {
       // CHECK: "foo.maybe_eligible_op"() {eligible} : () -> index
  // CHECK-NEXT: "test.one_region_op"()
  "test.one_region_op"() ({
    %0 = "foo.maybe_eligible_op" () : () -> (index)
    "foo.yield"(%0) : (index) -> ()
  }) {hoist_eligible_ops}: () -> ()
  return
}

// -----

// There are no patterns in this test that apply to "test.symbol". This test is
// to ensure that symbols are not getting removed due to being "trivially dead"
// as part of a greedy rewrite. Symbols are never trivially dead.

// CHECK: "test.symbol"() <{sym_name = "foo"}>
"test.symbol"() <{sym_name = "foo"}> : () -> ()
