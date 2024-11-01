// RUN: mlir-opt %s -test-patterns="max-iterations=1" | FileCheck %s

// CHECK-LABEL: func @add_to_worklist_after_inplace_update()
func.func @add_to_worklist_after_inplace_update() {
  // The following op is updated in-place and should be added back to the
  // worklist of the GreedyPatternRewriteDriver (regardless of the value of
  // config.max_iterations).

  // CHECK: "test.any_attr_of_i32_str"() {attr = 3 : i32} : () -> ()
  "test.any_attr_of_i32_str"() {attr = 0 : i32} : () -> ()
  return
}
