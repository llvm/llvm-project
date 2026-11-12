// RUN: mlir-opt %s -allow-unregistered-dialect -test-legalize-patterns="test-legalize-folding-mode=never" | FileCheck %s

// CHECK-LABEL: @remove_foldable_op(
func.func @remove_foldable_op(%arg0 : i32) -> (i32) {
  // Check that op was not folded.
  // CHECK: "test.op_with_region_fold"
  %0 = "test.op_with_region_fold"(%arg0) ({
    "foo.op_with_region_terminator"() : () -> ()
  }) : (i32) -> (i32)
  "test.return"(%0) : (i32) -> ()
}

