// RUN: mlir-opt %s -test-rewrite-dynamic-op | FileCheck %s

// Test that `test.one_operand_two_results` is replaced with
// `test.generic_dynamic_op`.

// CHECK-LABEL: func @rewrite_dynamic_op
func.func @rewrite_dynamic_op(%arg0: i32) {
  // CHECK-NEXT: %{{.*}}:2 = "test.dynamic_generic"(%arg0) : (i32) -> (i32, i32)
  %0:2 = "test.dynamic_one_operand_two_results"(%arg0) : (i32) -> (i32, i32)
  // CHECK-NEXT: return
  return
}
