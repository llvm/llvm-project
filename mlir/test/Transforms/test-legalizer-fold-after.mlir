// RUN: mlir-opt %s -test-legalize-patterns="test-legalize-folding-mode=after-patterns" | FileCheck %s

// CHECK-LABEL: @fold_legalization
func.func @fold_legalization() -> i32 {
  // CHECK-NOT: op_in_place_self_fold
  // CHECK: 97
  %1 = "test.op_in_place_self_fold"() : () -> (i32)
  "test.return"(%1) : (i32) -> ()
}
