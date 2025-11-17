// Test with the default (one application of the folder) and then with 2 iterations.
// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(test-single-fold))" | FileCheck %s --check-prefixes=CHECK,CHECK-ONE
// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(test-single-fold{max-iterations=2}))" | FileCheck %s --check-prefixes=CHECK,CHECK-TWO


// Folding entirely this requires to move the constant to the right
// before invoking the op-specific folder.
// With one iteration, we just push the constant to the right.
// With a second iteration, we actually fold the "add" (x+0->x)
// CHECK: func @recurse_fold_traits(%[[ARG0:.*]]: i32)
func.func @recurse_fold_traits(%arg0 : i32) -> i32 {
  %cst0 = arith.constant 0 : i32
// CHECK-ONE:  %[[ADD:.*]] = arith.addi %[[ARG0]], 
  %res = arith.addi %cst0, %arg0 : i32
// CHECK-ONE:   return %[[ADD]] : i32
// CHECK-TWO:   return %[[ARG0]] : i32
  return %res : i32
}
