// RUN: mlir-opt --convert-scf-to-spirv --reconcile-unrealized-casts %s | FileCheck %s

// Make sure that this test pass handles function signature conversion properly.

// CHECK-LABEL: spirv.func @add_scalar
// CHECK-SAME: (%[[ARG0:.+]]: i32) -> i32
func.func @add_scalar(%arg0: i32) -> i32 {
  // CHECK-NEXT: %[[RES:.+]] = spirv.IAdd %[[ARG0]], %[[ARG0]] : i32
  // CHECK-NEXT: spirv.ReturnValue %[[RES]] : i32
  %0 = arith.addi %arg0, %arg0 : i32
  return %0 : i32
}

// CHECK-LABEL: spirv.func @add_index
// CHECK-SAME: (%[[ARG0:.+]]: i32) -> i32
func.func @add_index(%arg0: index) -> index {
  // CHECK-NEXT: %[[RES:.+]] = spirv.IAdd %[[ARG0]], %[[ARG0]] : i32
  // CHECK-NEXT: spirv.ReturnValue %[[RES]] : i32
  %0 = arith.addi %arg0, %arg0 : index
  return %0 : index
}
