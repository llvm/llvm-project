// RUN: mlir-opt %s -canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// UnrealizedConversionCastOp
//===----------------------------------------------------------------------===//

// Test folding conversion casts feeding into other casts.
// CHECK-LABEL: func @multiple_conversion_casts
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]:
func.func @multiple_conversion_casts(%arg0: i32, %arg1: i32) -> (i32, i32) {
  // CHECK-NOT: unrealized_conversion_cast
  // CHECK: return %[[ARG0]], %[[ARG1]]
  %inputs:2 = builtin.unrealized_conversion_cast %arg0, %arg1 : i32, i32 to i64, i64
  %outputs:2 = builtin.unrealized_conversion_cast %inputs#0, %inputs#1 : i64, i64 to i32, i32
  return %outputs#0, %outputs#1 : i32, i32
}

// CHECK-LABEL: func @multiple_conversion_casts
func.func @multiple_conversion_casts_failure(%arg0: i32, %arg1: i32, %arg2: i64) -> (i32, i32) {
  // CHECK: unrealized_conversion_cast
  // CHECK: unrealized_conversion_cast
  %inputs:2 = builtin.unrealized_conversion_cast %arg0, %arg1 : i32, i32 to i64, i64
  %outputs:2 = builtin.unrealized_conversion_cast %arg2, %inputs#1 : i64, i64 to i32, i32
  return %outputs#0, %outputs#1 : i32, i32
}
