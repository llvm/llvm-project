// RUN: mlir-opt -convert-to-spirv="run-signature-conversion=false run-vector-unrolling=false" -split-input-file %s | FileCheck %s

// CHECK-LABEL: @return_scalar
// CHECK-SAME: %[[ARG0:.*]]: i32
// CHECK: spirv.ReturnValue %[[ARG0]]
func.func @return_scalar(%arg0 : i32) -> i32 {
  return %arg0 : i32
}

// CHECK-LABEL: @return_vector
// CHECK-SAME: %[[ARG0:.*]]: vector<4xi32>
// CHECK: spirv.ReturnValue %[[ARG0]]
func.func @return_vector(%arg0 : vector<4xi32>) -> vector<4xi32> {
  return %arg0 : vector<4xi32>
}
