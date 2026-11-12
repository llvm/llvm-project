// RUN: mlir-opt %s --tosa-to-linalg-pipeline="validation=false" | FileCheck %s

// Check that --tosa-to-linalg-pipeline="validation=false" lowers to linalg
// and does not leave a tosa.target_env module attribute behind.

// CHECK-LABEL: module
// CHECK-NOT: tosa.target_env
// CHECK-LABEL: func.func @simple_abs
// CHECK-NOT: tosa.
// CHECK: linalg.
func.func @simple_abs(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = tosa.abs %arg0 : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
