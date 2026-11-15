// RUN: mlir-opt --convert-elementwise-to-linalg %s | FileCheck %s

// CHECK-LABEL: func.func @select_dynamic(
// CHECK: arith.select {{.*}} : tensor<?xi32>
// CHECK: tensor.dim
func.func @select_dynamic(%pick: i1, %a: tensor<?xi32>, %b: tensor<?xi32>) -> index {
  %choice = arith.select %pick, %a, %b : tensor<?xi32>
  %zero = arith.constant 0 : index
  %size = tensor.dim %choice, %zero : tensor<?xi32>
  return %size : index
}

// CHECK-LABEL: func.func @select_static(
// CHECK: linalg.generic
func.func @select_static(%pick: i1, %a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %choice = arith.select %pick, %a, %b : tensor<4xi32>
  return %choice : tensor<4xi32>
}
