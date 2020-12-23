// RUN: mlir-opt <%s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @cast(
func @cast(%arg0: tensor<*xf32>, %arg1 : tensor<4x4xf32>, %arg2: tensor<?x?xf32>) {
  // CHECK: tensor.cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
  %0 = tensor.cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
  // CHECK: tensor.cast %arg1 : tensor<4x4xf32> to tensor<*xf32>
  %1 = tensor.cast %arg1 : tensor<4x4xf32> to tensor<*xf32>
  // CHECK: tensor.cast %arg2 : tensor<?x?xf32> to tensor<4x?xf32>
  %2 = tensor.cast %arg2 : tensor<?x?xf32> to tensor<4x?xf32>
  // CHECK: tensor.cast %2 : tensor<4x?xf32> to tensor<?x?xf32>
  %3 = tensor.cast %2 : tensor<4x?xf32> to tensor<?x?xf32>
  return
}

// CHECK-LABEL:   func @extract(
// CHECK-SAME:                  %[[TENSOR:.*]]: tensor<?x?x?xf32>,
// CHECK-SAME:                  %[[INDEX:.*]]: index) {
func @extract(%arg0: tensor<?x?x?xf32>, %arg1: index) {
  // CHECK: tensor.extract %[[TENSOR]][%[[INDEX]], %[[INDEX]], %[[INDEX]]] : tensor<?x?x?xf32>
  %0 = tensor.extract %arg0[%arg1, %arg1, %arg1] : tensor<?x?x?xf32>
  return
}
