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

// CHECK-LABEL: func @tensor.from_elements() {
func @tensor.from_elements() {
  %c0 = "std.constant"() {value = 0: index} : () -> index
  // CHECK: %0 = tensor.from_elements %c0 : tensor<1xindex>
  %0 = tensor.from_elements %c0 : tensor<1xindex>

  %c1 = "std.constant"() {value = 1: index} : () -> index
  // CHECK: %1 = tensor.from_elements %c0, %c1 : tensor<2xindex>
  %1 = tensor.from_elements %c0, %c1 : tensor<2xindex>

  %c0_f32 = "std.constant"() {value = 0.0: f32} : () -> f32
  // CHECK: [[C0_F32:%.*]] = constant
  // CHECK: %2 = tensor.from_elements [[C0_F32]] : tensor<1xf32>
  %2 = tensor.from_elements %c0_f32 : tensor<1xf32>

  // CHECK: tensor.from_elements : tensor<0xindex>
  %3 = tensor.from_elements : tensor<0xindex>

  return
}

// CHECK-LABEL: @tensor.generate
func @tensor.generate(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  %tnsr = tensor.generate %m, %n {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = constant 8.0 : f32
      tensor.yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}
