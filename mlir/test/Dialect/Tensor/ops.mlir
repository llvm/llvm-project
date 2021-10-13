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

// CHECK-LABEL:   func @insert(
// CHECK-SAME:                  %[[SCALAR:.*]]: f32
// CHECK-SAME:                  %[[INDEX:.*]]: index
// CHECK-SAME:                  %[[DEST1:.*]]: tensor<?x?x?xf32>
// CHECK-SAME:                  %[[DEST2:.*]]: tensor<*xf32>
func @insert(%arg0: f32, %arg1: index, %arg2: tensor<?x?x?xf32>, %arg3: tensor<*xf32>) {
  // CHECK: tensor.insert %[[SCALAR]] into %[[DEST1]][%[[INDEX]], %[[INDEX]], %[[INDEX]]] : tensor<?x?x?xf32>
  %0 = tensor.insert %arg0 into %arg2[%arg1, %arg1, %arg1] : tensor<?x?x?xf32>
  // CHECK: tensor.insert %[[SCALAR]] into %[[DEST2]][%[[INDEX]], %[[INDEX]], %[[INDEX]]] : tensor<*xf32>
  %1 = tensor.insert %arg0 into %arg3[%arg1, %arg1, %arg1] : tensor<*xf32>
  return
}

// CHECK-LABEL: func @tensor.from_elements() {
func @tensor.from_elements() {
  %c0 = "arith.constant"() {value = 0: index} : () -> index
  // CHECK: %0 = tensor.from_elements %c0 : tensor<1xindex>
  %0 = tensor.from_elements %c0 : tensor<1xindex>

  %c1 = "arith.constant"() {value = 1: index} : () -> index
  // CHECK: %1 = tensor.from_elements %c0, %c1 : tensor<2xindex>
  %1 = tensor.from_elements %c0, %c1 : tensor<2xindex>

  %c0_f32 = "arith.constant"() {value = 0.0: f32} : () -> f32
  // CHECK: [[C0_F32:%.*]] = arith.constant
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
      %elem = arith.constant 8.0 : f32
      tensor.yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// CHECK-LABEL: func @tensor_reshape
func @tensor_reshape(%unranked: tensor<*xf32>, %shape1: tensor<1xi32>,
         %shape2: tensor<2xi32>, %shape3: tensor<?xi32>) -> tensor<*xf32> {
  %dyn_vec = tensor.reshape %unranked(%shape1)
               : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
  %dyn_mat = tensor.reshape %dyn_vec(%shape2)
               : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %new_unranked = tensor.reshape %dyn_mat(%shape3)
               : (tensor<?x?xf32>, tensor<?xi32>) -> tensor<*xf32>
  return %new_unranked : tensor<*xf32>
}
