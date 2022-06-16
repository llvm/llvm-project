// RUN: mlir-opt --split-input-file --tosa-to-tensor %s -o -| FileCheck %s

// CHECK-LABLE: func @slice
func.func @slice(%arg0: tensor<6xf32>) ->() {
  // CHECK: [[SLICE:%.+]] = tensor.extract_slice %arg0[2] [1] [1]
  %0 = "tosa.slice"(%arg0) {start = [2], size = [1]} : (tensor<6xf32>)  -> (tensor<1xf32>)
  return
}

// -----

// CHECK-LABLE: func @slice_dyn
func.func @slice_dyn(%arg0: tensor<?xf32>) -> (tensor<?xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[DIM:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[C2:.+]] = arith.constant 2 : index
  // CHECK: %[[SUB:.+]] = arith.subi %[[DIM]], %[[C2]]
  // CHECK: %2 = tensor.extract_slice %arg0[2] [%[[SUB]]] [1]
  %0 = "tosa.slice"(%arg0) {start = [2], size = [-1]} : (tensor<?xf32>)  -> (tensor<?xf32>)
  return %0 : tensor<?xf32>
}
