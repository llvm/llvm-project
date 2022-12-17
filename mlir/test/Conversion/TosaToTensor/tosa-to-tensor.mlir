// RUN: mlir-opt --split-input-file --tosa-to-tensor %s -o -| FileCheck %s

// CHECK-LABEL: @slice
func.func @slice(%arg0: tensor<6xf32>) ->() {
  // CHECK: [[SLICE:%.+]] = tensor.extract_slice %arg0[2] [1] [1]
  %0 = "tosa.slice"(%arg0) {start = [2], size = [1]} : (tensor<6xf32>)  -> (tensor<1xf32>)
  return
}

// -----

// CHECK-LABEL: @slice_dyn
func.func @slice_dyn(%arg0: tensor<?xf32>) -> (tensor<?xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[DIM:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[C2:.+]] = arith.constant 2 : index
  // CHECK: %[[SUB:.+]] = arith.subi %[[DIM]], %[[C2]]
  // CHECK: tensor.extract_slice %arg0[2] [%[[SUB]]] [1]
  %0 = "tosa.slice"(%arg0) {start = [2], size = [-1]} : (tensor<?xf32>)  -> (tensor<?xf32>)
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @pad_float
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @pad_float(%arg0 : tensor<1x2xf32>) -> (tensor<4x9xf32>) {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // TODO: Output contains multiple "arith.constant 1 : index".
  // CHECK-DAG: [[INDEX1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[INDEX2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[INDEX3:%.+]] = arith.constant 3 : index
  // CHECK-DAG: [[INDEX4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: tensor.pad %[[ARG0]] low{{\[}}%{{.*}}, [[INDEX3]]] high{{\[}}[[INDEX2]], [[INDEX4]]]  {
  // CHECK:   tensor.yield [[CST]]
  // CHECK: } : tensor<1x2xf32> to tensor<4x9xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xf32>, tensor<2x2xi32>)  -> (tensor<4x9xf32>)
  return %1 : tensor<4x9xf32>
}

func.func @pad_int(%arg0 : tensor<1x2xi32>) -> (tensor<4x9xi32>) {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: [[CST:%.+]] = arith.constant 0 : i32
  // CHECK: tensor.pad
  // CHECK:   tensor.yield [[CST]]
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xi32>, tensor<2x2xi32>)  -> (tensor<4x9xi32>)
  return %1 : tensor<4x9xi32>
}

func.func @pad_quant(%arg0 : tensor<1x2xi32>) -> (tensor<4x9xi32>) {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: [[CST:%.+]] = arith.constant 42 : i32
  // CHECK: tensor.pad
  // CHECK:   tensor.yield [[CST]]
  %1 = "tosa.pad"(%arg0, %0) {quantization_info = #tosa.pad_quant<input_zp = 42>} : (tensor<1x2xi32>, tensor<2x2xi32>)  -> (tensor<4x9xi32>)
  return %1 : tensor<4x9xi32>
}

// -----

func.func @pad_float_explicit(%arg0 : tensor<1x2xf32>) -> (tensor<4x9xf32>) {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // TODO: Output contains multiple "arith.constant 1 : index".
  // CHECK-DAG: [[INDEX1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[INDEX2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[INDEX3:%.+]] = arith.constant 3 : index
  // CHECK-DAG: [[INDEX4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[CST:%.+]] = arith.constant 4.200000e+01 : f32
  // CHECK: tensor.pad %[[ARG0]] low{{\[}}%{{.*}}, [[INDEX3]]] high{{\[}}[[INDEX2]], [[INDEX4]]]  {
  // CHECK:   tensor.yield [[CST]]
  // CHECK: } : tensor<1x2xf32> to tensor<4x9xf32>
  %1 = arith.constant dense<42.0> : tensor<f32>
  %2 = "tosa.pad"(%arg0, %0, %1)  : (tensor<1x2xf32>, tensor<2x2xi32>, tensor<f32>)  -> (tensor<4x9xf32>)
  return %2 : tensor<4x9xf32>
}

// -----

func.func @pad_dyn_input(%arg0 : tensor<?x2xf32>) -> (tensor<?x9xf32>) {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // TODO: Output contains multiple "arith.constant 1 : index".
  // CHECK-DAG: [[INDEX1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[INDEX2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[INDEX3:%.+]] = arith.constant 3 : index
  // CHECK-DAG: [[INDEX4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: tensor.pad %[[ARG0]] low{{\[}}%{{.*}}, [[INDEX3]]] high{{\[}}[[INDEX2]], [[INDEX4]]]  {
  // CHECK:   tensor.yield [[CST]]
  // CHECK: } : tensor<?x2xf32> to tensor<?x9xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<?x2xf32>, tensor<2x2xi32>)  -> (tensor<?x9xf32>)
  return %1 : tensor<?x9xf32>
}

func.func @pad_dyn_padding(%arg0 : tensor<1x2xf32>) -> (tensor<?x9xf32>) {
  %0 = arith.constant dense<[[-1, 2], [3, 4]]> : tensor<2x2xi32>
  // TODO: Output contains multiple "arith.constant 1 : index".
  // CHECK-DAG: [[INDEX1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[INDEX2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[INDEX3:%.+]] = arith.constant 3 : index
  // CHECK-DAG: [[INDEX4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: tensor.pad %[[ARG0]] low{{\[}}%{{.*}}, [[INDEX3]]] high{{\[}}[[INDEX2]], [[INDEX4]]]  {
  // CHECK:   tensor.yield [[CST]]
  // CHECK: } : tensor<1x2xf32> to tensor<?x9xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xf32>, tensor<2x2xi32>)  -> (tensor<?x9xf32>)
  return %1 : tensor<?x9xf32>
}
