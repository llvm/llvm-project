// RUN: mlir-opt --split-input-file --tosa-to-tensor %s -o -| FileCheck %s

// CHECK-LABEL: @test_reshape_downrank
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_reshape_downrank(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  // CHECK: [[RESHAPE:%.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 6>} : (tensor<2x3xf32>) -> tensor<6xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<6xf32>
}

// -----

// CHECK-LABEL: @test_reshape_downrank_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_reshape_downrank_dyn(%arg0: tensor<2x?xf32>) -> tensor<?xf32> {
  // CHECK: [[RESHAPE:%.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: -1>} : (tensor<2x?xf32>) -> tensor<?xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @test_reshape_uprank
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_reshape_uprank(%arg0: tensor<6xf32>) -> tensor<2x3xf32> {
  // CHECK: [[RESHAPE:%.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 2, 3>} : (tensor<6xf32>) -> tensor<2x3xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @test_reshape_uprank_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_reshape_uprank_dyn(%arg0: tensor<?xf32>) -> tensor<2x?xf32> {
  // CHECK: [[RESHAPE:%.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 2, -1>} : (tensor<?xf32>) -> tensor<2x?xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: @test_reshape_samerank
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<3x2xf32>)
func.func @test_reshape_samerank(%arg0: tensor<3x2xf32>) -> tensor<2x3xf32> {
  // CHECK-NEXT: %[[RESHAPE1:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
  // CHECK-NEXT: %[[RESHAPE2:.*]] = tensor.expand_shape %[[RESHAPE1]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 2, 3>} : (tensor<3x2xf32>) -> tensor<2x3xf32>
  // CHECK-NEXT: return %[[RESHAPE2]]
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @test_reshape_samerank_dyn
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<?x2xf32>)
func.func @test_reshape_samerank_dyn(%arg0: tensor<?x2xf32>) -> tensor<2x?xf32> {
  // CHECK-NEXT: %[[RESHAPE1:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
  // CHECK-NEXT: %[[RESHAPE2:.*]] = tensor.expand_shape %[[RESHAPE1]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 2, -1>} : (tensor<?x2xf32>) -> tensor<2x?xf32>
  // CHECK-NEXT: return %[[RESHAPE2]]
  return %0 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: @test_reshape_downrank_6D
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @test_reshape_downrank_6D(%arg0: tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32> {
  // CHECK: tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2], [3], [4, 5]]
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 6, 5, 77>} : (tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32>
  return %0 : tensor<6x5x77xf32>
}

// -----

// CHECK-LABEL: @test_reshape_downrank_6D_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @test_reshape_downrank_6D_dyn(%arg0: tensor<1x2x?x5x7x11xf32>) -> tensor<?x5x77xf32> {
  // CHECK: tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2, 3, 4, 5]]
  // CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1, 2]]
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: -1, 5, 77>} : (tensor<1x2x?x5x7x11xf32>) -> tensor<?x5x77xf32>
  return %0 : tensor<?x5x77xf32>
}

// -----

// CHECK-LABLE: func @slice
func.func @slice(%arg0: tensor<6xf32>) ->() {
  // CHECK: [[SLICE:%.+]] = tensor.extract_slice %arg0[2] [1] [1]
  %0 = "tosa.slice"(%arg0) {start = array<i64: 2>, size = array<i64: 1>} : (tensor<6xf32>)  -> (tensor<1xf32>)
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
  %0 = "tosa.slice"(%arg0) {start = array<i64: 2>, size = array<i64: -1>} : (tensor<?xf32>)  -> (tensor<?xf32>)
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

// -----

// CHECK-LABEL: @concat
// CHECK-SAME: %[[ARG0:.+]]: tensor<5x1xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<6x1xf32>
func.func @concat(%arg0: tensor<5x1xf32>, %arg1: tensor<6x1xf32>) -> () {
  // CHECK-DAG: [[INIT:%.+]] = tensor.empty() : tensor<11x1xf32>
  // CHECK-DAG: [[INSERT0:%.+]] = tensor.insert_slice %[[ARG0]] into [[INIT]][0, 0] [5, 1] [1, 1]
  // CHECK-DAG: [[INSERT1:%.+]] = tensor.insert_slice %[[ARG1]] into [[INSERT0]][5, 0] [6, 1] [1, 1]
  %0 = "tosa.concat"(%arg0, %arg1) { axis = 0 : i32} : (tensor<5x1xf32>, tensor<6x1xf32>)  -> (tensor<11x1xf32>)

  // CHECK-DAG: [[INIT:%.+]] = tensor.empty() : tensor<5x2xf32>
  // CHECK-DAG: [[INSERT0:%.+]] = tensor.insert_slice %[[ARG0]] into [[INIT]][0, 0] [5, 1] [1, 1]
  // CHECK: [[INSERT1:%.+]] = tensor.insert_slice %[[ARG0]] into [[INSERT0]][0, 1] [5, 1] [1, 1]
  %1 = "tosa.concat"(%arg0, %arg0) { axis = 1 : i32} : (tensor<5x1xf32>, tensor<5x1xf32>)  -> (tensor<5x2xf32>)
  return
}

// -----

// CHECK-LABEL: @concat_non_axis_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]
func.func @concat_non_axis_dyn(%arg0: tensor<5x?xf32>, %arg1: tensor<6x?xf32>) -> () {
  // CHECK-DAG: %[[AXIS:.+]] = arith.constant 0
  // CHECK-DAG: %[[IDX1:.+]] = arith.constant 1
  // CHECK-DAG: %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[IDX1]]
  // CHECK-DAG: %[[INIT:.+]] = tensor.empty(%[[DIM0]]) : tensor<11x?xf32>
  // CHECK-DAG: %[[IDX1_1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[IDX1_1]]
  // CHECK-DAG: %[[INSERT0:.+]] = tensor.insert_slice %[[ARG0]] into %[[INIT]][0, 0] [5, %[[DIM1]]] [1, 1]
  // CHECK-DAG: %[[IDX1_2:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[DIM2:.+]] = tensor.dim %[[ARG1]], %[[IDX1_2]] : tensor<6x?xf32>
  // CHECK: %[[INSERT1:.+]] = tensor.insert_slice %[[ARG1]] into %[[INSERT0]][5, 0] [6, %[[DIM2]]] [1, 1]
  %0 = "tosa.concat"(%arg0, %arg1) { axis = 0 : i32} : (tensor<5x?xf32>, tensor<6x?xf32>)  -> (tensor<11x?xf32>)
  return
}

// -----

// CHECK-LABEL: @concat_axis_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @concat_axis_dyn(%arg0: tensor<?x3xf32>, %arg1: tensor<?x3xf32>) -> () {
  // CHECK-DAG: %[[AXIS:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[IDX0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[IDX0]] : tensor<?x3xf32>
  // CHECK-DAG: %[[DIM1:.+]] = tensor.dim %[[ARG1]], %[[AXIS]] : tensor<?x3xf32>
  // CHECK-DAG: %[[SUM:.+]] = arith.addi %[[DIM0]], %[[DIM1]] : index
  // CHECK-DAG: %[[INIT:.+]] = tensor.empty(%[[SUM]]) : tensor<?x3xf32>
  // CHECK-DAG: %[[IDX0_1:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[DIM2:.+]] = tensor.dim %[[ARG0]], %[[IDX0_1]] : tensor<?x3xf32>
  // CHECK-DAG: %[[INSERT0:.+]] = tensor.insert_slice %[[ARG0]] into %[[INIT]][0, 0] [%[[DIM2]], 3] [1, 1] : tensor<?x3xf32> into tensor<?x3xf32>
  // CHECK-DAG: %[[IDX0_2:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[DIM3:.+]] = tensor.dim %[[ARG1]], %[[IDX0_2]] : tensor<?x3xf32>
  // CHECK: %[[INSERT1:.+]] = tensor.insert_slice %[[ARG1]] into %[[INSERT0]][%[[DIM0]], 0] [%[[DIM3]], 3] [1, 1] : tensor<?x3xf32> into tensor<?x3xf32>

  %0 = "tosa.concat"(%arg0, %arg1) { axis = 0 : i32} : (tensor<?x3xf32>, tensor<?x3xf32>)  -> (tensor<?x3xf32>)
  return
}

// -----

// CHECK-LABEL: @concat_axis_dyn_mixed
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG2:[0-9a-zA-Z_]*]]:
func.func @concat_axis_dyn_mixed(%arg0: tensor<?x1xf32>, %arg1: tensor<?x1xf32>, %arg2: tensor<?x1xf32>) -> () {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C0_0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[OFFSET0:.+]] = tensor.dim %[[ARG0]], %[[C0_0]] : tensor<?x1xf32>
  // CHECK-DAG: %[[DIM1_0:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x1xf32>
  // CHECK-DAG: %[[OFFSET1:.+]] = arith.addi %[[OFFSET0]], %[[DIM1_0]] : index
  // CHECK-DAG: %[[DIM2_2:.+]] = tensor.dim %[[ARG2]], %[[C0]] : tensor<?x1xf32>
  // CHECK-DAG: %[[OFFSET2:.+]] = arith.addi %[[OFFSET1]], %[[DIM2_2]] : index
  // CHECK-DAG: %[[INIT:.+]] = tensor.empty() : tensor<5x1xf32>
  // CHECK-DAG: %[[C0_3:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[DIM_4:.+]] = tensor.dim %[[ARG0]], %[[C0_3]] : tensor<?x1xf32>
  // CHECK-DAG: %[[INSERT0:.+]] = tensor.insert_slice %[[ARG0]] into %[[INIT]][0, 0] [%[[DIM_4]], 1] [1, 1] : tensor<?x1xf32> into tensor<5x1xf32>
  // CHECK-DAG: %[[C0_4:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[DIM_6:.+]] = tensor.dim %[[ARG1]], %[[C0_4]] : tensor<?x1xf32>
  // CHECK-DAG: %[[INSERT1:.+]] = tensor.insert_slice %[[ARG1]] into %[[INSERT0]][%[[OFFSET0]], 0] [%[[DIM_6]], 1] [1, 1] : tensor<?x1xf32> into tensor<5x1xf32>
  // CHECK-DAG: %[[C0_8:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[DIM_9:.+]] = tensor.dim %[[ARG2]], %[[C0_8]] : tensor<?x1xf32>
  // CHECK-DAG: %[[INSERT3:.+]] = tensor.insert_slice %[[ARG2]] into %[[INSERT1]][%[[OFFSET1]], 0] [%[[DIM_9]], 1] [1, 1] : tensor<?x1xf32> into tensor<5x1xf32>

  // CHECK: return

  %0 = "tosa.concat"(%arg0, %arg1, %arg2) <{axis = 0 : i32}> : (tensor<?x1xf32>, tensor<?x1xf32>, tensor<?x1xf32>) -> tensor<5x1xf32>
  return
}

// -----

// CHECK-LABEL: @concat_non_axis_dyn_mixed
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG2:[0-9a-zA-Z_]*]]:
func.func @concat_non_axis_dyn_mixed(%arg0: tensor<?x1xf32>, %arg1: tensor<?x1xf32>, %arg2: tensor<?x1xf32>) -> () {
  // CHECK-DAG: %[[UNUSED0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[UNUSED1:.+]] = tensor.dim %[[ARG0]], %[[UNUSED0]] : tensor<?x1xf32>

  // CHECK-DAG: %[[INIT:.+]] = tensor.empty() : tensor<5x3xf32>
  // CHECK-DAG: %[[C0_0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[DIM0_0:.+]] = tensor.dim %[[ARG0]], %[[C0_0]] : tensor<?x1xf32>
  // CHECK-DAG: %[[INSERT0:.+]] = tensor.insert_slice %[[ARG0]] into %[[INIT]][0, 0] [%[[DIM0_0]], 1] [1, 1] : tensor<?x1xf32> into tensor<5x3xf32>
  // CHECK-DAG: %[[C0_1:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[DIM1_0:.+]] = tensor.dim %[[ARG1]], %[[C0_1]] : tensor<?x1xf32>
  // CHECK-DAG: %[[INSERT1:.+]] = tensor.insert_slice %[[ARG1]] into %[[INSERT0]][0, 1] [%[[DIM1_0]], 1] [1, 1] : tensor<?x1xf32> into tensor<5x3xf32>
  // CHECK-DAG: %[[C0_2:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[DIM2_0:.+]] = tensor.dim %[[ARG2]], %[[C0_2]] : tensor<?x1xf32>
  // CHECK-DAG: %[[INSERT2:.+]] = tensor.insert_slice %[[ARG2]] into %[[INSERT1]][0, 2] [%[[DIM2_0]], 1] [1, 1] : tensor<?x1xf32> into tensor<5x3xf32>
  // CHECK: return

  %0 = "tosa.concat"(%arg0, %arg1, %arg2) <{axis = 1 : i32}> : (tensor<?x1xf32>, tensor<?x1xf32>, tensor<?x1xf32>) -> tensor<5x3xf32>
  return
}
