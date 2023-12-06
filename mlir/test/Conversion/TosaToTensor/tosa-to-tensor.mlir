// RUN: mlir-opt --split-input-file --tosa-to-tensor %s -o -| FileCheck %s

// -----

// CHECK-LABEL: test_reshape_0d_same_s2s_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<f32>
// CHECK: return %[[ARG_0]] : tensor<f32>
func.func @test_reshape_0d_same_s2s_explicit(%arg0: tensor<f32>) -> tensor<f32> {
  %s = tosa.const_shape { value = dense<> : tensor<0xindex> } : () -> !tosa.shape<0>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<f32>, !tosa.shape<0>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: test_reshape_0d_up_s2d_auto
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<f32>
// CHECK: %[[VAL_0:.*]] = tensor.expand_shape %[[ARG_0]] [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK: %[[VAL_1:.*]] = tensor.cast %[[VAL_0]] : tensor<1xf32> to tensor<?xf32>
// CHECK: return %[[VAL_1]] : tensor<?xf32>
func.func @test_reshape_0d_up_s2d_auto(%arg0: tensor<f32>) -> tensor<?xf32> {
  %s = tosa.const_shape { value = dense<-1> : tensor<1xindex> } : () -> !tosa.shape<1>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<f32>, !tosa.shape<1>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: test_reshape_0d_up_s2d_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<f32>
// CHECK: %[[VAL_0:.*]] = tensor.expand_shape %[[ARG_0]] [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK: %[[VAL_1:.*]] = tensor.cast %[[VAL_0]] : tensor<1xf32> to tensor<?xf32>
// CHECK: return %[[VAL_1]] : tensor<?xf32>
func.func @test_reshape_0d_up_s2d_explicit(%arg0: tensor<f32>) -> tensor<?xf32> {
  %s = tosa.const_shape { value = dense<1> : tensor<1xindex> } : () -> !tosa.shape<1>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<f32>, !tosa.shape<1>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: test_reshape_0d_up_s2s_auto
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<f32>
// CHECK: %[[VAL_0:.*]] = tensor.expand_shape %[[ARG_0]] [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK: return %[[VAL_0]] : tensor<1xf32>
func.func @test_reshape_0d_up_s2s_auto(%arg0: tensor<f32>) -> tensor<1xf32> {
  %s = tosa.const_shape { value = dense<-1> : tensor<1xindex> } : () -> !tosa.shape<1>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<f32>, !tosa.shape<1>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: test_reshape_0d_up_s2s_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<f32>
// CHECK: %[[VAL_0:.*]] = tensor.expand_shape %[[ARG_0]] [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK: return %[[VAL_0]] : tensor<1xf32>
func.func @test_reshape_0d_up_s2s_explicit(%arg0: tensor<f32>) -> tensor<1xf32> {
  %s = tosa.const_shape { value = dense<1> : tensor<1xindex> } : () -> !tosa.shape<1>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<f32>, !tosa.shape<1>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: test_reshape_1d_down_d2s_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?xf32>
// CHECK: %[[VAL_0:.*]] = tensor.cast %[[ARG_0]] : tensor<?xf32> to tensor<1xf32>
// CHECK: %[[VAL_1:.*]] = tensor.collapse_shape %[[VAL_0]] [] : tensor<1xf32> into tensor<f32>
// CHECK: return %[[VAL_1]] : tensor<f32>
func.func @test_reshape_1d_down_d2s_explicit(%arg0: tensor<?xf32>) -> tensor<f32> {
  %s = tosa.const_shape { value = dense<> : tensor<0xindex> } : () -> !tosa.shape<0>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<?xf32>, !tosa.shape<0>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: test_reshape_1d_down_s2s_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<1xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] [] : tensor<1xf32> into tensor<f32>
// CHECK: return %[[VAL_0]] : tensor<f32>
func.func @test_reshape_1d_down_s2s_explicit(%arg0: tensor<1xf32>) -> tensor<f32> {
  %s = tosa.const_shape { value = dense<> : tensor<0xindex> } : () -> !tosa.shape<0>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<1xf32>, !tosa.shape<0>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: test_reshape_1d_up_d2d_auto
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?xf32>
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[VAL_0:.*]] = arith.divsi %[[DIM]], %[[C2]] : index
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[ARG_0]] {{\[\[}}0, 1]] output_shape [2, %[[VAL_0]]] : tensor<?xf32> into tensor<2x?xf32>
// CHECK: return %[[EXPANDED]] : tensor<2x?xf32>
func.func @test_reshape_1d_up_d2d_auto(%arg0: tensor<?xf32>) -> tensor<2x?xf32> {
  %s = tosa.const_shape { value = dense<[2, -1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<?xf32>, !tosa.shape<2>) -> tensor<2x?xf32>
  return %0 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: test_reshape_1d_up_s2s_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<6xf32>
// CHECK: %[[VAL_0:.*]] = tensor.expand_shape %[[ARG_0]] {{\[\[}}0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>
// CHECK: return %[[VAL_0]] : tensor<2x3xf32>
func.func @test_reshape_1d_up_s2s_explicit(%arg0: tensor<6xf32>) -> tensor<2x3xf32> {
  %s = tosa.const_shape { value = dense<[2, 3]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<6xf32>, !tosa.shape<2>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: test_reshape_2d_down_d2d_auto
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<2x?xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1]] : tensor<2x?xf32> into tensor<?xf32>
// CHECK: return %[[VAL_0]] : tensor<?xf32>
func.func @test_reshape_2d_down_d2d_auto(%arg0: tensor<2x?xf32>) -> tensor<?xf32> {
  %s = tosa.const_shape { value = dense<-1> : tensor<1xindex> } : () -> !tosa.shape<1>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<2x?xf32>, !tosa.shape<1>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: test_reshape_2d_down_s2s_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<2x3xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1]] : tensor<2x3xf32> into tensor<6xf32>
// CHECK: return %[[VAL_0]] : tensor<6xf32>
func.func @test_reshape_2d_down_s2s_explicit(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  %s = tosa.const_shape { value = dense<6> : tensor<1xindex> } : () -> !tosa.shape<1>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<2x3xf32>, !tosa.shape<1>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// -----

// CHECK-LABEL: test_reshape_2d_same_d2d_auto
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?x2xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1]] : tensor<?x2xf32> into tensor<?xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[VAL_0]], %[[C0]] : tensor<?xf32>
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[DIV:.*]] = arith.divsi %[[DIM]], %[[C2]] : index
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1]] output_shape [2, %[[DIV]]] : tensor<?xf32> into tensor<2x?xf32>
// CHECK: return %[[EXPANDED]] : tensor<2x?xf32>
func.func @test_reshape_2d_same_d2d_auto(%arg0: tensor<?x2xf32>) -> tensor<2x?xf32> {
  %s = tosa.const_shape { value = dense<[2, -1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<?x2xf32>, !tosa.shape<2>) -> tensor<2x?xf32>
  return %0 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: test_reshape_2d_same_s2d_auto
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<2x4xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1]] : tensor<2x4xf32> into tensor<8xf32>
// CHECK: %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1]] output_shape [4, 2] : tensor<8xf32> into tensor<4x2xf32>
// CHECK: %[[VAL_2:.*]] = tensor.cast %[[VAL_1]] : tensor<4x2xf32> to tensor<?x2xf32>
// CHECK: return %[[VAL_2]] : tensor<?x2xf32>
func.func @test_reshape_2d_same_s2d_auto(%arg0: tensor<2x4xf32>) -> tensor<?x2xf32> {
  %s = tosa.const_shape { value = dense<[-1, 2]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<2x4xf32>, !tosa.shape<2>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}


// -----

// CHECK-LABEL: test_reshape_2d_same_s2d_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<2x4xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1]] : tensor<2x4xf32> into tensor<8xf32>
// CHECK: %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1]] output_shape [4, 2] : tensor<8xf32> into tensor<4x2xf32>
// CHECK: %[[VAL_2:.*]] = tensor.cast %[[VAL_1]] : tensor<4x2xf32> to tensor<?x2xf32>
// CHECK: return %[[VAL_2]] : tensor<?x2xf32>
func.func @test_reshape_2d_same_s2d_explicit(%arg0: tensor<2x4xf32>) -> tensor<?x2xf32> {
  %s = tosa.const_shape { value = dense<[4, 2]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<2x4xf32>, !tosa.shape<2>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: test_reshape_2d_same_s2s_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<3x2xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1]] : tensor<3x2xf32> into tensor<6xf32>
// CHECK: %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>
// CHECK: return %[[VAL_1]] : tensor<2x3xf32>
func.func @test_reshape_2d_same_s2s_explicit(%arg0: tensor<3x2xf32>) -> tensor<2x3xf32> {
  %s = tosa.const_shape { value = dense<[2, 3]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<3x2xf32>, !tosa.shape<2>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: test_reshape_3d_same_d2d_auto_empty
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<3x2x?xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1, 2]] : tensor<3x2x?xf32> into tensor<?xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[VAL_0]], %[[C0]] : tensor<?xf32>
// CHECK: %[[C0_0:.*]] = arith.constant 0 : index
// CHECK: %[[DIV:.*]] = arith.divsi %[[DIM]], %[[C0_0]] : index
// CHECK: %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1, 2]] output_shape [0, 3, %[[DIV]]] : tensor<?xf32> into tensor<0x3x?xf32>
// CHECK: %[[VAL_2:.*]] = tensor.cast %[[VAL_1]] : tensor<0x3x?xf32> to tensor<?x?x?xf32>
// CHECK: return %[[VAL_2]] : tensor<?x?x?xf32>
func.func @test_reshape_3d_same_d2d_auto_empty(%arg0: tensor<3x2x?xf32>) -> tensor<?x?x?xf32> {
  %s = tosa.const_shape { value = dense<[0, 3, -1]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<3x2x?xf32>, !tosa.shape<3>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: test_reshape_3d_same_d2d_auto
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<2x?x?xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1, 2]] : tensor<2x?x?xf32> into tensor<?xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[VAL_0]], %[[C0]] : tensor<?xf32>
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[DIV:.*]] = arith.divsi %[[DIM]], %[[C8]] : index
// CHECK: %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1, 2]] output_shape [2, %[[DIV]], 4] : tensor<?xf32> into tensor<2x?x4xf32>
// CHECK: %[[VAL_2:.*]] = tensor.cast %[[VAL_1]] : tensor<2x?x4xf32> to tensor<?x?x?xf32>
// CHECK: return %[[VAL_2]] : tensor<?x?x?xf32>
func.func @test_reshape_3d_same_d2d_auto(%arg0: tensor<2x?x?xf32>) -> tensor<?x?x?xf32> {
  %s = tosa.const_shape { value = dense<[2, -1, 4]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<2x?x?xf32>, !tosa.shape<3>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: test_reshape_3d_same_d2d_auto_identity
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?x3x4xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1, 2]] : tensor<?x3x4xf32> into tensor<?xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[VAL_0]], %[[C0]] : tensor<?xf32>
// CHECK: %[[C6:.*]] = arith.constant 6 : index
// CHECK: %[[DIV:.*]] = arith.divsi %[[DIM]], %[[C6]] : index
// CHECK: %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1, 2]] output_shape [2, 3, %[[DIV]]] : tensor<?xf32> into tensor<2x3x?xf32>
// CHECK: return %[[VAL_1]] : tensor<2x3x?xf32>
func.func @test_reshape_3d_same_d2d_auto_identity(%arg0: tensor<?x3x4xf32>) -> tensor<2x3x?xf32> {
  %s = tosa.const_shape { value = dense<[2, 3, -1]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<?x3x4xf32>, !tosa.shape<3>) -> tensor<2x3x?xf32>
  return %0 : tensor<2x3x?xf32>
}

// -----

// CHECK-LABEL: test_reshape_3d_same_d2d_explicit_empty
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<3x2x?xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1, 2]] : tensor<3x2x?xf32> into tensor<?xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[VAL_0]], %[[C0]] : tensor<?xf32>
// CHECK: %[[C6:.*]] = arith.constant 6 : index
// CHECK: %[[DIV:.*]] = arith.divsi %[[DIM]], %[[C6]] : index
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1, 2]] output_shape [%[[DIV]], 3, 2] : tensor<?xf32> into tensor<?x3x2xf32>
// CHECK: %[[VAL_2:.*]] = tensor.cast %[[EXPANDED]] : tensor<?x3x2xf32> to tensor<?x?x?xf32>
// CHECK: return %[[VAL_2]] : tensor<?x?x?xf32>
func.func @test_reshape_3d_same_d2d_explicit_empty(%arg0: tensor<3x2x?xf32>) -> tensor<?x?x?xf32> {
  %s = tosa.const_shape { value = dense<[0, 3, 2]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<3x2x?xf32>, !tosa.shape<3>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: test_reshape_3d_same_d2d_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1, 2]] : tensor<?x?x?xf32> into tensor<?xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[VAL_0]], %[[C0]] : tensor<?xf32>
// CHECK: %[[C12:.*]] = arith.constant 12 : index
// CHECK: %[[DIV:.*]] = arith.divsi %[[DIM]], %[[C12]] : index
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1, 2]] output_shape [%[[DIV]], 3, 4] : tensor<?xf32> into tensor<?x3x4xf32>
// CHECK: %[[VAL_2:.*]] = tensor.cast %[[EXPANDED]] : tensor<?x3x4xf32> to tensor<?x?x?xf32>
// CHECK: return %[[VAL_2]] : tensor<?x?x?xf32>
func.func @test_reshape_3d_same_d2d_explicit(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %s = tosa.const_shape { value = dense<[2, 3, 4]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<?x?x?xf32>, !tosa.shape<3>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: test_reshape_3d_same_d2d_explicit_identity
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?x3x4xf32>
// CHECK: %[[VAL_0:.*]] = tensor.cast %[[ARG_0]] : tensor<?x3x4xf32> to tensor<2x3x?xf32>
// CHECK: return %[[VAL_0]] : tensor<2x3x?xf32>
func.func @test_reshape_3d_same_d2d_explicit_identity(%arg0: tensor<?x3x4xf32>) -> tensor<2x3x?xf32> {
  %s = tosa.const_shape { value = dense<[2, 3, 4]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<?x3x4xf32>, !tosa.shape<3>) -> tensor<2x3x?xf32>
  return %0 : tensor<2x3x?xf32>
}

// -----

// CHECK-LABEL: test_reshape_3d_same_d2s_auto
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1, 2]] : tensor<?x?x?xf32> into tensor<?xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[VAL_0]], %[[C0]] : tensor<?xf32>
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[DIV:.*]] = arith.divsi %[[DIM]], %[[C8]] : index
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1, 2]] output_shape [2, %[[DIV]], 4] : tensor<?xf32> into tensor<2x?x4xf32>
// CHECK: %[[VAL_2:.*]] = tensor.cast %[[EXPANDED]] : tensor<2x?x4xf32> to tensor<2x3x4xf32>
// CHECK: return %[[VAL_2]] : tensor<2x3x4xf32>
func.func @test_reshape_3d_same_d2s_auto(%arg0: tensor<?x?x?xf32>) -> tensor<2x3x4xf32> {
  %s = tosa.const_shape { value = dense<[2, -1, 4]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<?x?x?xf32>, !tosa.shape<3>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: test_reshape_3d_same_d2s_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1, 2]] : tensor<?x?x?xf32> into tensor<?xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[VAL_0]], %[[C0]] : tensor<?xf32>
// CHECK: %[[C12:.*]] = arith.constant 12 : index
// CHECK: %[[DIV:.*]] = arith.divsi %[[DIM]], %[[C12]] : index
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1, 2]] output_shape [%[[DIV]], 3, 4] : tensor<?xf32> into tensor<?x3x4xf32>
// CHECK: %[[VAL_2:.*]] = tensor.cast %[[EXPANDED]] : tensor<?x3x4xf32> to tensor<2x3x4xf32>
// CHECK: return %[[VAL_2]] : tensor<2x3x4xf32>
func.func @test_reshape_3d_same_d2s_explicit(%arg0: tensor<?x?x?xf32>) -> tensor<2x3x4xf32> {
  %s = tosa.const_shape { value = dense<[2, 3, 4]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<?x?x?xf32>, !tosa.shape<3>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: test_reshape_3d_same_s2s_explicit_identity
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<2x3x4xf32>
// CHECK: return %[[ARG_0]] : tensor<2x3x4xf32>
func.func @test_reshape_3d_same_s2s_explicit_identity(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %s = tosa.const_shape { value = dense<[2, 3, 4]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<2x3x4xf32>, !tosa.shape<3>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: test_reshape_3d_up_d2s_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1, 2]] : tensor<?x?x?xf32> into tensor<?xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[COLLAPSED]], %[[C0]] : tensor<?xf32>
// CHECK: %[[C6:.*]] = arith.constant 6 : index
// CHECK: %[[VAL_0:.*]] = arith.divsi %[[DIM]], %[[C6]] : index
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[COLLAPSED]] {{\[\[}}0, 1, 2, 3]] output_shape [%[[VAL_0]], 3, 2, 1] : tensor<?xf32> into tensor<?x3x2x1xf32>
// CHECK: %[[CAST:.*]] = tensor.cast %[[EXPANDED]] : tensor<?x3x2x1xf32> to tensor<1x3x2x1xf32>
// CHECK: return %[[CAST]] : tensor<1x3x2x1xf32>
func.func @test_reshape_3d_up_d2s_explicit(%input: tensor<?x?x?xf32>) -> tensor<1x3x2x1xf32> {
  %s = tosa.const_shape { value = dense<[1, 3, 2, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %0 = tosa.reshape %input, %s : (tensor<?x?x?xf32>, !tosa.shape<4>) -> tensor<1x3x2x1xf32>
  return %0 : tensor<1x3x2x1xf32>
}

// -----

// CHECK-LABEL: test_reshape_4d_down_d2s_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?x?x?x?xf32>
// CHECK: %[[VAL_0:.*]] = tensor.cast %[[ARG_0]] : tensor<?x?x?x?xf32> to tensor<1x1x1x1xf32>
// CHECK: %[[VAL_1:.*]] = tensor.collapse_shape %[[VAL_0]] [] : tensor<1x1x1x1xf32> into tensor<f32>
// CHECK: return %[[VAL_1]] : tensor<f32>
func.func @test_reshape_4d_down_d2s_explicit(%arg0: tensor<?x?x?x?xf32>) -> tensor<f32> {
  %s = tosa.const_shape { value = dense<> : tensor<0xindex> } : () -> !tosa.shape<0>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<?x?x?x?xf32>, !tosa.shape<0>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: test_reshape_5d_down_d2d_auto
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?x?x?x2x3xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1, 2, 3, 4]] : tensor<?x?x?x2x3xf32> into tensor<?xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[COLLAPSED]], %[[C0]] : tensor<?xf32>
// CHECK: %[[C6:.*]] = arith.constant 6 : index
// CHECK: %[[VAL_0:.*]] = arith.divsi %[[DIM]], %[[C6]] : index
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[COLLAPSED]] {{\[\[}}0, 1, 2]] output_shape [%[[VAL_0]], 2, 3] : tensor<?xf32> into tensor<?x2x3xf32>
// CHECK: return %[[EXPANDED]] : tensor<?x2x3xf32>
func.func @test_reshape_5d_down_d2d_auto(%arg0: tensor<?x?x?x2x3xf32>) -> tensor<?x2x3xf32> {
  %s = tosa.const_shape { value = dense<[-1, 2, 3]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<?x?x?x2x3xf32>, !tosa.shape<3>) -> tensor<?x2x3xf32>
  return %0 : tensor<?x2x3xf32>
}

// -----

// CHECK-LABEL: test_reshape_6d_down_d2d_auto
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<1x2x?x5x7x11xf32>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1, 2, 3, 4, 5]] : tensor<1x2x?x5x7x11xf32> into tensor<?xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[COLLAPSED]], %[[C0]] : tensor<?xf32>
// CHECK: %[[C385:.*]] = arith.constant 385 : index
// CHECK: %[[VAL_0:.*]] = arith.divsi %[[DIM]], %[[C385]] : index
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[COLLAPSED]] {{\[\[}}0, 1, 2]] output_shape [%[[VAL_0]], 5, 77] : tensor<?xf32> into tensor<?x5x77xf32>
// CHECK: return %[[EXPANDED]] : tensor<?x5x77xf32>
func.func @test_reshape_6d_down_d2d_auto(%arg0: tensor<1x2x?x5x7x11xf32>) -> tensor<?x5x77xf32> {
  %s = tosa.const_shape { value = dense<[-1, 5, 77]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<1x2x?x5x7x11xf32>, !tosa.shape<3>) -> tensor<?x5x77xf32>
  return %0 : tensor<?x5x77xf32>
}

// -----

// CHECK-LABEL: test_reshape_6d_down_s2s_auto
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<1x2x3x5x7x11xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1, 2], [3], [4, 5]] : tensor<1x2x3x5x7x11xf32> into tensor<6x5x77xf32>
// CHECK: return %[[VAL_0]] : tensor<6x5x77xf32>
func.func @test_reshape_6d_down_s2s_auto(%arg0: tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32> {
  %s = tosa.const_shape { value = dense<[6, 5, -1]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<1x2x3x5x7x11xf32>, !tosa.shape<3>) -> tensor<6x5x77xf32>
  return %0 : tensor<6x5x77xf32>
}

// -----

// This test would previously fail on GCC with certain compiler flags.
// The GCC issue would cause invalid IR after tosa-to-tensor, so this test
// locks down that the code goes through tosa-to-tensor and verifies.
//
// See https://github.com/llvm/llvm-project/pull/91521 for a full description.

// -----

// CHECK-LABEL: reshape_bug_fix
// CHECK: tensor.expand_shape
func.func @reshape_bug_fix(%arg0: tensor<?xf32>) -> tensor<1x1x1x?xf32> {
  %1 = "tosa.const_shape"() {value = dense<[1, 1, 1, -1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %0 = "tosa.reshape"(%arg0, %1) : (tensor<?xf32>, !tosa.shape<4>) -> tensor<1x1x1x?xf32>
  return %0 : tensor<1x1x1x?xf32>
}

// -----

// CHECK-LABEL: test_reshape_6d_down_s2s_explicit
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<1x2x3x5x7x11xf32>
// CHECK: %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG_0]] {{\[\[}}0, 1, 2], [3], [4, 5]] : tensor<1x2x3x5x7x11xf32> into tensor<6x5x77xf32>
// CHECK: return %[[VAL_0]] : tensor<6x5x77xf32>
func.func @test_reshape_6d_down_s2s_explicit(%arg0: tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32> {
  %s = tosa.const_shape { value = dense<[6, 5, 77]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = "tosa.reshape"(%arg0, %s) : (tensor<1x2x3x5x7x11xf32>, !tosa.shape<3>) -> tensor<6x5x77xf32>
  return %0 : tensor<6x5x77xf32>
}

// -----

// CHECK-LABEL: @test_reshape_samerank_unsigned
// CHECK-SAME: (%[[VAL_0:.*]]: tensor<3x2xui8>)
func.func @test_reshape_samerank_unsigned(%arg0: tensor<3x2xui8>) -> tensor<2x3xui8> {
  // CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<3x2xui8> to tensor<3x2xi8>
  // CHECK: %[[RESHAPE1:.*]] = tensor.collapse_shape %[[CAST1]] {{\[}}[0, 1]] : tensor<3x2xi8> into tensor<6xi8>
  // CHECK: %[[RESHAPE2:.*]] = tensor.expand_shape %[[RESHAPE1]] {{\[}}[0, 1]] output_shape {{\[}}2, 3] : tensor<6xi8> into tensor<2x3xi8>
  // CHECK: %[[CAST2:.*]] = builtin.unrealized_conversion_cast %[[RESHAPE2]] : tensor<2x3xi8> to tensor<2x3xui8
  %s = tosa.const_shape { value = dense<[2, 3]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %0 = "tosa.reshape"(%arg0, %s): (tensor<3x2xui8>, !tosa.shape<2>) -> tensor<2x3xui8>
  return %0 : tensor<2x3xui8>
}

// -----

// CHECK-LABEL: func @slice
func.func @slice(%arg0: tensor<6xf32>) ->() {
  // CHECK: [[SLICE:%.+]] = tensor.extract_slice %arg0[2] [1] [1]
  %0 = tosa.const_shape  {value = dense<2> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = tosa.const_shape  {value = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = tosa.slice %arg0, %0, %1 : (tensor<6xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<1xf32>
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
  %0 = tosa.const_shape  {value = dense<2> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = tosa.const_shape  {value = dense<-1> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = tosa.slice %arg0, %0, %1 : (tensor<?xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<?xf32>
  return %2 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @pad_float
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @pad_float(%arg0 : tensor<1x2xf32>) -> (tensor<4x9xf32>) {
  %0 = tosa.const_shape {value = dense<[1, 2, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK-DAG: [[INDEX1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[INDEX2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[INDEX3:%.+]] = arith.constant 3 : index
  // CHECK-DAG: [[INDEX4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: tensor.pad %[[ARG0]] low{{\[}}[[INDEX1]], [[INDEX3]]] high{{\[}}[[INDEX2]], [[INDEX4]]]  {
  // CHECK:   tensor.yield [[CST]]
  // CHECK: } : tensor<1x2xf32> to tensor<4x9xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xf32>, !tosa.shape<4>)  -> (tensor<4x9xf32>)
  return %1 : tensor<4x9xf32>
}
// -----

func.func @pad_int(%arg0 : tensor<1x2xi32>) -> (tensor<4x9xi32>) {
  %0 = tosa.const_shape {value = dense<[1, 2, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK: [[CST:%.+]] = arith.constant 0 : i32
  // CHECK: tensor.pad
  // CHECK:   tensor.yield [[CST]]
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xi32>, !tosa.shape<4>)  -> (tensor<4x9xi32>)
  return %1 : tensor<4x9xi32>
}
// -----

func.func @pad_quant(%arg0 : tensor<1x2xi32>) -> (tensor<4x9xi32>) {
  %0 = tosa.const_shape {value = dense<[1, 2, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK: [[CST:%.+]] = arith.constant 42 : i32
  // CHECK: tensor.pad
  // CHECK:   tensor.yield [[CST]]
  %1 = "tosa.pad"(%arg0, %0) {quantization_info = #tosa.pad_quant<input_zp = 42>} : (tensor<1x2xi32>, !tosa.shape<4>)  -> (tensor<4x9xi32>)
  return %1 : tensor<4x9xi32>
}

// -----

func.func @pad_float_explicit(%arg0 : tensor<1x2xf32>) -> (tensor<4x9xf32>) {
  %0 = tosa.const_shape {value = dense<[1, 2, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK-DAG: [[INDEX1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[INDEX2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[INDEX3:%.+]] = arith.constant 3 : index
  // CHECK-DAG: [[INDEX4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[CST:%.+]] = arith.constant 4.200000e+01 : f32
  // CHECK: tensor.pad %[[ARG0]] low{{\[}}[[INDEX1]], [[INDEX3]]] high{{\[}}[[INDEX2]], [[INDEX4]]]  {
  // CHECK:   tensor.yield [[CST]]
  // CHECK: } : tensor<1x2xf32> to tensor<4x9xf32>
  %1 = arith.constant dense<42.0> : tensor<f32>
  %2 = "tosa.pad"(%arg0, %0, %1)  : (tensor<1x2xf32>, !tosa.shape<4>, tensor<f32>)  -> (tensor<4x9xf32>)
  return %2 : tensor<4x9xf32>
}

// -----

func.func @pad_dyn_input(%arg0 : tensor<?x2xf32>) -> (tensor<?x9xf32>) {
  %0 = tosa.const_shape {value = dense<[1, 2, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK-DAG: [[INDEX1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[INDEX2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[INDEX3:%.+]] = arith.constant 3 : index
  // CHECK-DAG: [[INDEX4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: tensor.pad %[[ARG0]] low{{\[}}[[INDEX1]], [[INDEX3]]] high{{\[}}[[INDEX2]], [[INDEX4]]]  {
  // CHECK:   tensor.yield [[CST]]
  // CHECK: } : tensor<?x2xf32> to tensor<?x9xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<?x2xf32>, !tosa.shape<4>)  -> (tensor<?x9xf32>)
  return %1 : tensor<?x9xf32>
}
// -----

func.func @pad_dyn_padding(%arg0 : tensor<1x2xf32>) -> (tensor<?x9xf32>) {
  %0 = tosa.const_shape {value = dense<[-1, 2, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK-DAG: [[INDEX1:%.+]] = arith.constant -1 : index
  // CHECK-DAG: [[INDEX2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[INDEX3:%.+]] = arith.constant 3 : index
  // CHECK-DAG: [[INDEX4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: tensor.pad %[[ARG0]] low{{\[}}[[INDEX1]], [[INDEX3]]] high{{\[}}[[INDEX2]], [[INDEX4]]]  {
  // CHECK:   tensor.yield [[CST]]
  // CHECK: } : tensor<1x2xf32> to tensor<?x9xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xf32>, !tosa.shape<4>)  -> (tensor<?x9xf32>)
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
