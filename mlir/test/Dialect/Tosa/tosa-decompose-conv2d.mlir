// RUN: mlir-opt --split-input-file --tosa-optional-decompositions %s | FileCheck %s

// -----

// CHECK-LABEL: @conv2d_as_fully_connected
func.func @conv2d_as_fully_connected(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<3x1x1x2xf32>, %arg2: tensor<3xf32>) -> tensor<4x10x10x3xf32> {
  // CHECK-NOT: tosa.conv2d
  // CHECK: %[[VAR0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 400, 2>}
  // CHECK-SAME: -> tensor<400x2xf32>
  // CHECK: %[[VAR1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 3, 2>}
  // CHECK-SAME: -> tensor<3x2xf32>
  // CHECK: %[[VAR2:.*]] = tosa.fully_connected %[[VAR0]], %[[VAR1]], %arg2
  // CHECK-SAME: -> tensor<400x3xf32>
  // CHECK: %[[VAR3:.*]] = tosa.reshape %[[VAR2]] {new_shape = array<i64: 4, 10, 10, 3>}
  // CHECK-SAME: -> tensor<4x10x10x3xf32>
  // CHECK: return %[[VAR3]]
  %0 = tosa.conv2d %arg0, %arg1, %arg2 {pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<4x10x10x2xf32>, tensor<3x1x1x2xf32>, tensor<3xf32>) -> tensor<4x10x10x3xf32>
  return %0 : tensor<4x10x10x3xf32>
}

// -----

// CHECK-LABEL: @conv2d_as_fully_connected_quant
func.func @conv2d_as_fully_connected_quant(%arg0: tensor<4x10x10x2xi8>, %arg1: tensor<3x1x1x2xi8>, %arg2: tensor<3xi32>) -> tensor<4x10x10x3xi32> {
  // CHECK-NOT: tosa.conv2d
  // CHECK: %[[VAR0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 400, 2>}
  // CHECK-SAME: -> tensor<400x2xi8>
  // CHECK: %[[VAR1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 3, 2>}
  // CHECK-SAME: -> tensor<3x2xi8>
  // CHECK: %[[VAR2:.*]] = tosa.fully_connected %[[VAR0]], %[[VAR1]], %arg2
  // CHECK-SAME: quantization_info = #tosa.conv_quant<input_zp = 42, weight_zp = 24>
  // CHECK-SAME: -> tensor<400x3xi32>
  // CHECK: %[[VAR3:.*]] = tosa.reshape %[[VAR2]] {new_shape = array<i64: 4, 10, 10, 3>}
  // CHECK-SAME: -> tensor<4x10x10x3xi32>
  // CHECK: return %[[VAR3]]
  %0 = tosa.conv2d %arg0, %arg1, %arg2 {pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>, quantization_info = #tosa.conv_quant<input_zp = 42, weight_zp = 24>} : (tensor<4x10x10x2xi8>, tensor<3x1x1x2xi8>, tensor<3xi32>) -> tensor<4x10x10x3xi32>
  return %0 : tensor<4x10x10x3xi32>
}

// -----

// CHECK-LABEL:   func.func @conv_with_dynamic_dim(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<?x14x14x64xi8>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: tensor<384x1x1x64xi8>,
// CHECK-SAME:                                     %[[VAL_2:.*]]: tensor<384xi32>) -> tensor<?x14x14x384xi32> {
func.func @conv_with_dynamic_dim(%arg0: tensor<?x14x14x64xi8>, %arg1: tensor<384x1x1x64xi8>, %arg2: tensor<384xi32>) -> tensor<?x14x14x384xi32> {
// CHECK:           %[[VAL_3:.*]] = tosa.reshape %[[VAL_0]] {new_shape = array<i64: -1, 64>} : (tensor<?x14x14x64xi8>) -> tensor<?x64xi8>
// CHECK:           %[[VAL_4:.*]] = tosa.reshape %[[VAL_1]] {new_shape = array<i64: 384, 64>} : (tensor<384x1x1x64xi8>) -> tensor<384x64xi8>
// CHECK:           %[[VAL_5:.*]] = tosa.fully_connected %[[VAL_3]], %[[VAL_4]], %[[VAL_2]] {quantization_info = #tosa.conv_quant<input_zp = -6, weight_zp = 11>} : (tensor<?x64xi8>, tensor<384x64xi8>, tensor<384xi32>) -> tensor<?x384xi32>
// CHECK:           %[[VAL_6:.*]] = tosa.reshape %[[VAL_5]] {new_shape = array<i64: -1, 14, 14, 384>} : (tensor<?x384xi32>) -> tensor<?x14x14x384xi32>
// CHECK:           return %[[VAL_6]] : tensor<?x14x14x384xi32>
// CHECK:         }
  %0 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, quantization_info = #tosa.conv_quant<input_zp = -6, weight_zp = 11>, stride = array<i64: 1, 1>} : (tensor<?x14x14x64xi8>, tensor<384x1x1x64xi8>, tensor<384xi32>) -> tensor<?x14x14x384xi32>
  return %0 : tensor<?x14x14x384xi32>
}

// -----

// CHECK-LABEL: @conv2d_as_fully_connected_padded
func.func @conv2d_as_fully_connected_padded(%arg0: tensor<4x10x10x2xi8>, %arg1: tensor<3x1x1x2xi8>, %arg2: tensor<3xi32>) -> tensor<4x12x12x3xi32> {
  // CHECK-DAG: %[[PAD_SHAPE:.+]] = "tosa.const"() <{value = dense<{{\[\[}}0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>}
  // CHECK-DAG: %[[PAD_VAL:.+]] = "tosa.const"() <{value = dense<42> : tensor<i8>}
  // CHECK-DAG: %[[PAD:.+]] = tosa.pad %arg0, %[[PAD_SHAPE]], %[[PAD_VAL]] : (tensor<4x10x10x2xi8>, tensor<4x2xi64>, tensor<i8>) -> tensor<4x12x12x2xi8>
  // CHECK-DAG: %[[RESHAPE_INPUT:.+]] = tosa.reshape %[[PAD]] {new_shape = array<i64: 576, 2>}
  // CHECK-DAG: %[[RESHAPE_FILTER:.+]] = tosa.reshape %arg1 {new_shape = array<i64: 3, 2>}
  // CHECK-DAG: %[[FULLY:.+]] = tosa.fully_connected %[[RESHAPE_INPUT]], %[[RESHAPE_FILTER]], %arg2 {quantization_info = #tosa.conv_quant<input_zp = 42, weight_zp = 24>}
  // CHECK: %[[RESHAPE:.+]] = tosa.reshape %[[FULLY]] {new_shape = array<i64: 4, 12, 12, 3>}
  %0 = tosa.conv2d %arg0, %arg1, %arg2 {pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>, quantization_info = #tosa.conv_quant<input_zp = 42, weight_zp = 24>} : (tensor<4x10x10x2xi8>, tensor<3x1x1x2xi8>, tensor<3xi32>) -> tensor<4x12x12x3xi32>
  return %0 : tensor<4x12x12x3xi32>
}
