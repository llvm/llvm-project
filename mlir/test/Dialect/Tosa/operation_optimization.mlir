// RUN: mlir-opt --split-input-file --tosa-optimization %s | FileCheck %s

// -----

// CHECK-LABEL: @conv2d_as_fully_connected
func @conv2d_as_fully_connected(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<3x1x1x2xf32>, %arg2: tensor<3xf32>) -> tensor<4x10x10x3xf32> {
  // CHECK-NOT: "tosa.conv2d"
  // CHECK: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = [400, 2]}
  // CHECK-SAME: -> tensor<400x2xf32>
  // CHECK: %[[VAR1:.*]] = "tosa.reshape"(%arg1) {new_shape = [3, 2]}
  // CHECK-SAME: -> tensor<3x2xf32>
  // CHECK: %[[VAR2:.*]] = "tosa.fully_connected"(%[[VAR0]], %[[VAR1]], %arg2)
  // CHECK-SAME: -> tensor<400x3xf32>
  // CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = [4, 10, 10, 3]}
  // CHECK-SAME: -> tensor<4x10x10x3xf32>
  // CHECK: return %[[VAR3]]
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<4x10x10x2xf32>, tensor<3x1x1x2xf32>, tensor<3xf32>) -> tensor<4x10x10x3xf32>
  return %0 : tensor<4x10x10x3xf32>
}

// -----

// CHECK-LABEL: @conv2d_as_fully_connected_quant
func @conv2d_as_fully_connected_quant(%arg0: tensor<4x10x10x2xi8>, %arg1: tensor<3x1x1x2xi8>, %arg2: tensor<3xi32>) -> tensor<4x10x10x3xi32> {
  // CHECK-NOT: "tosa.conv2d"
  // CHECK: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = [400, 2]}
  // CHECK-SAME: -> tensor<400x2xi8>
  // CHECK: %[[VAR1:.*]] = "tosa.reshape"(%arg1) {new_shape = [3, 2]}
  // CHECK-SAME: -> tensor<3x2xi8>
  // CHECK: %[[VAR2:.*]] = "tosa.fully_connected"(%[[VAR0]], %[[VAR1]], %arg2)
  // CHECK-SAME: quantization_info = {input_zp = 42 : i32, weight_zp = 24 : i32}
  // CHECK-SAME: -> tensor<400x3xi32>
  // CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = [4, 10, 10, 3]}
  // CHECK-SAME: -> tensor<4x10x10x3xi32>
  // CHECK: return %[[VAR3]]
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], quantization_info = {input_zp = 42 : i32, weight_zp = 24 : i32}} : (tensor<4x10x10x2xi8>, tensor<3x1x1x2xi8>, tensor<3xi32>) -> tensor<4x10x10x3xi32>
  return %0 : tensor<4x10x10x3xi32>
}

// -----

// CHECK-LABEL: @depthwise_conv2d_as_mul
func @depthwise_conv2d_as_mul(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<1x1x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<4x10x10x6xf32> {
  // CHECK-NOT: "tosa.depthwise_conv2d"
  // CHECK: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = [4, 10, 10, 2, 1]}
  // CHECK-SAME: -> tensor<4x10x10x2x1xf32>
  // CHECK: %[[VAR1:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 1, 1, 2, 3]}
  // CHECK-SAME: -> tensor<1x1x1x2x3xf32>
  // CHECK: %[[VAR2:.*]] = "tosa.mul"(%[[VAR0]], %[[VAR1]])
  // CHECK-SAME: -> tensor<4x10x10x2x3xf32>
  // CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = [4, 10, 10, 6]}
  // CHECK-SAME: -> tensor<4x10x10x6xf32>
  // CHECK: %[[VAR4:.*]] = "tosa.add"(%[[VAR3]], %arg2)
  // CHECK-SAME: -> tensor<4x10x10x6xf32>
  // CHECK: return %[[VAR4]]
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<4x10x10x2xf32>, tensor<1x1x2x3xf32>, tensor<6xf32>) -> tensor<4x10x10x6xf32>
  return %0 : tensor<4x10x10x6xf32>
}

// -----

// CHECK-LABEL: @depthwise_conv2d_as_mul_q
func @depthwise_conv2d_as_mul_q(%arg0: tensor<4x10x10x2xi8>, %arg1: tensor<1x1x2x3xi8>, %arg2: tensor<6xi32>) -> tensor<4x10x10x6xi32> {
  // CHECK: "tosa.depthwise_conv2d"
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], quantization_info = {input_zp = 0 : i32, weight_zp = 0 : i32}} : (tensor<4x10x10x2xi8>, tensor<1x1x2x3xi8>, tensor<6xi32>) -> tensor<4x10x10x6xi32>
  return %0 : tensor<4x10x10x6xi32>
}

// -----
