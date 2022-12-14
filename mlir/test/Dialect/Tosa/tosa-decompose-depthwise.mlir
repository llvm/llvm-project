// RUN: mlir-opt --split-input-file --tosa-optional-decompositions %s | FileCheck %s

// -----

// CHECK-LABEL: @depthwise_conv2d_as_mul
func.func @depthwise_conv2d_as_mul(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<1x1x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<4x10x10x6xf32> {
  // CHECK-NOT: "tosa.depthwise_conv2d"
  // CHECK: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = [4, 10, 10, 2, 1]}
  // CHECK-SAME: -> tensor<4x10x10x2x1xf32>
  // CHECK: %[[VAR2:.*]] = "tosa.mul"(%[[VAR0]], %arg1)
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
func.func @depthwise_conv2d_as_mul_q(%arg0: tensor<4x10x10x2xi8>, %arg1: tensor<1x1x2x3xi8>, %arg2: tensor<6xi32>) -> tensor<4x10x10x6xi32> {
  // CHECK: %[[iZp:.+]] = "tosa.const"() {value = dense<7> : tensor<i32>}
  // CHECK: %[[wZp:.+]] = "tosa.const"() {value = dense<11> : tensor<i32>}
  // CHECK: %[[rIn:.+]] = "tosa.reshape"(%arg0) {new_shape = [4, 10, 10, 2, 1]}
  // CHECK: %[[cIn:.+]] = "tosa.cast"(%[[rIn]]) : (tensor<4x10x10x2x1xi8>) -> tensor<4x10x10x2x1xi32>
  // CHECK: %[[cWe:.+]] = "tosa.cast"(%arg1) : (tensor<1x1x2x3xi8>) -> tensor<1x1x2x3xi32>
  // CHECK: %[[sIn:.+]] = "tosa.sub"(%[[cIn]], %[[iZp]])
  // CHECK: %[[sWe:.+]] = "tosa.sub"(%[[cWe]], %[[wZp]])
  // CHECK: %[[mul:.+]] = "tosa.mul"(%[[sIn]], %[[sWe]]) {shift = 0 : i32}
  // CHECK: %[[reO:.+]] = "tosa.reshape"(%[[mul]]) {new_shape = [4, 10, 10, 6]}
  // CHECK: %[[add:.+]] = "tosa.add"(%[[reO]], %arg2)
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], quantization_info = #tosa.conv_quant<input_zp = 7, weight_zp = 11>} : (tensor<4x10x10x2xi8>, tensor<1x1x2x3xi8>, tensor<6xi32>) -> tensor<4x10x10x6xi32>
  return %0 : tensor<4x10x10x6xi32>
}

// -----

// CHECK-LABEL: @depthwise_conv2d_as_mul_padded
func.func @depthwise_conv2d_as_mul_padded(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<1x1x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<4x12x12x6xf32> {
  // CHECK: %[[pad:.+]] = "tosa.const"() {value = dense<{{\[\[}}0, 0], [1, 1], [1, 1], [0, 0], [0, 0]]> : tensor<5x2xi64>}
  // CHECK: %[[zero:.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
  // CHECK: %[[reIn:.+]] = "tosa.reshape"(%arg0) {new_shape = [4, 10, 10, 2, 1]}
  // CHECK: %[[padded:.+]] = "tosa.pad"(%[[reIn]], %[[pad]], %[[zero]]) : (tensor<4x10x10x2x1xf32>, tensor<5x2xi64>, tensor<f32>) -> tensor<4x12x12x2x1xf32>
  // CHECK: %[[mul:.+]] = "tosa.mul"(%3, %arg1) {shift = 0 : i32}
  // CHECK: %[[reOut:.+]] = "tosa.reshape"(%[[mul]]) {new_shape = [4, 12, 12, 6]}
  // CHECK: %[[add:.+]] = "tosa.add"(%[[reOut]], %arg2)
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [1, 1, 1, 1], stride = [1, 1], dilation = [1, 1]} : (tensor<4x10x10x2xf32>, tensor<1x1x2x3xf32>, tensor<6xf32>) -> tensor<4x12x12x6xf32>
  return %0 : tensor<4x12x12x6xf32>
}
