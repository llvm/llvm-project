// RUN: mlir-opt --split-input-file --tosa-decompose-transpose-conv %s | FileCheck %s

// CHECK-LABEL: @transpose_conv2d
func @transpose_conv2d(%arg0: tensor<2x16x14x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) -> tensor<2x?x?x5xf32> {
  // CHECK: %[[REV1:.+]] = "tosa.reverse"(%arg1) {axis = 1 : i64}
  // CHECK: %[[REV2:.+]] = "tosa.reverse"(%[[REV1]]) {axis = 2 : i64}
  // CHECK: "tosa.conv2d"(%arg0, %[[REV2]], %arg2) {dilation = [1, 1], pad = [2, 2, 5, 5], stride = [1, 1]}
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], out_pad = [0, 0], out_shape = [-1, -1, -1, -1], stride = [1, 1]} : (tensor<2x16x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x18x19x5xf32>
  %1 = tensor.cast %0 : tensor<2x18x19x5xf32> to tensor<2x?x?x5xf32>
  return %1 : tensor<2x?x?x5xf32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_quantized
func @transpose_conv2d_quantized(%arg0: tensor<2x16x14x3xi8>, %arg1: tensor<5x3x6x3xi8>, %arg2: tensor<5xi32>) -> (tensor<2x18x19x5xi32>) {
  // CHECK: %[[REV1:.+]] = "tosa.reverse"(%arg1) {axis = 1 : i64}
  // CHECK: %[[REV2:.+]] = "tosa.reverse"(%[[REV1]]) {axis = 2 : i64}
  // CHECK: "tosa.conv2d"(%arg0, %[[REV2]], %arg2) {dilation = [1, 1], pad = [2, 2, 5, 5], quantization_info = {input_zp = -22 : i32, weight_zp = 42 : i32}, stride = [1, 1]}
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], out_pad = [0, 0], quantization_info = {input_zp = -22 : i32, weight_zp = 42 : i32}, out_shape = [-1, -1, -1, -1], stride = [1, 1]} : (tensor<2x16x14x3xi8>, tensor<5x3x6x3xi8>, tensor<5xi32>) -> tensor<2x18x19x5xi32>
  return %0 : tensor<2x18x19x5xi32>
}

// ----

// CHECK-LABEL: @transpose_conv2d_dilated
func @transpose_conv2d_dilated(%arg0: tensor<2x16x14x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) -> tensor<2x?x?x5xf32> {
  // CHECK: %[[REV1:.+]] = "tosa.reverse"(%arg1) {axis = 1 : i64}
  // CHECK: %[[REV2:.+]] = "tosa.reverse"(%[[REV1]]) {axis = 2 : i64}
  // CHECK: "tosa.conv2d"(%arg0, %[[REV2]], %arg2) {dilation = [2, 3], pad = [4, 4, 15, 15], stride = [1, 1]}
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {dilation = [2, 3], out_pad = [0, 0], out_shape = [-1, -1, -1, -1], stride = [1, 1]} : (tensor<2x16x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x20x29x5xf32>
  %1 = tensor.cast %0 : tensor<2x20x29x5xf32> to tensor<2x?x?x5xf32>
  return %1 : tensor<2x?x?x5xf32>
}

// ----

// CHECK-LABEL: @transpose_conv2d_strided
func @transpose_conv2d_strided(%arg0: tensor<2x17x15x3xf32>, %arg1: tensor<5x3x5x3xf32>, %arg2: tensor<5xf32>) -> tensor<2x?x?x5xf32> {
  // Manipulate the weight matrix to handle striding.
  // CHECK-DAG: %[[PADV:.+]]  = "tosa.const"() {value = dense<{{\[\[}}0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi32>}
  // CHECK-DAG: %[[TRANSV:.+]]  = "tosa.const"() {value = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[PADW:.+]]  = "tosa.pad"(%arg1, %[[PADV]])
  // CHECK-DAG: %[[RESW1:.+]]  = "tosa.reshape"(%[[PADW]]) {new_shape = [5, 2, 2, 2, 3, 3]}
  // CHECK-DAG: %[[TRANS:.+]]  = "tosa.transpose"(%[[RESW1]], %[[TRANSV]])
  // CHECK-DAG: %[[RESW2:.+]]  = "tosa.reshape"(%[[TRANS]]) {new_shape = [30, 2, 2, 3]}
  // CHECK-DAG: %[[REV1:.+]]  = "tosa.reverse"(%[[RESW2]]) {axis = 1 : i64}
  // CHECK-DAG: %[[NEWWEIGHT:.+]] = "tosa.reverse"(%[[REV1]]) {axis = 2 : i64}

  // Pad out the input matrix to handle the transpose conv.
  // CHECK-DAG: %[[PAD:.+]]  = "tosa.const"() {value = dense<{{\[\[}}0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>}
  // CHECK-DAG: %[[TRANS2:.+]]  = "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[NEWINPUT:.+]] = "tosa.pad"(%arg0, %[[PAD]])

  // Manipulate the final shape.
  // CHECK-DAG: %[[BIAS:.+]]  = "tosa.const"() {value = dense<0.000000e+00> : tensor<30xf32>}
  // CHECK-DAG: %[[CONV:.+]] = "tosa.conv2d"(%[[NEWINPUT]], %[[NEWWEIGHT]], %[[BIAS]]) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]}
  // CHECK-DAG: %[[RESHAPE_OUT_1:.+]] = "tosa.reshape"(%[[CONV]]) {new_shape = [2, 18, 16, 2, 3, 5]}
  // CHECK-DAG: %[[TRANS_OUT:.+]] = "tosa.transpose"(%[[RESHAPE_OUT_1]], %[[TRANS2]])
  // CHECK-DAG: %[[RESHAPE_OUT_2:.+]] = "tosa.reshape"(%[[TRANS_OUT]]) {new_shape = [2, 36, 48, 5]}
  // CHECK-DAG: %[[SLICE:.+]] = "tosa.slice"(%[[RESHAPE_OUT_2]]) {size = [2, 35, 47, 5], start = [0, 0, 0, 0]}
  // CHECK: %[[ADD:.+]] = "tosa.add"(%[[SLICE]], %arg2)
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], out_pad = [0, 0], out_shape = [-1, -1, -1, -1], stride = [2, 3]} : (tensor<2x17x15x3xf32>, tensor<5x3x5x3xf32>, tensor<5xf32>) -> tensor<2x35x47x5xf32>
  %1 = tensor.cast %0 : tensor<2x35x47x5xf32> to tensor<2x?x?x5xf32>
  return %1 : tensor<2x?x?x5xf32>
}

// ----

// CHECK-LABEL: @transpose_conv2d_strided_quantized
func @transpose_conv2d_strided_quantized(%arg0: tensor<2x17x15x3xi8>, %arg1: tensor<5x3x5x3xi8>, %arg2: tensor<5xi32>) -> (tensor<2x35x47x5xi32>) {
  // Manipulate the weight matrix to handle striding.
  // CHECK-DAG: %[[PADV:.+]]  = "tosa.const"() {value = dense<{{\[\[}}0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi32>}
  // CHECK-DAG: %[[TRANSV:.+]]  = "tosa.const"() {value = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[PADW:.+]]  = "tosa.pad"(%arg1, %[[PADV]]) {quantization_info = {input_zp = 42 : i32}}
  // CHECK-DAG: %[[RESW1:.+]]  = "tosa.reshape"(%[[PADW]]) {new_shape = [5, 2, 2, 2, 3, 3]}
  // CHECK-DAG: %[[TRANS:.+]]  = "tosa.transpose"(%[[RESW1]], %[[TRANSV]])
  // CHECK-DAG: %[[RESW2:.+]]  = "tosa.reshape"(%[[TRANS]]) {new_shape = [30, 2, 2, 3]}
  // CHECK-DAG: %[[REV1:.+]]  = "tosa.reverse"(%[[RESW2]]) {axis = 1 : i64}
  // CHECK-DAG: %[[NEWWEIGHT:.+]] = "tosa.reverse"(%[[REV1]]) {axis = 2 : i64}

  // Pad out the input matrix to handle the transpose conv.
  // CHECK-DAG: %[[PAD:.+]]  = "tosa.const"() {value = dense<{{\[\[}}0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>}
  // CHECK-DAG: %[[TRANS2:.+]]  = "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[NEWINPUT:.+]] = "tosa.pad"(%arg0, %[[PAD]]) {quantization_info = {input_zp = -22 : i32}}

  // Manipulate the final shape.
  // CHECK-DAG: %[[BIAS:.+]]  = "tosa.const"() {value = dense<0> : tensor<30xi32>}
  // CHECK-DAG: %[[CONV:.+]] = "tosa.conv2d"(%[[NEWINPUT]], %[[NEWWEIGHT]], %[[BIAS]]) {dilation = [1, 1], pad = [0, 0, 0, 0], quantization_info = {input_zp = -22 : i32, weight_zp = 42 : i32}, stride = [1, 1]}
  // CHECK-DAG: %[[RESHAPE_OUT_1:.+]] = "tosa.reshape"(%[[CONV]]) {new_shape = [2, 18, 16, 2, 3, 5]}
  // CHECK-DAG: %[[TRANS_OUT:.+]] = "tosa.transpose"(%[[RESHAPE_OUT_1]], %[[TRANS2]])
  // CHECK-DAG: %[[RESHAPE_OUT_2:.+]] = "tosa.reshape"(%[[TRANS_OUT]]) {new_shape = [2, 36, 48, 5]}
  // CHECK-DAG: %[[SLICE:.+]] = "tosa.slice"(%[[RESHAPE_OUT_2]]) {size = [2, 35, 47, 5], start = [0, 0, 0, 0]}
  // CHECK: %[[ADD:.+]] = "tosa.add"(%[[SLICE]], %arg2)
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], out_pad = [0, 0], quantization_info = {input_zp = -22 : i32, weight_zp = 42 : i32}, out_shape = [-1, -1, -1, -1], stride = [2, 3]} : (tensor<2x17x15x3xi8>, tensor<5x3x5x3xi8>, tensor<5xi32>) -> tensor<2x35x47x5xi32>
  return %0 : tensor<2x35x47x5xi32>
}
