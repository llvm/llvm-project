// RUN: mlir-opt --split-input-file --tosa-optional-decompositions %s | FileCheck %s

// CHECK-LABEL: @transpose_conv2d
func.func @transpose_conv2d(%arg0: tensor<2x16x14x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) -> tensor<2x18x19x5xf32> {
  // CHECK: %[[REV1:.+]] = "tosa.reverse"(%arg1) <{axis = 1 : i64}
  // CHECK: %[[REV2:.+]] = "tosa.reverse"(%[[REV1]]) <{axis = 2 : i64}
  // CHECK: "tosa.conv2d"(%arg0, %[[REV2]], %arg2)
  // CHECK-SAME: dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 5, 5>, stride = array<i64: 1, 1>
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<2x16x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x18x19x5xf32>
  return %0 : tensor<2x18x19x5xf32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_quantized

func.func @transpose_conv2d_quantized(%arg0: tensor<2x16x14x3xi8>, %arg1: tensor<5x3x6x3xi8>, %arg2: tensor<5xi32>) -> (tensor<2x18x19x5xi32>) {
  // CHECK: %[[REV1:.+]] = "tosa.reverse"(%arg1) <{axis = 1 : i64}
  // CHECK: %[[REV2:.+]] = "tosa.reverse"(%[[REV1]]) <{axis = 2 : i64}
  // CHECK: "tosa.conv2d"(%arg0, %[[REV2]], %arg2) <{dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 5, 5>, quantization_info = #tosa.conv_quant<input_zp = -22, weight_zp = 42>, stride = array<i64: 1, 1>}
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {out_pad = array<i64: 0, 0, 0, 0>, quantization_info = #tosa.conv_quant<input_zp = -22, weight_zp = 42>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<2x16x14x3xi8>, tensor<5x3x6x3xi8>, tensor<5xi32>) -> tensor<2x18x19x5xi32>
  return %0 : tensor<2x18x19x5xi32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_quantized_padded
func.func @transpose_conv2d_quantized_padded(%arg0: tensor<2x16x14x3xi8>, %arg1: tensor<5x3x6x3xi8>, %arg2: tensor<5xi32>) -> (tensor<2x21x26x5xi32>) {
  // CHECK-DAG: %[[REV0:.+]] = "tosa.reverse"(%0) <{axis = 2 : i64}
  // CHECK-DAG: %[[REV1:.+]] = "tosa.reverse"(%arg1) <{axis = 1 : i64}
  // CHECK: "tosa.conv2d"(%arg0, %1, %arg2)
  // CHECK-SAME: dilation = array<i64: 1, 1>, pad = array<i64: 3, 4, 8, 9>,
  // CHECK-SAME: quantization_info = #tosa.conv_quant<input_zp = -22, weight_zp = 42>, stride = array<i64: 1, 1>}
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {
    out_pad = array<i64: 1, 2, 3, 4>,
    quantization_info = #tosa.conv_quant<input_zp = -22, weight_zp = 42>,
    out_shape = array<i64: -1, -1, -1, -1>,
    stride = array<i64: 1, 1>} : (tensor<2x16x14x3xi8>, tensor<5x3x6x3xi8>, tensor<5xi32>) -> tensor<2x21x26x5xi32>
  return %0 : tensor<2x21x26x5xi32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_strided
func.func @transpose_conv2d_strided(%arg0: tensor<2x17x15x3xf32>, %arg1: tensor<5x3x5x3xf32>, %arg2: tensor<5xf32>) -> tensor<2x?x?x5xf32> {
  // Manipulate the weight matrix to handle striding.
  // CHECK-DAG: %[[PADV:.+]]  = "tosa.const"() <{value = dense<{{\[\[}}0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi32>}
  // CHECK-DAG: %[[TRANSV:.+]]  = "tosa.const"() <{value = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[PADW:.+]]  = "tosa.pad"(%arg1, %[[PADV]])
  // CHECK-DAG: %[[RESW1:.+]]  = "tosa.reshape"(%[[PADW]]) <{new_shape = array<i64: 5, 2, 2, 2, 3, 3>}
  // CHECK-DAG: %[[TRANS:.+]]  = "tosa.transpose"(%[[RESW1]], %[[TRANSV]])
  // CHECK-DAG: %[[RESW2:.+]]  = "tosa.reshape"(%[[TRANS]]) <{new_shape = array<i64: 30, 2, 2, 3>}
  // CHECK-DAG: %[[REV1:.+]]  = "tosa.reverse"(%[[RESW2]]) <{axis = 1 : i64}
  // CHECK-DAG: %[[NEWWEIGHT:.+]] = "tosa.reverse"(%[[REV1]]) <{axis = 2 : i64}

  // Pad out the input matrix to handle the transpose conv.
  // CHECK-DAG: %[[PAD:.+]]  = "tosa.const"() <{value = dense<{{\[\[}}0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>}
  // CHECK-DAG: %[[TRANS2:.+]]  = "tosa.const"() <{value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[NEWINPUT:.+]] = "tosa.pad"(%arg0, %[[PAD]])

  // Manipulate the final shape.
  // CHECK-DAG: %[[BIAS:.+]]  = "tosa.const"() <{value = dense<0.000000e+00> : tensor<30xf32>}
  // CHECK-DAG: %[[CONV:.+]] = "tosa.conv2d"(%[[NEWINPUT]], %[[NEWWEIGHT]], %[[BIAS]]) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
  // CHECK-DAG: %[[RESHAPE_OUT_1:.+]] = "tosa.reshape"(%[[CONV]]) <{new_shape = array<i64: 2, 18, 16, 2, 3, 5>}
  // CHECK-DAG: %[[TRANS_OUT:.+]] = "tosa.transpose"(%[[RESHAPE_OUT_1]], %[[TRANS2]])
  // CHECK-DAG: %[[RESHAPE_OUT_2:.+]] = "tosa.reshape"(%[[TRANS_OUT]]) <{new_shape = array<i64: 2, 36, 48, 5>}
  // CHECK-DAG: %[[SLICE:.+]] = "tosa.slice"(%[[RESHAPE_OUT_2]]) <{size = array<i64: 2, 35, 47, 5>, start = array<i64: 0, 0, 0, 0>}
  // CHECK-DAG: %[[RESHAPE_ARG2:.+]] = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 1, 1, 5>}
  // CHECK: %[[ADD:.+]] = "tosa.add"(%[[SLICE]], %[[RESHAPE_ARG2]])
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 2, 3>} : (tensor<2x17x15x3xf32>, tensor<5x3x5x3xf32>, tensor<5xf32>) -> tensor<2x35x47x5xf32>
  %1 = tensor.cast %0 : tensor<2x35x47x5xf32> to tensor<2x?x?x5xf32>
  return %1 : tensor<2x?x?x5xf32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_strided_quantized
func.func @transpose_conv2d_strided_quantized(%arg0: tensor<2x17x15x3xi8>, %arg1: tensor<5x3x5x3xi8>, %arg2: tensor<5xi32>) -> (tensor<2x35x47x5xi32>) {
  // Manipulate the weight matrix to handle striding.
  // CHECK-DAG: %[[PADV:.+]]  = "tosa.const"() <{value = dense<{{\[\[}}0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi32>}
  // CHECK-DAG: %[[TRANSV:.+]]  = "tosa.const"() <{value = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[PADW:.+]]  = "tosa.pad"(%arg1, %[[PADV]]) <{quantization_info = #tosa.pad_quant<input_zp = 42>}
  // CHECK-DAG: %[[RESW1:.+]]  = "tosa.reshape"(%[[PADW]]) <{new_shape = array<i64: 5, 2, 2, 2, 3, 3>}
  // CHECK-DAG: %[[TRANS:.+]]  = "tosa.transpose"(%[[RESW1]], %[[TRANSV]])
  // CHECK-DAG: %[[RESW2:.+]]  = "tosa.reshape"(%[[TRANS]]) <{new_shape = array<i64: 30, 2, 2, 3>}
  // CHECK-DAG: %[[REV1:.+]]  = "tosa.reverse"(%[[RESW2]]) <{axis = 1 : i64}
  // CHECK-DAG: %[[NEWWEIGHT:.+]] = "tosa.reverse"(%[[REV1]]) <{axis = 2 : i64}

  // Pad out the input matrix to handle the transpose conv.
  // CHECK-DAG: %[[PAD:.+]]  = "tosa.const"() <{value = dense<{{\[\[}}0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>}
  // CHECK-DAG: %[[TRANS2:.+]]  = "tosa.const"() <{value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[NEWINPUT:.+]] = "tosa.pad"(%arg0, %[[PAD]]) <{quantization_info = #tosa.pad_quant<input_zp = -22>}

  // Manipulate the final shape.
  // CHECK-DAG: %[[BIAS:.+]]  = "tosa.const"() <{value = dense<0> : tensor<30xi32>}
  // CHECK-DAG: %[[CONV:.+]] = "tosa.conv2d"(%[[NEWINPUT]], %[[NEWWEIGHT]], %[[BIAS]]) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, quantization_info = #tosa.conv_quant<input_zp = -22, weight_zp = 42>, stride = array<i64: 1, 1>}
  // CHECK-DAG: %[[RESHAPE_OUT_1:.+]] = "tosa.reshape"(%[[CONV]]) <{new_shape = array<i64: 2, 18, 16, 2, 3, 5>}
  // CHECK-DAG: %[[TRANS_OUT:.+]] = "tosa.transpose"(%[[RESHAPE_OUT_1]], %[[TRANS2]])
  // CHECK-DAG: %[[RESHAPE_OUT_2:.+]] = "tosa.reshape"(%[[TRANS_OUT]]) <{new_shape = array<i64: 2, 36, 48, 5>}
  // CHECK-DAG: %[[SLICE:.+]] = "tosa.slice"(%[[RESHAPE_OUT_2]]) <{size = array<i64: 2, 35, 47, 5>, start = array<i64: 0, 0, 0, 0>}
  // CHECK-DAG: %[[RESHAPE_ARG2:.+]] = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 1, 1, 5>}
  // CHECK: %[[ADD:.+]] = "tosa.add"(%[[SLICE]], %[[RESHAPE_ARG2]])
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {out_pad = array<i64: 0, 0, 0, 0>, quantization_info = #tosa.conv_quant<input_zp = -22, weight_zp = 42>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 2, 3>} : (tensor<2x17x15x3xi8>, tensor<5x3x5x3xi8>, tensor<5xi32>) -> tensor<2x35x47x5xi32>
  return %0 : tensor<2x35x47x5xi32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_strided_overpad
func.func @transpose_conv2d_strided_overpad(%arg0 : tensor<1x16x1x1xi8>, %arg1 : tensor<1x2x1x1xi8>, %arg2 : tensor<1xi32>) -> (tensor<1x19x2x1xi32>) {
  // CHECK: %[[WEIGHT_PAD:.+]] = "tosa.const"()
  // CHECK-SAME{literal}: value = dense<[[0, 0], [0, 0], [0, 1], [0, 0]]> : tensor<4x2xi32>
  // CHECK: %[[WEIGHT_PERMS:.+]] = "tosa.const"() <{value = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi32>}
  // CHECK: %[[INPUT_PAD:.+]] = "tosa.const"()
  // CHECK-SAME{literal}: value = dense<[[0, 0], [1, 1], [0, 0], [0, 0]]> : tensor<4x2xi32>}
  // CHECK: %[[ZERO:.+]] = "tosa.const"() <{value = dense<0> : tensor<2xi32>}
  // CHECK: %[[RESULT_PERMS:.+]] = "tosa.const"() <{value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
  // CHECK: %[[RESULT_PAD:.+]] = "tosa.const"()
  // CHECK-SAME{literal}: value = dense<[[0, 0], [2, 0], [0, 0], [0, 0]]> : tensor<4x2xi32>}
  // CHECK: %[[PAD_WEIGHT:.+]] = "tosa.pad"(%arg1, %[[WEIGHT_PAD]]) <{quantization_info = #tosa.pad_quant<input_zp = 93>}
  // CHECK: %[[RESHAPE_WEIGHT_0:.+]] = "tosa.reshape"(%[[PAD_WEIGHT]]) <{new_shape = array<i64: 1, 2, 1, 1, 2, 1>}
  // CHECK: %[[TRANSPOSE_WEIGHT:.+]] = "tosa.transpose"(%[[RESHAPE_WEIGHT_0]], %[[WEIGHT_PERMS]])
  // CHECK: %[[RESHAPE_WEIGHT_1:.+]] = "tosa.reshape"(%[[TRANSPOSE_WEIGHT]]) <{new_shape = array<i64: 2, 2, 1, 1>}
  // CHECK: %[[REVERSE:.+]] = "tosa.reverse"(%[[RESHAPE_WEIGHT_1]]) <{axis = 1 : i64}
  // CHECK: %[[PAD_INPUT:.+]] = "tosa.pad"(%arg0, %[[INPUT_PAD]]) <{quantization_info = #tosa.pad_quant<input_zp = -103>}
  // CHECK: %[[CONV:.+]] = "tosa.conv2d"(%[[PAD_INPUT]], %[[REVERSE]], %[[ZERO]])
  // CHECK-SAME{literal}: dilation = [1, 1], pad = [0, 0, 0, 0], quantization_info = #tosa.conv_quant<input_zp = -103, weight_zp = 93>, stride = [1, 1]}
  // CHECK: %[[RESHAPE_RESULT_0:.+]] = "tosa.reshape"(%[[CONV]]) <{new_shape = array<i64: 1, 17, 1, 1, 2, 1>}
  // CHECK: %[[TRANSPOSE_RESULT:.+]] = "tosa.transpose"(%[[RESHAPE_RESULT_0]], %[[RESULT_PERMS]])
  // CHECK: %[[RESHAPE_RESULT_1:.+]] = "tosa.reshape"(%[[TRANSPOSE_RESULT]]) <{new_shape = array<i64: 1, 17, 2, 1>}
  // CHECK: %[[PAD_RESULT:.+]] = "tosa.pad"(%[[RESHAPE_RESULT_1]], %[[RESULT_PAD]])
  // CHECK: %[[RESHAPE_ARG2:.+]] = "tosa.reshape"(%arg2) <{new_shape = array<i64: 1, 1, 1, 1>}
  // CHECK: %[[ADD:.+]] = "tosa.add"(%[[PAD_RESULT]], %[[RESHAPE_ARG2]])
  %2 =  "tosa.transpose_conv2d"(%arg0, %arg1, %arg2)  {
    out_pad = array<i64: 2, 0, 0, 1>,
    out_shape = array<i64: 1, -1, -1, 1>,
    stride = array<i64: 1, 2>,
    quantization_info = #tosa.conv_quant<input_zp = -103, weight_zp = 93>} :
    (tensor<1x16x1x1xi8>, tensor<1x2x1x1xi8>, tensor<1xi32>) -> (tensor<1x19x2x1xi32>)
  "func.return" (%2) : (tensor<1x19x2x1xi32>) -> ()
}
