// RUN: mlir-opt --split-input-file --tosa-optional-decompositions %s | FileCheck %s

// CHECK-LABEL: @transpose_conv2d
func.func @transpose_conv2d(%arg0: tensor<2x16x14x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) -> tensor<2x18x19x5xf32> {
  // CHECK: %[[REV1:.+]] = tosa.reverse %arg1 {axis = 1 : i32}
  // CHECK: %[[REV2:.+]] = tosa.reverse %[[REV1]] {axis = 2 : i32}
  // CHECK: tosa.conv2d %arg0, %[[REV2]], %arg2
  // CHECK-SAME: dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 5, 5>, stride = array<i64: 1, 1>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2{acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<2x16x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x18x19x5xf32>
  return %0 : tensor<2x18x19x5xf32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_quantized

func.func @transpose_conv2d_quantized(%arg0: tensor<2x16x14x3xi8>, %arg1: tensor<5x3x6x3xi8>, %arg2: tensor<5xi32>) -> (tensor<2x18x19x5xi32>) {
  // CHECK-DAG: %[[INPUT_ZP:.+]]  = "tosa.const"() <{value = dense<-6> : tensor<1xi8>}
  // CHECK-DAG: %[[WEIGHT_ZP:.+]]  = "tosa.const"() <{value = dense<11> : tensor<1xi8>}
  // CHECK: %[[REV1:.+]] = tosa.reverse %arg1 {axis = 1 : i32}
  // CHECK: %[[REV2:.+]] = tosa.reverse %[[REV1]] {axis = 2 : i32}
  // CHECK: tosa.conv2d %arg0, %[[REV2]], %arg2, %[[INPUT_ZP]], %[[WEIGHT_ZP]] {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 5, 5>, stride = array<i64: 1, 1>}
  %input_zp = "tosa.const"() {value = dense<-6> : tensor<1xi8>} : () -> tensor<1xi8>
  %weight_zp = "tosa.const"() {value = dense<11> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = i32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<2x16x14x3xi8>, tensor<5x3x6x3xi8>, tensor<5xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<2x18x19x5xi32>
  return %0 : tensor<2x18x19x5xi32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_quantized_padded
func.func @transpose_conv2d_quantized_padded(%arg0: tensor<2x16x14x3xi8>, %arg1: tensor<5x3x6x3xi8>, %arg2: tensor<5xi32>) -> (tensor<2x21x26x5xi32>) {
  // CHECK-DAG: %[[INPUT_ZP:.+]]  = "tosa.const"() <{value = dense<-22> : tensor<1xi8>}
  // CHECK-DAG: %[[WEIGHT_ZP:.+]]  = "tosa.const"() <{value = dense<42> : tensor<1xi8>}
  // CHECK-DAG: %[[REV0:.+]] = tosa.reverse %2 {axis = 2 : i32}
  // CHECK-DAG: %[[REV1:.+]] = tosa.reverse %arg1 {axis = 1 : i32}
  // CHECK: tosa.conv2d %arg0, %3, %arg2, %[[INPUT_ZP]], %[[WEIGHT_ZP]]
  // CHECK-SAME: dilation = array<i64: 1, 1>, pad = array<i64: 3, 4, 8, 9>,
  // CHECK-SAME: stride = array<i64: 1, 1>}
  %input_zp = "tosa.const"() {value = dense<-22> : tensor<1xi8>} : () -> tensor<1xi8>
  %weight_zp = "tosa.const"() {value = dense<42> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {
    acc_type = i32,
    out_pad = array<i64: 1, 2, 3, 4>,
    out_shape = array<i64: -1, -1, -1, -1>,
    stride = array<i64: 1, 1>} : (tensor<2x16x14x3xi8>, tensor<5x3x6x3xi8>, tensor<5xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<2x21x26x5xi32>
  return %0 : tensor<2x21x26x5xi32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_strided

func.func @transpose_conv2d_strided(%arg0: tensor<2x17x15x3xf32>, %arg1: tensor<5x3x5x3xf32>, %arg2: tensor<5xf32>) -> tensor<2x?x?x5xf32> {
  // Manipulate the weight matrix to handle striding.
  // CHECK-DAG: %[[PADV:.+]] = tosa.const_shape {value = dense<[0, 0, 0, 1, 0, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  // CHECK-DAG: %[[TRANSV:.+]]  = "tosa.const"() <{value = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[PADW:.+]]  = tosa.pad %arg1, %[[PADV]]
  // CHECK-DAG: %[[CONST1:.+]] = tosa.const_shape {value = dense<[5, 2, 2, 2, 3, 3]> : tensor<6xindex>}
  // CHECK-DAG: %[[RESW1:.+]]  = tosa.reshape %[[PADW]], %[[CONST1]]
  // CHECK-DAG: %[[TRANS:.+]]  = tosa.transpose %[[RESW1]], %[[TRANSV]]
  // CHECK-DAG: %[[CONST3:.+]] = tosa.const_shape {value = dense<[30, 2, 2, 3]> : tensor<4xindex>}
  // CHECK-DAG: %[[RESW2:.+]]  = tosa.reshape %[[TRANS]], %[[CONST3]]
  // CHECK-DAG: %[[REV1:.+]]  = tosa.reverse %[[RESW2]] {axis = 1 : i32}
  // CHECK-DAG: %[[NEWWEIGHT:.+]] = tosa.reverse %[[REV1]] {axis = 2 : i32}
  // CHECK-DAG: %[[SIZE:.*]] = tosa.const_shape  {value = dense<[2, 35, 47, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK-DAG: %[[START:.*]] = tosa.const_shape  {value = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>

  // Pad out the input matrix to handle the transpose conv.
  // CHECK-DAG: %[[PAD:.+]] = tosa.const_shape {value = dense<[0, 0, 1, 1, 1, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  // CHECK-DAG: %[[TRANS2:.+]]  = "tosa.const"() <{value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[NEWINPUT:.+]] = tosa.pad %arg0, %[[PAD]]

  // Manipulate the final shape.
  // CHECK-DAG: %[[BIAS:.+]]  = "tosa.const"() <{value = dense<0.000000e+00> : tensor<30xf32>}
  // CHECK-DAG: %[[CONV:.+]] = tosa.conv2d %[[NEWINPUT]], %[[NEWWEIGHT]], %[[BIAS]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
  // CHECK-DAG: %[[CONST6:.+]] = tosa.const_shape {value = dense<[2, 18, 16, 2, 3, 5]> : tensor<6xindex>}
  // CHECK-DAG: %[[RESHAPE_OUT_1:.+]] = tosa.reshape %[[CONV]], %[[CONST6]]
  // CHECK-DAG: %[[TRANS_OUT:.+]] = tosa.transpose %[[RESHAPE_OUT_1]], %[[TRANS2]]
  // CHECK-DAG: %[[CONST8:.+]] = tosa.const_shape {value = dense<[2, 36, 48, 5]> : tensor<4xindex>
  // CHECK-DAG: %[[RESHAPE_OUT_2:.+]] = tosa.reshape %[[TRANS_OUT]], %[[CONST8]]
  // CHECK-DAG: %[[SLICE:.+]] = tosa.slice %[[RESHAPE_OUT_2]], %[[START]], %[[SIZE]]
  // CHECK-DAG: %[[CONST9:.+]] = tosa.const_shape {value = dense<[1, 1, 1, 5]> : tensor<4xindex>}
  // CHECK-DAG: %[[RESHAPE_ARG2:.+]] = tosa.reshape %arg2, %[[CONST9]]
  // CHECK: %[[ADD:.+]] = tosa.add %[[SLICE]], %[[RESHAPE_ARG2]]
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2{acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 2, 3>} : (tensor<2x17x15x3xf32>, tensor<5x3x5x3xf32>, tensor<5xf32>) -> tensor<2x35x47x5xf32>
  %1 = tensor.cast %0 : tensor<2x35x47x5xf32> to tensor<2x?x?x5xf32>
  return %1 : tensor<2x?x?x5xf32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_strided_quantized

func.func @transpose_conv2d_strided_quantized(%arg0: tensor<2x17x15x3xi8>, %arg1: tensor<5x3x5x3xi8>, %arg2: tensor<5xi32>) -> (tensor<2x35x47x5xi32>) {
  // Manipulate the weight matrix to handle striding.
  // CHECK-DAG: %[[PADV:.+]]  = tosa.const_shape {value = dense<[0, 0, 0, 1, 0, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  // CHECK-DAG: %[[TRANSV:.+]]  = "tosa.const"() <{value = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[PADW:.+]]  = tosa.pad %arg1, %[[PADV]] {quantization_info = #tosa.pad_quant<input_zp = 42>}
  // CHECK-DAG: %[[CONST1:.+]] = tosa.const_shape {value = dense<[5, 2, 2, 2, 3, 3]> : tensor<6xindex>}
  // CHECK-DAG: %[[RESW1:.+]]  = tosa.reshape %[[PADW]], %[[CONST1]]
  // CHECK-DAG: %[[TRANS:.+]]  = tosa.transpose %[[RESW1]], %[[TRANSV]]
  // CHECK-DAG: %[[CONST3:.+]] = tosa.const_shape {value = dense<[30, 2, 2, 3]> : tensor<4xindex>}
  // CHECK-DAG: %[[RESW2:.+]]  = tosa.reshape %[[TRANS]], %[[CONST3]]
  // CHECK-DAG: %[[REV1:.+]]  = tosa.reverse %[[RESW2]] {axis = 1 : i32}
  // CHECK-DAG: %[[NEWWEIGHT:.+]] = tosa.reverse %[[REV1]] {axis = 2 : i32}
  // CHECK-DAG: %[[SIZE:.*]] = tosa.const_shape  {value = dense<[2, 35, 47, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK-DAG: %[[START:.*]] = tosa.const_shape  {value = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>

  // Pad out the input matrix to handle the transpose conv.
  // CHECK-DAG: %[[PAD:.+]]  = tosa.const_shape {value = dense<[0, 0, 1, 1, 1, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  // CHECK-DAG: %[[TRANS2:.+]]  = "tosa.const"() <{value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[NEWINPUT:.+]] = tosa.pad %arg0, %[[PAD]] {quantization_info = #tosa.pad_quant<input_zp = -22>}

  // Manipulate the final shape.
  // CHECK-DAG: %[[BIAS:.+]]  = "tosa.const"() <{value = dense<0> : tensor<30xi32>}
  // CHECK-DAG: %[[INPUT_ZP:.+]]  = "tosa.const"() <{value = dense<-22> : tensor<1xi8>}
  // CHECK-DAG: %[[WEIGHT_ZP:.+]]  = "tosa.const"() <{value = dense<42> : tensor<1xi8>}
  // CHECK-DAG: %[[CONV:.+]] = tosa.conv2d %[[NEWINPUT]], %[[NEWWEIGHT]], %[[BIAS]], %[[INPUT_ZP]], %[[WEIGHT_ZP]] {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
  // CHECK-DAG: %[[CONV_NEW_SHAPE:.*]] = tosa.const_shape  {value = dense<[2, 18, 16, 2, 3, 5]> : tensor<6xindex>}
  // CHECK-DAG: %[[RESHAPE_OUT_1:.+]] = tosa.reshape %[[CONV]], %[[CONV_NEW_SHAPE]]
  // CHECK-DAG: %[[TRANS_OUT:.+]] = tosa.transpose %[[RESHAPE_OUT_1]], %[[TRANS2]]
  // CHECK-DAG: %[[TEANS_NEW_SHAPE:.+]] = tosa.const_shape {value = dense<[2, 36, 48, 5]> : tensor<4xindex>}
  // CHECK-DAG: %[[RESHAPE_OUT_2:.+]] = tosa.reshape %[[TRANS_OUT]], %[[TEANS_NEW_SHAPE]]
  // CHECK-DAG: %[[SLICE:.+]] = tosa.slice %[[RESHAPE_OUT_2]], %[[START]], %[[SIZE]]
  // CHECK-DAG: %[[ARG2_NEW_SHAPE:.+]] = tosa.const_shape {value = dense<[1, 1, 1, 5]> : tensor<4xindex>}
  // CHECK-DAG: %[[RESHAPE_ARG2:.+]] = tosa.reshape %arg2, %[[ARG2_NEW_SHAPE]]
  // CHECK: %[[ADD:.+]] = tosa.add %[[SLICE]], %[[RESHAPE_ARG2]]
  %input_zp = "tosa.const"() {value = dense<-22> : tensor<1xi8>} : () -> tensor<1xi8>
  %weight_zp = "tosa.const"() {value = dense<42> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = i32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 2, 3>} : (tensor<2x17x15x3xi8>, tensor<5x3x5x3xi8>, tensor<5xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<2x35x47x5xi32>
  return %0 : tensor<2x35x47x5xi32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_strided_overpad
func.func @transpose_conv2d_strided_overpad(%arg0 : tensor<1x16x1x1xi8>, %arg1 : tensor<1x2x1x1xi8>, %arg2 : tensor<1xi32>) -> (tensor<1x19x2x1xi32>) {
  // CHECK-DAG: %[[WEIGHT_PAD:.+]] = tosa.const_shape  {value = dense<[0, 0, 0, 0, 0, 1, 0, 0]> : tensor<8xindex>}
  // CHECK-DAG: %[[CONST1:.+]] = tosa.const_shape {value = dense<[1, 2, 1, 1, 2, 1]> : tensor<6xindex>}
  // CHECK-DAG: %[[WEIGHT_PERMS:.+]] = "tosa.const"() <{value = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[CONST3:.+]] = tosa.const_shape {value = dense<[2, 2, 1, 1]> : tensor<4xindex>}
  // CHECK-DAG: %[[INPUT_PAD:.+]] = tosa.const_shape  {value = dense<[0, 0, 1, 1, 0, 0, 0, 0]> : tensor<8xindex>}
  // CHECK-DAG: %[[ZERO:.+]] = "tosa.const"() <{value = dense<0> : tensor<2xi32>}
  // CHECK-DAG: %[[CONST6:.+]] = tosa.const_shape {value = dense<[1, 17, 1, 1, 2, 1]> : tensor<6xindex>}
  // CHECK-DAG: %[[RESULT_PERMS:.+]] = "tosa.const"() <{value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
  // CHECK-DAG: %[[CONST8:.+]] = tosa.const_shape {value = dense<[1, 17, 2, 1]> : tensor<4xindex>}
  // CHECK-DAG: %[[RESULT_PAD:.+]] = tosa.const_shape  {value = dense<[0, 0, 2, 0, 0, 0, 0, 0]> : tensor<8xindex>}
  // CHECK-DAG: %[[CONST10:.+]] = tosa.const_shape {value = dense<1> : tensor<4xindex>}
  // CHECK-DAG: %[[INPUT_ZP:.*]] = "tosa.const"() <{value = dense<-103> : tensor<1xi8>}>
  // CHECK-DAG: %[[WEIGHT_ZP:.*]] = "tosa.const"() <{value = dense<93> : tensor<1xi8>}>
  // CHECK: %[[PAD_WEIGHT:.+]] = tosa.pad %arg1, %[[WEIGHT_PAD]] {quantization_info = #tosa.pad_quant<input_zp = 93>}
  // CHECK: %[[RESHAPE_WEIGHT_0:.+]] = tosa.reshape %[[PAD_WEIGHT]], %[[CONST1]]
  // CHECK: %[[TRANSPOSE_WEIGHT:.+]] = tosa.transpose %[[RESHAPE_WEIGHT_0]], %[[WEIGHT_PERMS]]
  // CHECK: %[[RESHAPE_WEIGHT_1:.+]] = tosa.reshape %[[TRANSPOSE_WEIGHT]], %[[CONST3]]
  // CHECK: %[[REVERSE:.+]] = tosa.reverse %[[RESHAPE_WEIGHT_1]] {axis = 1 : i32}
  // CHECK: %[[PAD_INPUT:.+]] = tosa.pad %arg0, %[[INPUT_PAD]] {quantization_info = #tosa.pad_quant<input_zp = -103>}
  // CHECK: %[[CONV:.+]] = tosa.conv2d %[[PAD_INPUT]], %[[REVERSE]], %[[ZERO]], %[[INPUT_ZP]], %[[WEIGHT_ZP]] {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
  // CHECK: %[[RESHAPE_RESULT_0:.+]] = tosa.reshape %[[CONV]], %[[CONST6]]
  // CHECK: %[[TRANSPOSE_RESULT:.+]] = tosa.transpose %[[RESHAPE_RESULT_0]], %[[RESULT_PERMS]]
  // CHECK: %[[RESHAPE_RESULT_1:.+]] = tosa.reshape %[[TRANSPOSE_RESULT]], %[[CONST8]]
  // CHECK: %[[PAD_RESULT:.+]] = tosa.pad %[[RESHAPE_RESULT_1]], %[[RESULT_PAD]]
  // CHECK: %[[RESHAPE_ARG2:.+]] = tosa.reshape %arg2, %[[CONST10]]
  // CHECK: %[[ADD:.+]] = tosa.add %[[PAD_RESULT]], %[[RESHAPE_ARG2]]
  %input_zp = "tosa.const"() {value = dense<-103> : tensor<1xi8>} : () -> tensor<1xi8>
  %weight_zp = "tosa.const"() {value = dense<93> : tensor<1xi8>} : () -> tensor<1xi8>
  %2 =  tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {
    acc_type = i32,
    out_pad = array<i64: 2, 0, 0, 1>,
    out_shape = array<i64: 1, -1, -1, 1>,
    stride = array<i64: 1, 2>} :
    (tensor<1x16x1x1xi8>, tensor<1x2x1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x19x2x1xi32>
  "func.return" (%2) : (tensor<1x19x2x1xi32>) -> ()
}
