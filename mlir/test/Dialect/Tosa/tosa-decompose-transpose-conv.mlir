// RUN: mlir-opt --split-input-file --tosa-optional-decompositions %s | FileCheck %s

// CHECK-LABEL: @transpose_conv2d
func.func @transpose_conv2d(%arg0: tensor<2x16x14x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) -> tensor<2x18x19x5xf32> {
  // CHECK-DAG: %[[ZP:.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  // CHECK: %[[REV1:.+]] = tosa.reverse %arg1 {axis = 1 : i32}
  // CHECK: %[[REV2:.+]] = tosa.reverse %[[REV1]] {axis = 2 : i32}
  // CHECK: tosa.conv2d %arg0, %[[REV2]], %arg2, %[[ZP]], %[[ZP]]
  // CHECK-SAME: dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 5, 5>, stride = array<i64: 1, 1>
  %zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %zp, %zp {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x16x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x18x19x5xf32>
  return %0 : tensor<2x18x19x5xf32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_quantized

func.func @transpose_conv2d_quantized(%arg0: tensor<2x16x14x3xi8>, %arg1: tensor<5x3x6x3xi8>, %arg2: tensor<5xi32>) -> (tensor<2x18x19x5xi32>) {
  // CHECK-DAG: %[[INPUT_ZP:.+]]  = "tosa.const"() <{values = dense<-6> : tensor<1xi8>}
  // CHECK-DAG: %[[WEIGHT_ZP:.+]]  = "tosa.const"() <{values = dense<11> : tensor<1xi8>}
  // CHECK: %[[REV1:.+]] = tosa.reverse %arg1 {axis = 1 : i32}
  // CHECK: %[[REV2:.+]] = tosa.reverse %[[REV1]] {axis = 2 : i32}
  // CHECK: tosa.conv2d %arg0, %[[REV2]], %arg2, %[[INPUT_ZP]], %[[WEIGHT_ZP]] {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 5, 5>, stride = array<i64: 1, 1>}
  %input_zp = "tosa.const"() {values = dense<-6> : tensor<1xi8>} : () -> tensor<1xi8>
  %weight_zp = "tosa.const"() {values = dense<11> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = i32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x16x14x3xi8>, tensor<5x3x6x3xi8>, tensor<5xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<2x18x19x5xi32>
  return %0 : tensor<2x18x19x5xi32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_quantized_padded
func.func @transpose_conv2d_quantized_padded(%arg0: tensor<2x16x14x3xi8>, %arg1: tensor<5x3x6x3xi8>, %arg2: tensor<5xi32>) -> (tensor<2x21x26x5xi32>) {
  // CHECK-DAG: %[[INPUT_ZP:.+]] = "tosa.const"() <{values = dense<-22> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[WEIGHT_ZP:.+]] = "tosa.const"() <{values = dense<42> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[REV0:.+]] = tosa.reverse %arg1 {axis = 1 : i32}
  // CHECK-DAG: %[[REV1:.+]] = tosa.reverse %[[REV0]] {axis = 2 : i32}
  // CHECK: tosa.conv2d %arg0, %[[REV1]], %arg2, %[[INPUT_ZP]], %[[WEIGHT_ZP]] {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 3, 4, 8, 9>, stride = array<i64: 1, 1>}
  %input_zp = "tosa.const"() <{values = dense<-22> : tensor<1xi8>}> : () -> tensor<1xi8>
  %weight_zp = "tosa.const"() <{values = dense<42> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {
    acc_type = i32,
    out_pad = array<i64: 1, 2, 3, 4>,
    stride = array<i64: 1, 1>} : (tensor<2x16x14x3xi8>, tensor<5x3x6x3xi8>, tensor<5xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<2x21x26x5xi32>
  return %0 : tensor<2x21x26x5xi32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_strided

func.func @transpose_conv2d_strided(%arg0: tensor<2x17x15x3xf32>, %arg1: tensor<5x3x5x3xf32>, %arg2: tensor<5xf32>) -> tensor<2x?x?x5xf32> {
  // Manipulate the weight matrix to handle striding.
  // CHECK-DAG: %[[PADV:.+]] = tosa.const_shape {values = dense<[0, 0, 0, 1, 0, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  // CHECK-DAG: %[[PADW:.+]]  = tosa.pad %arg1, %[[PADV]]
  // CHECK-DAG: %[[CONST1:.+]] = tosa.const_shape {values = dense<[5, 2, 2, 2, 3, 3]> : tensor<6xindex>}
  // CHECK-DAG: %[[RESW1:.+]]  = tosa.reshape %[[PADW]], %[[CONST1]]
  // CHECK-DAG: %[[TRANS:.+]]  = tosa.transpose %[[RESW1]] {perms = array<i32: 2, 4, 0, 1, 3, 5>}
  // CHECK-DAG: %[[CONST3:.+]] = tosa.const_shape {values = dense<[30, 2, 2, 3]> : tensor<4xindex>}
  // CHECK-DAG: %[[RESW2:.+]]  = tosa.reshape %[[TRANS]], %[[CONST3]]
  // CHECK-DAG: %[[REV1:.+]]  = tosa.reverse %[[RESW2]] {axis = 1 : i32}
  // CHECK-DAG: %[[NEWWEIGHT:.+]] = tosa.reverse %[[REV1]] {axis = 2 : i32}
  // CHECK-DAG: %[[SIZE:.*]] = tosa.const_shape  {values = dense<[2, 35, 47, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK-DAG: %[[START:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>

  // Pad out the input matrix to handle the transpose conv.
  // CHECK-DAG: %[[PAD:.+]] = tosa.const_shape {values = dense<[0, 0, 1, 1, 1, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  // CHECK-DAG: %[[NEWINPUT:.+]] = tosa.pad %arg0, %[[PAD]]

  // Manipulate the final shape.
  // CHECK-DAG: %[[ZP:.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  // CHECK-DAG: %[[BIAS:.+]]  = "tosa.const"() <{values = dense<0.000000e+00> : tensor<30xf32>}
  // CHECK-DAG: %[[CONV:.+]] = tosa.conv2d %[[NEWINPUT]], %[[NEWWEIGHT]], %[[BIAS]], %[[ZP]], %[[ZP]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
  // CHECK-DAG: %[[CONST6:.+]] = tosa.const_shape {values = dense<[2, 18, 16, 2, 3, 5]> : tensor<6xindex>}
  // CHECK-DAG: %[[RESHAPE_OUT_1:.+]] = tosa.reshape %[[CONV]], %[[CONST6]]
  // CHECK-DAG: %[[TRANS_OUT:.+]] = tosa.transpose %[[RESHAPE_OUT_1]] {perms = array<i32: 0, 1, 3, 2, 4, 5>}
  // CHECK-DAG: %[[CONST8:.+]] = tosa.const_shape {values = dense<[2, 36, 48, 5]> : tensor<4xindex>
  // CHECK-DAG: %[[RESHAPE_OUT_2:.+]] = tosa.reshape %[[TRANS_OUT]], %[[CONST8]]
  // CHECK-DAG: %[[SLICE:.+]] = tosa.slice %[[RESHAPE_OUT_2]], %[[START]], %[[SIZE]]
  // CHECK-DAG: %[[CONST9:.+]] = tosa.const_shape {values = dense<[1, 1, 1, 5]> : tensor<4xindex>}
  // CHECK-DAG: %[[RESHAPE_ARG2:.+]] = tosa.reshape %arg2, %[[CONST9]]
  // CHECK: %[[ADD:.+]] = tosa.add %[[SLICE]], %[[RESHAPE_ARG2]]
  %zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %zp, %zp {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 3>} : (tensor<2x17x15x3xf32>, tensor<5x3x5x3xf32>, tensor<5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x35x47x5xf32>
  %1 = tensor.cast %0 : tensor<2x35x47x5xf32> to tensor<2x?x?x5xf32>
  return %1 : tensor<2x?x?x5xf32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_strided_quantized

func.func @transpose_conv2d_strided_quantized(%arg0: tensor<2x17x15x3xi8>, %arg1: tensor<5x3x5x3xi8>, %arg2: tensor<5xi32>) -> (tensor<2x35x47x5xi32>) {
  // Manipulate the weight matrix to handle striding.
  // CHECK-DAG: %[[INPUT_ZP:.+]] = "tosa.const"() <{values = dense<-22> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[WEIGHT_ZP:.+]] = "tosa.const"() <{values = dense<42> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[PADV:.+]]  = tosa.const_shape {values = dense<[0, 0, 0, 1, 0, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  // CHECK-DAG: %[[PADW:.+]]  = tosa.pad %arg1, %[[PADV]], %[[WEIGHT_ZP]]
  // CHECK-DAG: %[[CONST1:.+]] = tosa.const_shape {values = dense<[5, 2, 2, 2, 3, 3]> : tensor<6xindex>}
  // CHECK-DAG: %[[RESW1:.+]]  = tosa.reshape %[[PADW]], %[[CONST1]]
  // CHECK-DAG: %[[TRANS:.+]]  = tosa.transpose %[[RESW1]] {perms = array<i32: 2, 4, 0, 1, 3, 5>}
  // CHECK-DAG: %[[CONST3:.+]] = tosa.const_shape {values = dense<[30, 2, 2, 3]> : tensor<4xindex>}
  // CHECK-DAG: %[[RESW2:.+]]  = tosa.reshape %[[TRANS]], %[[CONST3]]
  // CHECK-DAG: %[[REV1:.+]]  = tosa.reverse %[[RESW2]] {axis = 1 : i32}
  // CHECK-DAG: %[[NEWWEIGHT:.+]] = tosa.reverse %[[REV1]] {axis = 2 : i32}

  // Pad out the input matrix to handle the transpose conv.
  // CHECK-DAG: %[[PAD:.+]]  = tosa.const_shape {values = dense<[0, 0, 1, 1, 1, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  // CHECK-DAG: %[[NEWINPUT:.+]] = tosa.pad %arg0, %[[PAD]], %[[INPUT_ZP]]

  // Manipulate the final shape.
  // CHECK-DAG: %[[BIAS:.+]]  = "tosa.const"() <{values = dense<0> : tensor<30xi32>}
  // CHECK-DAG: %[[CONV:.+]] = tosa.conv2d %[[NEWINPUT]], %[[NEWWEIGHT]], %[[BIAS]], %[[INPUT_ZP]], %[[WEIGHT_ZP]] {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
  // CHECK-DAG: %[[CONST6:.+]] = tosa.const_shape {values = dense<[2, 18, 16, 2, 3, 5]> : tensor<6xindex>}
  // CHECK-DAG: %[[RESHAPE_OUT_1:.+]] = tosa.reshape %[[CONV]], %[[CONST6]]
  // CHECK-DAG: %[[TRANS_OUT:.+]] = tosa.transpose %[[RESHAPE_OUT_1]] {perms = array<i32: 0, 1, 3, 2, 4, 5>}
  // CHECK-DAG: %[[CONST8:.+]] = tosa.const_shape {values = dense<[2, 36, 48, 5]> : tensor<4xindex>}
  // CHECK-DAG: %[[RESHAPE_OUT_2:.+]] = tosa.reshape %[[TRANS_OUT]], %[[CONST8]]
  // CHECK-DAG: %[[START:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>}
  // CHECK-DAG: %[[SIZE:.*]] = tosa.const_shape  {values = dense<[2, 35, 47, 5]> : tensor<4xindex>}
  // CHECK-DAG: %[[SLICE:.*]] = tosa.slice %[[RESHAPE_OUT_2]], %[[START]], %[[SIZE]]
  // CHECK-DAG: %[[CONST9:.+]] = tosa.const_shape {values = dense<[1, 1, 1, 5]> : tensor<4xindex>}
  // CHECK-DAG: %[[RESHAPE_ARG2:.+]] = tosa.reshape %arg2, %[[CONST9]]
  // CHECK: %[[ADD:.+]] = tosa.add %[[SLICE]], %[[RESHAPE_ARG2]]
  %input_zp = "tosa.const"() <{values = dense<-22> : tensor<1xi8>}> : () -> tensor<1xi8>
  %weight_zp = "tosa.const"() <{values = dense<42> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = i32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 3>} : (tensor<2x17x15x3xi8>, tensor<5x3x5x3xi8>, tensor<5xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<2x35x47x5xi32>
  return %0 : tensor<2x35x47x5xi32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_strided_quantized_quant_input
func.func @transpose_conv2d_strided_quantized_quant_input(%arg0: tensor<2x17x15x3x!quant.uniform<i8:f32, 0.015684274956583977:-1>>, %arg1: tensor<5x3x5x3x!quant.uniform<i8:f32, 0.015684274956583977:-1>>, %arg2: tensor<5xi32>) -> (tensor<2x35x47x5xi32>) {
  // Checks a regression. A typo in `createPadConstTensor` caused the conversion to crash
  // CHECK-DAG: %[[PAD_SHAPE:.+]] = tosa.const_shape {values = dense<[0, 0, 1, 1, 1, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  // CHECK-DAG: %[[PAD_CONST:.+]] = "tosa.const"() <{values = dense<-22> : tensor<1xi8>}> : () -> tensor<1x!quant.uniform<i8:f32, 0.015684274956583977:-1>>
  // CHECK: %[[PAD:.+]] = tosa.pad %arg0, %[[PAD_SHAPE]], %[[PAD_CONST]] : (tensor<2x17x15x3x!quant.uniform<i8:f32, 0.015684274956583977:-1>>, !tosa.shape<8>, tensor<1x!quant.uniform<i8:f32, 0.015684274956583977:-1>>)
  %input_zp = "tosa.const"() <{values = dense<-22> : tensor<1xi8>}> : () -> tensor<1xi8>
  %weight_zp = "tosa.const"() <{values = dense<42> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = i32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 3>} : (tensor<2x17x15x3x!quant.uniform<i8:f32, 0.015684274956583977:-1>>, tensor<5x3x5x3x!quant.uniform<i8:f32, 0.015684274956583977:-1>>, tensor<5xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<2x35x47x5xi32>
  return %0 : tensor<2x35x47x5xi32>
}

// -----

// CHECK-LABEL: @transpose_conv2d_strided_overpad
func.func @transpose_conv2d_strided_overpad(%arg0 : tensor<1x16x1x1xi8>, %arg1 : tensor<1x2x1x1xi8>, %arg2 : tensor<1xi32>) -> (tensor<1x19x2x1xi32>) {
  // CHECK-DAG: %[[WEIGHT_PAD:.+]] = tosa.const_shape {values = dense<[0, 0, 0, 0, 0, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  // CHECK-DAG: %[[CONST1:.+]] = tosa.const_shape {values = dense<[1, 2, 1, 1, 2, 1]> : tensor<6xindex>}
  // CHECK-DAG: %[[INPUT_ZP:.+]] = "tosa.const"() <{values = dense<-103> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[WEIGHT_ZP:.+]] = "tosa.const"() <{values = dense<93> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[CONST3:.+]] = tosa.const_shape {values = dense<[2, 2, 1, 1]> : tensor<4xindex>}
  // CHECK-DAG: %[[INPUT_PAD:.+]] = tosa.const_shape {values = dense<[0, 0, 1, 1, 0, 0, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  // CHECK-DAG: %[[ZERO:.+]] = "tosa.const"() <{values = dense<0> : tensor<2xi32>}
  // CHECK-DAG: %[[CONST6:.+]] = tosa.const_shape {values = dense<[1, 17, 1, 1, 2, 1]> : tensor<6xindex>}
  // CHECK-DAG: %[[CONST8:.+]] = tosa.const_shape {values = dense<[1, 17, 2, 1]> : tensor<4xindex>}
  // CHECK-DAG: %[[RESULT_PAD:.+]] = tosa.const_shape {values = dense<[0, 0, 2, 0, 0, 0, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  // CHECK-DAG: %[[CONST10:.+]] = tosa.const_shape {values = dense<1> : tensor<4xindex>}
  // CHECK: %[[PAD_WEIGHT:.+]] = tosa.pad %arg1, %[[WEIGHT_PAD]], %[[WEIGHT_ZP]]
  // CHECK: %[[RESHAPE_WEIGHT_0:.+]] = tosa.reshape %[[PAD_WEIGHT]], %[[CONST1]]
  // CHECK: %[[TRANSPOSE_WEIGHT:.+]] = tosa.transpose %[[RESHAPE_WEIGHT_0]] {perms = array<i32: 2, 4, 0, 1, 3, 5>}
  // CHECK: %[[RESHAPE_WEIGHT_1:.+]] = tosa.reshape %[[TRANSPOSE_WEIGHT]], %[[CONST3]]
  // CHECK: %[[REVERSE:.+]] = tosa.reverse %[[RESHAPE_WEIGHT_1]] {axis = 1 : i32}
  // CHECK: %[[PAD_INPUT:.+]] = tosa.pad %arg0, %[[INPUT_PAD]], %[[INPUT_ZP]]
  // CHECK: %[[CONV:.+]] = tosa.conv2d %[[PAD_INPUT]], %[[REVERSE]], %[[ZERO]], %[[INPUT_ZP]], %[[WEIGHT_ZP]]
  // CHECK-SAME{literal}: dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]}
  // CHECK: %[[RESHAPE_RESULT_0:.+]] = tosa.reshape %[[CONV]], %[[CONST6]]
  // CHECK: %[[TRANSPOSE_RESULT:.+]] = tosa.transpose %[[RESHAPE_RESULT_0]] {perms = array<i32: 0, 1, 3, 2, 4, 5>}
  // CHECK: %[[RESHAPE_RESULT_1:.+]] = tosa.reshape %[[TRANSPOSE_RESULT]], %[[CONST8]]
  // CHECK: %[[PAD_RESULT:.+]] = tosa.pad %[[RESHAPE_RESULT_1]], %[[RESULT_PAD]]
  // CHECK: %[[RESHAPE_ARG2:.+]] = tosa.reshape %arg2, %[[CONST10]]
  // CHECK: %[[ADD:.+]] = tosa.add %[[PAD_RESULT]], %[[RESHAPE_ARG2]]
  %input_zp = "tosa.const"() <{values = dense<-103> : tensor<1xi8>}> : () -> tensor<1xi8>
  %weight_zp = "tosa.const"() <{values = dense<93> : tensor<1xi8>}> : () -> tensor<1xi8>
  %2 =  tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {
    acc_type = i32,
    out_pad = array<i64: 2, 0, 0, 1>,
    stride = array<i64: 1, 2>} :
    (tensor<1x16x1x1xi8>, tensor<1x2x1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x19x2x1xi32>
  "func.return" (%2) : (tensor<1x19x2x1xi32>) -> ()
}


// -----
// CHECK-LABEL: @transpose_conv2d_non_strided_dynamic_batch
// CHECK: tosa.conv2d
// CHECK-NOT: tosa.transpose_conv2d
func.func @transpose_conv2d_non_strided_dynamic_batch(%arg0: tensor<?x16x14x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) -> tensor<?x18x19x5xf32> {
  %zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %zp, %zp {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x16x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x18x19x5xf32>
  return %0 : tensor<?x18x19x5xf32>
}

// -----
// CHECK-LABEL: @transpose_conv2d_strided_dynamic_batch
// CHECK: tosa.conv2d
// CHECK-NOT: tosa.transpose_conv2d
func.func @transpose_conv2d_strided_dynamic_batch(%arg0: tensor<?x17x15x3xf32>, %arg1: tensor<5x3x5x3xf32>, %arg2: tensor<5xf32>) -> tensor<?x35x47x5xf32> {
  %zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %zp, %zp {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 3>} : (tensor<?x17x15x3xf32>, tensor<5x3x5x3xf32>, tensor<5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x35x47x5xf32>
  return %0 : tensor<?x35x47x5xf32>
}
