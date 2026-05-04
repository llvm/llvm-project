// RUN: mlir-opt --split-input-file --tosa-optional-decompositions %s | FileCheck %s

// -----

// CHECK-LABEL: @depthwise_conv2d_as_mul
func.func @depthwise_conv2d_as_mul(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<1x1x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<4x10x10x6xf32> {
  // CHECK-NOT: tosa.depthwise_conv2d
  // CHECK-DAG: %[[CONST0:.+]] = tosa.const_shape {values = dense<[4, 10, 10, 2, 1]> : tensor<5xindex>
  // CHECK-DAG: %[[CONST1:.+]] = tosa.const_shape {values = dense<[1, 1, 1, 2, 3]> : tensor<5xindex>
  // CHECK-DAG: %[[CONST2:.+]] = tosa.const_shape {values = dense<[4, 10, 10, 6]> : tensor<4xindex>
  // CHECK-DAG: %[[CONST3:.+]] = tosa.const_shape {values = dense<[1, 1, 1, 6]> : tensor<4xindex>
  // CHECK-DAG: %[[VAR0:.*]] = tosa.reshape %arg0, %[[CONST0]]
  // CHECK-SAME: -> tensor<4x10x10x2x1xf32>
  // CHECK: %[[VAR1:.*]] = tosa.reshape %arg1, %[[CONST1]]
  // CHECK-SAME: -> tensor<1x1x1x2x3xf32>
  // CHECK: %[[VAR2:.*]] = tosa.mul %[[VAR0]], %[[VAR1]]
  // CHECK-SAME: -> tensor<4x10x10x2x3xf32>
  // CHECK: %[[VAR3:.*]] = tosa.reshape %[[VAR2]], %[[CONST2]]
  // CHECK-SAME: -> tensor<4x10x10x6xf32>
  // CHECK: %[[VAR4:.*]] = tosa.reshape %arg2, %[[CONST3]]
  // CHECK-SAME: -> tensor<1x1x1x6xf32>
  // CHECK: %[[VAR5:.*]] = tosa.add %[[VAR3]], %[[VAR4]]
  // CHECK-SAME: -> tensor<4x10x10x6xf32>
  // CHECK: return %[[VAR5]]
  %zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %zp, %zp {acc_type = f32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<4x10x10x2xf32>, tensor<1x1x2x3xf32>, tensor<6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4x10x10x6xf32>
  return %0 : tensor<4x10x10x6xf32>
}

// -----

// CHECK-LABEL: @depthwise_conv2d_as_mul_q
func.func @depthwise_conv2d_as_mul_q(%arg0: tensor<4x10x10x2xi8>, %arg1: tensor<1x1x2x3xi8>, %arg2: tensor<6xi32>) -> tensor<4x10x10x6xi32> {
  // CHECK-DAG: %[[CONST0:.+]] = tosa.const_shape {values = dense<[4, 10, 10, 2, 1]> : tensor<5xindex>
  // CHECK-DAG: %[[INPUT_ZP:.+]] = "tosa.const"() <{values = dense<7> : tensor<1x1x1x1x1xi32>}
  // CHECK-DAG: %[[WEIGHT_ZP:.+]] = "tosa.const"() <{values = dense<11> : tensor<1x1x1x1xi32>}
  // CHECK-DAG: %[[CONST3:.+]] = tosa.const_shape {values = dense<[1, 1, 1, 2, 3]> : tensor<5xindex>
  // CHECK-DAG: %[[CONST4:.+]] = tosa.const_shape {values = dense<[4, 10, 10, 6]> : tensor<4xindex>
  // CHECK-DAG: %[[CONST5:.+]] = tosa.const_shape {values = dense<[1, 1, 1, 6]> : tensor<4xindex>
  // CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: %[[RESHAPE_I:.+]] = tosa.reshape %arg0, %[[CONST0]]
  // CHECK: %[[CAST_I:.+]] = tosa.cast %[[RESHAPE_I]] : (tensor<4x10x10x2x1xi8>) -> tensor<4x10x10x2x1xi32>
  // CHECK: %[[CAST_W:.+]] = tosa.cast %arg1 : (tensor<1x1x2x3xi8>) -> tensor<1x1x2x3xi32>
  // CHECK: %[[SUB_I:.+]] = tosa.sub %[[CAST_I]], %[[INPUT_ZP]]
  // CHECK: %[[SUB_W:.+]] = tosa.sub %[[CAST_W]], %[[WEIGHT_ZP]]
  // CHECK: %[[RESHAPE_W:.+]] = tosa.reshape %[[SUB_W]], %[[CONST3]]
  // CHECK: %[[MUL:.+]] = tosa.mul %[[SUB_I]], %[[RESHAPE_W]], %[[SHIFT]]
  // CHECK: %[[RESHAPE_O:.+]] = tosa.reshape %[[MUL]], %[[CONST4]]
  // CHECK: %[[RESHAPE_ARG2:.+]] = tosa.reshape %arg2, %[[CONST5]]
  // CHECK: %[[ADD:.+]] = tosa.add %[[RESHAPE_O]], %[[RESHAPE_ARG2]]
  %input_zp = "tosa.const"() {values = dense<7> : tensor<1xi8>} : () -> tensor<1xi8>
  %weight_zp = "tosa.const"() {values = dense<11> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = i32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<4x10x10x2xi8>, tensor<1x1x2x3xi8>, tensor<6xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<4x10x10x6xi32>
  return %0 : tensor<4x10x10x6xi32>
}

// -----

// CHECK-LABEL: @depthwise_conv2d_as_mul_padded
func.func @depthwise_conv2d_as_mul_padded(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<1x1x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<4x12x12x6xf32> {
  // CHECK-DAG: %[[CONST0:.+]] = tosa.const_shape {values = dense<[4, 10, 10, 2, 1]> : tensor<5xindex>}
  // CHECK-DAG: %[[PAD:.+]] = tosa.const_shape  {values = dense<[0, 0, 1, 1, 1, 1, 0, 0, 0, 0]> : tensor<10xindex>} : () -> !tosa.shape<10>
  // CHECK-DAG: %[[ZERO:.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}
  // CHECK-DAG: %[[CONST3:.+]] = tosa.const_shape {values = dense<[1, 1, 1, 2, 3]> : tensor<5xindex>}
  // CHECK-DAG: %[[CONST4:.+]] = tosa.const_shape {values = dense<[4, 12, 12, 6]> : tensor<4xindex>}
  // CHECK-DAG: %[[CONST5:.+]] = tosa.const_shape {values = dense<[1, 1, 1, 6]> : tensor<4xindex>}
  // CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: %[[RESHAPE_I:.+]] = tosa.reshape %arg0, %[[CONST0]]
  // CHECK: %[[PAD_I:.+]] = tosa.pad %[[RESHAPE_I]], %[[PAD]], %[[ZERO]] : (tensor<4x10x10x2x1xf32>, !tosa.shape<10>, tensor<1xf32>) -> tensor<4x12x12x2x1xf32>
  // CHECK: %[[RESHAPE_ARG1:.+]] = tosa.reshape %arg1, %[[CONST3]]
  // CHECK: %[[MUL:.+]] = tosa.mul %[[PAD_I]], %[[RESHAPE_ARG1]], %[[SHIFT]]
  // CHECK: %[[RESHAPE_O:.+]] = tosa.reshape %[[MUL]], %[[CONST4]]
  // CHECK: %[[RESHAPE_ARG2:.+]] = tosa.reshape %arg2, %[[CONST5]]
  // CHECK: %[[ADD:.+]] = tosa.add %[[RESHAPE_O]], %[[RESHAPE_ARG2]]
  %zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %zp, %zp {acc_type = f32, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<4x10x10x2xf32>, tensor<1x1x2x3xf32>, tensor<6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4x12x12x6xf32>
  return %0 : tensor<4x12x12x6xf32>
}

// -----

// Decompose only support integer or float types.

// CHECK-LABEL: @depthwise_conv2d_quant_type
func.func @depthwise_conv2d_quant_type(%arg0: tensor<4x10x10x2x!quant.uniform<i8:f32, 0.015684768557548523>>, %arg1: tensor<1x1x2x3x!quant.uniform<i8<-127:127>:f32, 0.015680249780416489>>, %arg2: tensor<6xi32>) -> tensor<4x10x10x6x!quant.uniform<i32:f32, 0.078431375324726104>> {
  %0 = "tosa.const"() <{values = dense<7> : tensor<1xi8>}> : () -> tensor<1xi8>
  %1 = "tosa.const"() <{values = dense<11> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: tosa.depthwise_conv2d
  %2 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %0, %1 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<4x10x10x2x!quant.uniform<i8:f32, 0.015684768557548523>>, tensor<1x1x2x3x!quant.uniform<i8<-127:127>:f32, 0.015680249780416489>>, tensor<6xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<4x10x10x6x!quant.uniform<i32:f32, 0.078431375324726104>>
  return %2 : tensor<4x10x10x6x!quant.uniform<i32:f32, 0.078431375324726104>>
}

// -----

// CHECK-LABEL: @depthwise_conv2d_no_const_zero_point
func.func @depthwise_conv2d_no_const_zero_point(%arg0: tensor<4x10x10x2xi8>, %arg1: tensor<1x1x2x3xi8>, %arg2: tensor<6xi32>, %arg3: tensor<1xi8>, %arg4: tensor<1xi8>) -> tensor<4x10x10x6xi32> {
  // CHECK: tosa.depthwise_conv2d
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = i32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<4x10x10x2xi8>, tensor<1x1x2x3xi8>, tensor<6xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<4x10x10x6xi32>
  return %0 : tensor<4x10x10x6xi32>
}

// -----
// CHECK-LABEL:   func.func @depthwise_conv2d_as_mul_dynamic_batch_bias(
// CHECK-SAME:      %[[INP:.*]]: tensor<?x10x10x2xf32>,
// CHECK-SAME:      %[[WTS:.*]]: tensor<1x1x2x3xf32>,
// CHECK-SAME:      %[[BIAS:.*]]: tensor<?xf32>) -> tensor<?x10x10x6xf32> {
// CHECK:           %[[BIAS_EXPANDED_SHAPE:.*]] = tosa.const_shape  {values = dense<[1, 1, 1, -1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[RES_EXPANDED_SHAPE:.*]] = tosa.const_shape  {values = dense<[-1, 10, 10, 6]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[MUL_SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[WTS_EXPANDED_SHAPE:.*]] = tosa.const_shape  {values = dense<[1, 1, 1, 2, 3]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK:           %[[INP_EXPANDED_SHAPE:.*]] = tosa.const_shape  {values = dense<[-1, 10, 10, 2, 1]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK:           %[[INP_RESHAPED:.*]] = tosa.reshape %[[INP]], %[[INP_EXPANDED_SHAPE]] : (tensor<?x10x10x2xf32>, !tosa.shape<5>) -> tensor<?x10x10x2x1xf32>
// CHECK:           %[[WTS_RESHAPED:.*]] = tosa.reshape %[[WTS]], %[[WTS_EXPANDED_SHAPE]] : (tensor<1x1x2x3xf32>, !tosa.shape<5>) -> tensor<1x1x1x2x3xf32>
// CHECK:           %[[MUL:.*]] = tosa.mul %[[INP_RESHAPED]], %[[WTS_RESHAPED]], %[[MUL_SHIFT]] : (tensor<?x10x10x2x1xf32>, tensor<1x1x1x2x3xf32>, tensor<1xi8>) -> tensor<?x10x10x2x3xf32>
// CHECK:           %[[RES_RESHAPED:.*]] = tosa.reshape %[[MUL]], %[[RES_EXPANDED_SHAPE]] : (tensor<?x10x10x2x3xf32>, !tosa.shape<4>) -> tensor<?x10x10x6xf32>
// CHECK:           %[[BIAS_RESHAPED:.*]] = tosa.reshape %[[BIAS]], %[[BIAS_EXPANDED_SHAPE]] : (tensor<?xf32>, !tosa.shape<4>) -> tensor<1x1x1x?xf32>
// CHECK:           %[[RES:.*]] = tosa.add %[[RES_RESHAPED]], %[[BIAS_RESHAPED]] : (tensor<?x10x10x6xf32>, tensor<1x1x1x?xf32>) -> tensor<?x10x10x6xf32>
// CHECK:           return %[[RES]]
func.func @depthwise_conv2d_as_mul_dynamic_batch_bias(%arg0: tensor<?x10x10x2xf32>, %arg1: tensor<1x1x2x3xf32>, %arg2: tensor<?xf32>) -> tensor<?x10x10x6xf32> {
  %zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %zp, %zp {acc_type = f32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<?x10x10x2xf32>, tensor<1x1x2x3xf32>, tensor<?xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x10x10x6xf32>
  return %0 : tensor<?x10x10x6xf32>
}
