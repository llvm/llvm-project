// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-linalg))" %s -verify-diagnostics

// CHECK-LABEL: @avg_pool2d_with_unsupported_quant_type
func.func @avg_pool2d_with_unsupported_quant_type(%arg0: tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>, %arg1: tensor<1xi8>, %arg2: tensor<1xi8>) -> tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>> {
  // expected-error@+1 {{failed to legalize operation 'tosa.avg_pool2d'}}
  %0 = "tosa.avg_pool2d"(%arg0, %arg1, %arg2) {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>
  return %0 : tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>
}

// -----

// CHECK-LABEL: @tensor_with_unknown_rank
func.func @tensor_with_unknown_rank(%arg0: tensor<*xi8>) -> tensor<*xi8> {
  // expected-error@+1 {{failed to legalize operation 'tosa.abs'}}
  %0 = "tosa.abs"(%arg0) : (tensor<*xi8>) -> tensor<*xi8>
  return %0 : tensor<*xi8>
}

// -----

// CHECK-LABEL: @unranked_add
func.func @unranked_add(%arg0 : tensor<10x10xf32> , %arg1 : tensor<10x10xf32>, %arg2 : tensor<*xf32>) -> (tensor<10x10xf32>) {
  // expected-error@+3 {{failed to legalize operation 'tosa.add'}}
  %reduce = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<10x10xf32>) -> tensor<10x1xf32>
  %1 = tosa.add %reduce, %arg1 : (tensor<10x1xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = tosa.add %1, %arg2 : (tensor<10x10xf32>, tensor<*xf32>) -> tensor<*xf32>
  %s = tosa.const_shape {values = dense<[10, 10]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %2 = tosa.reshape %0, %s : (tensor<*xf32>, !tosa.shape<2>) -> tensor<10x10xf32>
  return %2 : tensor<10x10xf32>
}

// -----

// CHECK-LABEL: @rescale_unsupported_type
func.func @rescale_unsupported_type(%arg0: tensor<13x21x3x!quant.uniform<u8:f32, 0.015655439347028732:127>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>> {
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32> } : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<127> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<-1> : tensor<1xi8>} : () -> tensor<1xi8>
  // expected-error@+1 {{failed to legalize operation 'tosa.rescale'}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = true, output_unsigned = false} : (tensor<13x21x3x!quant.uniform<u8:f32, 0.015655439347028732:127>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>
}

// -----

func.func @test_add_2d_different_ranks(%arg0: tensor<3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // expected-error@+1 {{'tosa.add' op operands don't have matching ranks}}
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// -----

func.func @cast_unsupported_type(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3x!quant.uniform<i16:f32, 0.078431375324726104:128>> {
  // expected-error@+1 {{failed to legalize operation 'tosa.cast'}}
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi32>) -> tensor<13x21x3x!quant.uniform<i16:f32, 0.078431375324726104:128>>
  return %0 : tensor<13x21x3x!quant.uniform<i16:f32, 0.078431375324726104:128>>
}

// -----

func.func @unranked_reduce(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // expected-error@+1 {{failed to legalize operation 'tosa.reduce_sum'}}
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func.func @unranked_gather(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x26xi32>) -> tensor<*xf32> {
  // expected-error@+1 {{failed to legalize operation 'tosa.gather'}}
  %0 = tosa.gather %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x26xi32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
