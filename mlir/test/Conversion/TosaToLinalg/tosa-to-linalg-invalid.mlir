// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-linalg))" %s -verify-diagnostics

// CHECK-LABEL: @avg_pool2d_with_unsupported_quant_type
func.func @avg_pool2d_with_unsupported_quant_type(%arg0: tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>) -> tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>> {
  // expected-error@+1 {{failed to legalize operation 'tosa.avg_pool2d'}}
  %0 = "tosa.avg_pool2d"(%arg0) {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>) -> tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>
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
  %2 = tosa.reshape %0 {new_shape = array<i64: 10, 10>} : (tensor<*xf32>) -> tensor<10x10xf32>
  return %2 : tensor<10x10xf32>
}

// -----

// CHECK-LABEL: @rfft2d_with_non_float_type
func.func @rfft2d_with_non_float_type(%arg0 : tensor<1x1x1xi32>) -> (tensor<1x1x1xi32>, tensor<1x1x1xi32>) {
  // expected-error@+1 {{failed to legalize operation 'tosa.rfft2d'}}
  %real, %imag = tosa.rfft2d %arg0 : (tensor<1x1x1xi32>) -> (tensor<1x1x1xi32>, tensor<1x1x1xi32>)
  return %real, %imag : tensor<1x1x1xi32>, tensor<1x1x1xi32>
}

// -----

// CHECK-LABEL: @rescale_unsupported_type
func.func @rescale_unsupported_type(%arg0: tensor<13x21x3x!quant.uniform<u8:f32, 0.015655439347028732:127>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>> {
  // expected-error@+1 {{failed to legalize operation 'tosa.rescale'}}
  %0 = tosa.rescale %arg0 {double_round = false, input_zp = 127 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<13x21x3x!quant.uniform<u8:f32, 0.015655439347028732:127>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>
}

// -----

func.func @test_add_2d_different_ranks(%arg0: tensor<3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // expected-error@+1 {{'tosa.add' op operands don't have matching ranks}}
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}
