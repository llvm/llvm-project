// RUN: mlir-opt %s --split-input-file --tosa-to-linalg-pipeline -verify-diagnostics | FileCheck %s


// -----

// check that -tosa-validate level checking kick in
func.func @tensor_with_unknown_rank(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  // expected-error@+1 {{'tosa.abs' op failed level check: unranked tensor}}
  %0 = "tosa.abs"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>
}

// -----

// check that tosa verify kick in
func.func @test_avg_pool2d_zero_dim_input(%arg0: tensor<1x0x?x9xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1x7x7x9xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op operand #0 must be 4-d tosa-conformant tensor, but got 'tensor<1x0x?x9xf32>'}}
    %0 = "tosa.avg_pool2d"(%arg0, %arg1, %arg2) {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>}
      : (tensor<1x0x?x9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x7x7x9xf32>
    return %0 : tensor<1x7x7x9xf32>
}

// -----

// check that --tosa-to-linalg kick in
func.func @avg_pool2d_with_unsupported_quant_type(%arg0: tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>, %arg1: tensor<1xi8>, %arg2: tensor<1xi8>) -> tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>> {
  // expected-error@+1 {{failed to legalize operation 'tosa.avg_pool2d'}}
  %0 = "tosa.avg_pool2d"(%arg0, %arg1, %arg2) {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>
  return %0 : tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>
}

// -----

// CHECK-LABEL: rescale_doubleround
func.func @rescale_doubleround(%arg0: tensor<8x9x7x14xi32>) -> tensor<8x9x7x14xi8> {
  %0 = "tosa.const"() <{values = dense<0> : tensor<14xi32>}> : () -> tensor<14xi32>
  %1 = "tosa.const"() <{values = dense<0> : tensor<14xi8>}> : () -> tensor<14xi8>
  %2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %3 = "tosa.const"() <{values = dense<-5> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: tosa.apply_scale
  %4 = tosa.rescale %arg0, %0, %1, %2, %3 {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<8x9x7x14xi32>, tensor<14xi32>, tensor<14xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<8x9x7x14xi8>
  return %4 : tensor<8x9x7x14xi8>
}

// -----

func.func @rescale_inexactround(%arg0: tensor<8x9x7x14xi32>) -> tensor<8x9x7x14xi8> {
  %0 = "tosa.const"() <{values = dense<0> : tensor<14xi32>}> : () -> tensor<14xi32>
  %1 = "tosa.const"() <{values = dense<0> : tensor<14xi8>}> : () -> tensor<14xi8>
  %2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %3 = "tosa.const"() <{values = dense<-5> : tensor<1xi8>}> : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.rescale' op failed attribute check: rounding_mode = INEXACT_ROUND requires extension [inexactround]}}
  %4 = tosa.rescale %arg0, %0, %1, %2, %3 {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = INEXACT_ROUND, scale32 = true} : (tensor<8x9x7x14xi32>, tensor<14xi32>, tensor<14xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<8x9x7x14xi8>
  return %4 : tensor<8x9x7x14xi8>
}
