//--------------------------------------------------------------------------------------------------
// Enable all supported profiles to focus the verification of expected extension requirement errors.
//--------------------------------------------------------------------------------------------------

// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-validate="profile=pro_int,pro_fp strict-op-spec-alignment"

// -----
func.func @test_fft2d(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
  // expected-error@+1 {{'tosa.fft2d' op illegal: requires [fft] but not enabled in target}}
  %0, %1 = tosa.fft2d %arg0, %arg1 {inverse = false} : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>)
  return %0, %1 : tensor<1x4x8xf32>, tensor<1x4x8xf32>
}

// -----
func.func @test_variable_read_type(%arg0: tensor<2x4x8xi32>) -> () {
  // expected-error@+1 {{'tosa.variable' op illegal: requires [variable] but not enabled in target}}
  tosa.variable @stored_var = dense<-1> : tensor<2x4x8xi32>
  // expected-error@+1 {{'tosa.variable.read' op illegal: requires [variable]}}
  %0 = tosa.variable.read @stored_var : tensor<2x4x8xi16>
  return
}

// -----
func.func @test_variable_write_type(%arg0: tensor<2x4x8xi16>) -> () {
  // expected-error@+1 {{'tosa.variable' op illegal: requires [variable] but not enabled in target}}
  tosa.variable @stored_var = dense<-1> : tensor<2x4x8xi32>
  // expected-error@+1 {{'tosa.variable.write' op illegal: requires [variable]}}
  tosa.variable.write @stored_var, %arg0 : tensor<2x4x8xi16>
  return
}

// -----
func.func @test_cast_bf16_i32(%arg0: tensor<13x21x3xbf16>) -> tensor<13x21x3xi32> {
  // expected-error@+1 {{'tosa.cast' op illegal: requires [bf16] but not enabled in target}}
  %0 = tosa.cast %arg0 : (tensor<13x21x3xbf16>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
func.func @test_cond_if(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> tensor<f32> {
  // expected-error@+1 {{'tosa.cond_if' op illegal: requires [controlflow]}}
  %0 = tosa.cond_if %arg2 -> (tensor<f32>) {
    %1 = tosa.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tosa.yield %1 : tensor<f32>
  } else {
    %1 = tosa.sub %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tosa.yield %1 : tensor<f32>
  }
  return %0 : tensor<f32>
}

// -----
func.func @test_while_loop(%arg0: tensor<10xi32>, %arg1: tensor<i32>) {
  %0 = "tosa.const"() {values = dense<0> : tensor<i32>} : () -> tensor<i32>
  // expected-error@+1 {{'tosa.while_loop' op illegal: requires [controlflow]}}
  %1:3 = tosa.while_loop (%arg2 = %0, %arg3 = %0, %arg4 = %arg0) : (tensor<i32>, tensor<i32>, tensor<10xi32>) -> (tensor<i32>, tensor<i32>, tensor<10xi32>) {
    %2 = tosa.greater_equal %arg3, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = tosa.logical_not %2 : (tensor<i1>) -> tensor<i1>
    tosa.yield %3 : tensor<i1>
  } do {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<10xi32>):
    %2 = "tosa.const"() {values = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = tosa.add %arg3, %2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = tosa.const_shape {values = dense<[1]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.reshape %2, %7 : (tensor<i32>, !tosa.shape<1>) -> tensor<1xi32>
    %5 = tosa.add %arg4, %4 : (tensor<10xi32>, tensor<1xi32>) -> tensor<10xi32>
    %6 = tosa.add %arg2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tosa.yield %6, %3, %5 : tensor<i32>, tensor<i32>, tensor<10xi32>
  }
  return
}

// -----

// CHECK-LABEL: test_single_round_rescale
func.func @test_single_round_rescale(%arg0: tensor<13x21x3xi8>) -> tensor<13x21x3xi8> {
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32> } : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  // CHECK tosa.rescale
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = "SINGLE_ROUND", input_zp = 127 : i32, output_zp = -1 : i32, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----

func.func @test_double_round_rescale(%arg0: tensor<13x21x3xi8>) -> tensor<13x21x3xi8> {
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32> } : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.rescale' op failed attribute check: rounding_mode = DOUBLE_ROUND requires extension [doubleround]}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = "DOUBLE_ROUND", input_zp = 127 : i32, output_zp = -1 : i32, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----

func.func @test_inexact_round_rescale(%arg0: tensor<13x21x3xi8>) -> tensor<13x21x3xi8> {
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32> } : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.rescale' op failed attribute check: rounding_mode = INEXACT_ROUND requires extension [inexactround]}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = "INEXACT_ROUND", input_zp = 127 : i32, output_zp = -1 : i32, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}
