//--------------------------------------------------------------------------------------------------
// Enable all supported profiles to focus the verification of expected extension requirement errors.
//--------------------------------------------------------------------------------------------------

// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-validate="profile=pro_int,pro_fp,mt strict-op-spec-alignment"

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

