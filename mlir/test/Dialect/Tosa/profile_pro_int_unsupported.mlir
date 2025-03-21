//--------------------------------------------------------------------------------------------------
// Enable all supported extensions to focus the verification of expected profile requirement errors.
//--------------------------------------------------------------------------------------------------

// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-validate="profile=pro_fp extension=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,doubleround,inexactround strict-op-spec-alignment"

// -----
func.func @test_table(%arg0 : tensor<4x5xi8>, %arg1 : tensor<513xi8>) -> () {
  // expected-error@+1 {{'tosa.table' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.table %arg0, %arg1 : (tensor<4x5xi8>, tensor<513xi8>) -> tensor<?x?xi8>
  return
}

// -----
func.func @test_reduce_max(%arg0: tensor<13x21x3xi16>) -> tensor<1x21x3xi16> {
  // expected-error@+1 {{'tosa.reduce_max' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<13x21x3xi16>) -> tensor<1x21x3xi16>
  return %0 : tensor<1x21x3xi16>
}

// -----
func.func @test_cast_i8_i32(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xi8> {
 // expected-error@+1 {{'tosa.cast' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi32>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----
func.func @test_rescale(%arg0: tensor<13x21x3xi8>) -> tensor<13x21x3xi32> {
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<127> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error@+1 {{'tosa.rescale' op illegal: requires [pro_int] but not enabled in target}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = "SINGLE_ROUND", scale32 = true, per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}
