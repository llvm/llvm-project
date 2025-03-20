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
func.func @test_variable_read_type(%arg0: tensor<2x4x8xi8>) -> () {
  // expected-error@+1 {{'tosa.variable' op illegal: requires [variable] but not enabled in target}}
  tosa.variable @stored_var = dense<-1> : tensor<2x4x8xi8>
  // expected-error@+1 {{'tosa.variable.read' op illegal: requires [variable]}}
  %0 = tosa.variable.read @stored_var : tensor<2x4x8xi16>
  return
}

// -----
func.func @test_variable_write_type(%arg0: tensor<2x4x8xi16>) -> () {
  // expected-error@+1 {{'tosa.variable' op illegal: requires [variable] but not enabled in target}}
  tosa.variable @stored_var = dense<-1> : tensor<2x4x8xi8>
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
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = "DOUBLE_ROUND", per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----

func.func @test_inexact_round_rescale(%arg0: tensor<13x21x3xi8>) -> tensor<13x21x3xi8> {
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32> } : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.rescale' op failed attribute check: rounding_mode = INEXACT_ROUND requires extension [inexactround]}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = "INEXACT_ROUND", per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----

func.func @test_pad_non_const(%arg0: tensor<13x21x3xi8>, %arg1: tensor<1xi8>) -> tensor<13x21x3xi8> {
  %0 = tosa.const_shape {values = dense<[0, 0, 0, 1, 0, 1]> : tensor<6xindex>} : () -> !tosa.shape<6>
  // expected-error@+1 {{'tosa.pad' op expected compile time resolvable constant, but got variable value for operand #2}}
  %1 = tosa.pad %arg0, %0, %arg1 : (tensor<13x21x3xi8>, !tosa.shape<6>, tensor<1xi8>) -> tensor<13x21x3xi8>
  return %1 : tensor<13x21x3xi8>
}

// -----

func.func @test_rescale_non_const_multiplier(%arg0: tensor<13x21x3xi32>, %multiplier: tensor<1xi32>) -> tensor<13x21x3xi32> {
  %zps = "tosa.const"() {values = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.rescale' op expected compile time resolvable constant, but got variable value for operand #1}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %zps, %zps {rounding_mode = "SINGLE_ROUND", input_zp = 0 : i32, output_zp = 0 : i32, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----

func.func @test_rescale_non_const_shift(%arg0: tensor<13x21x3xi32>, %shift: tensor<1xi8>) -> tensor<13x21x3xi32> {
  %zps = "tosa.const"() {values = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32> } : () -> tensor<1xi32>
  // expected-error@+1 {{'tosa.rescale' op expected compile time resolvable constant, but got variable value for operand #2}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %zps, %zps {rounding_mode = "SINGLE_ROUND", input_zp = 0 : i32, output_zp = 0 : i32, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----

func.func @test_conv2d_non_const_input_zp(%arg0: tensor<1x4x4x4xi8>, %arg1: tensor<8x1x1x4xi8>, %arg2: tensor<8xi32>, %arg3: tensor<1xi8>) -> tensor<1x4x4x8xi32> {
  %weight_zp = "tosa.const"() {values = dense<0> : tensor<1xi8> } : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.conv2d' op expected compile time resolvable constant, but got variable value for operand #3}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %weight_zp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = true} : (tensor<1x4x4x4xi8>, tensor<8x1x1x4xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// -----

func.func @test_conv3d_non_const_weight_zp(%arg0: tensor<1x4x8x21x17xi8>, %arg1: tensor<34x1x1x1x17xi8>, %arg2: tensor<34xi32>, %arg3: tensor<1xi8>) -> tensor<1x4x8x21x34xi32> {
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8> } : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.conv3d' op expected compile time resolvable constant, but got variable value for operand #4}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %input_zp, %arg3 {acc_type = i32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<1x4x8x21x17xi8>, tensor<34x1x1x1x17xi8>, tensor<34xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x8x21x34xi32>
  return %0 : tensor<1x4x8x21x34xi32>
}

// -----

func.func @test_depthwise_conv2d_non_const_input_zp(%arg0: tensor<1x4x4x4xi8>, %arg1: tensor<1x1x4x2xi8>, %arg2: tensor<8xi32>, %arg3: tensor<1xi8>) -> tensor<1x4x4x8xi32> {
  %weight_zp = "tosa.const"() {values = dense<0> : tensor<1xi8> } : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.depthwise_conv2d' op expected compile time resolvable constant, but got variable value for operand #3}}
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %weight_zp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xi8>, tensor<1x1x4x2xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// -----

func.func @test_transpose_conv2d_non_const_weight_zp(%arg0: tensor<1x4x4x4xi8>, %arg1: tensor<1x1x4x2xi8>, %arg2: tensor<8xi32>, %arg3: tensor<1xi8>) -> tensor<1x4x4x8xi32> {
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8> } : () -> tensor<1xi8>
  // expected-error@+1 {{'tosa.transpose_conv2d' op expected compile time resolvable constant, but got variable value for operand #4}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp, %arg3 {acc_type = i32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xi8>, tensor<1x1x4x2xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x4x8xi32>
  return %0 : tensor<1x4x4x8xi32>
}

// -----

func.func @test_matmul_non_const_a_zp(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>, %a_zp: tensor<1xf32>, %b_zp: tensor<1xf32>) -> tensor<1x14x28xf32> {
  // expected-error@+1 {{'tosa.matmul' op expected compile time resolvable constant, but got variable value for operand #2}}
  %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp : (tensor<1x14x19xf32>, tensor<1x19x28xf32>, tensor<1xf32>, tensor<1xf32>)  -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----

func.func @test_matmul_non_const_b_zp(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>, %b_zp: tensor<1xf32>) -> tensor<1x14x28xf32> {
  %a_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32> } : () -> tensor<1xf32>
  // expected-error@+1 {{'tosa.matmul' op expected compile time resolvable constant, but got variable value for operand #3}}
  %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp : (tensor<1x14x19xf32>, tensor<1x19x28xf32>, tensor<1xf32>, tensor<1xf32>)  -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----

func.func @test_mul_non_const(%arg0: tensor<13x21x3xi8>, %arg1: tensor<13x1x3xi8>, %shift: tensor<1xi8>) -> tensor<13x21x3xi8> {
  // expected-error@+1 {{'tosa.mul' op expected compile time resolvable constant, but got variable value for operand #2}}
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xi8>, tensor<13x1x3xi8>, tensor<1xi8>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----

func.func @test_table_non_const(%arg0 : tensor<4x5xi8>, %arg1 : tensor<513xi8>) -> () {
  // expected-error@+1 {{'tosa.table' op expected compile time resolvable constant, but got variable value for operand #1}}
  %0 = tosa.table %arg0, %arg1 : (tensor<4x5xi8>, tensor<513xi8>) -> tensor<4x5xi8>
  return
}

// -----

func.func @test_rescale_non_const_input_zp(%arg0: tensor<13x21x3xi32>, %input_zp: tensor<1xi32>) -> tensor<13x21x3xi32> {
  %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32> } : () -> tensor<1xi32>
  %output_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // expected-error@+1 {{'tosa.rescale' op expected compile time resolvable constant, but got variable value for operand #3}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = true, rounding_mode = "SINGLE_ROUND", per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----

func.func @test_negate_non_const_input1_zp(%arg0: tensor<1xf32>, %input_zp: tensor<1xf32>) -> tensor<1xf32> {
  %output_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  // expected-error@+1 {{'tosa.negate' op expected compile time resolvable constant, but got variable value for operand #1}}
  %0 = tosa.negate %arg0, %input_zp, %output_zp : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

func.func @test_avg_pool2d_non_const_output_zp(%arg0: tensor<1x32x32x8xf32>, %output_zp: tensor<1xf32>) -> tensor<1x32x32x8xf32> {
  %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  // expected-error@+1 {{'tosa.avg_pool2d' op expected compile time resolvable constant, but got variable value for operand #2}}
  %0 = "tosa.avg_pool2d"(%arg0, %input_zp, %output_zp) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, acc_type = f32} :
         (tensor<1x32x32x8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}
