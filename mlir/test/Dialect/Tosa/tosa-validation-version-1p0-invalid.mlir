// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="specification_version=1.0 profiles=pro_int,pro_fp extensions=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,dynamic,doubleround,inexactround" -tosa-validate="strict-op-spec-alignment"

// -----

func.func @test_matmul_fp8_mixed_precision_operands(%arg0: tensor<1x14x19xf8E4M3FN>, %arg1: tensor<1x19x28xf8E5M2>) -> tensor<1x14x28xf16> {
  %azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  %bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  // expected-error@+1 {{'tosa.matmul' op illegal: the target specification version (1.0) is not backwards compatible with the op compliance specification version (1.1)}}
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x14x19xf8E4M3FN>, tensor<1x19x28xf8E5M2>, tensor<1xf8E4M3FN>, tensor<1xf8E5M2>)  -> tensor<1x14x28xf16>
  return %0 : tensor<1x14x28xf16>
}

// -----

func.func @test_matmul_fp8_input_fp32_acc_type(%arg0: tensor<1x14x19xf8E4M3FN>, %arg1: tensor<1x19x28xf8E4M3FN>) -> tensor<1x14x28xf32> {
  %azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  %bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  // expected-error@+1 {{'tosa.matmul' op illegal: the target specification version (1.0) is not backwards compatible with the op compliance specification version (1.1)}}
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x14x19xf8E4M3FN>, tensor<1x19x28xf8E4M3FN>, tensor<1xf8E4M3FN>, tensor<1xf8E4M3FN>)  -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----

func.func @test_conv2d_fp8_acc32(%arg0: tensor<1x4x4x4xf8E5M2>, %arg1: tensor<8x1x1x4xf8E5M2>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  // expected-error@+1 {{'tosa.conv2d' op illegal: the target specification version (1.0) is not backwards compatible with the op compliance specification version (1.1)}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xf8E5M2>, tensor<8x1x1x4xf8E5M2>, tensor<8xf32>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----

func.func @test_conv3d_fp8_acc32(%arg0: tensor<1x4x8x21x17xf8E5M2>, %arg1: tensor<34x1x1x1x17xf8E5M2>, %arg2: tensor<34xf32>) -> tensor<1x4x8x21x34xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  // expected-error@+1 {{'tosa.conv3d' op illegal: the target specification version (1.0) is not backwards compatible with the op compliance specification version (1.1)}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<1x4x8x21x17xf8E5M2>, tensor<34x1x1x1x17xf8E5M2>, tensor<34xf32>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x8x21x34xf32>
  return %0 : tensor<1x4x8x21x34xf32>
}

// -----

func.func @test_depthwise_conv2d_fp8_acc32(%arg0: tensor<1x4x4x4xf8E5M2>, %arg1: tensor<1x1x4x2xf8E5M2>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  // expected-error@+1 {{'tosa.depthwise_conv2d' op illegal: the target specification version (1.0) is not backwards compatible with the op compliance specification version (1.1)}}
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xf8E5M2>, tensor<1x1x4x2xf8E5M2>, tensor<8xf32>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----

func.func @test_transpose_conv2d_fp8_acc32(%arg0: tensor<1x32x32x8xf8E5M2>, %arg1: tensor<16x1x1x8xf8E5M2>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  // expected-error@+1 {{'tosa.transpose_conv2d' op illegal: the target specification version (1.0) is not backwards compatible with the op compliance specification version (1.1)}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf8E5M2>, tensor<16x1x1x8xf8E5M2>, tensor<16xf32>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_dyanmic_dims(%arg0: tensor<?x8x16xi8>) -> tensor<?x16xi32> {
  // expected-error@+1 {{'tosa.argmax' op failed level check: operand shape dimension cannot be dynamic when targeting TOSA specification version 1.0 or below}}
  %0 = tosa.argmax %arg0 { axis = 1 : i32 } : (tensor<?x8x16xi8>) -> tensor<?x16xi32>
  return %0 : tensor<?x16xi32>
}
