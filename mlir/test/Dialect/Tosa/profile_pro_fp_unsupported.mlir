//--------------------------------------------------------------------------------------------------
// Enable all supported extensions to focus the verification of expected profile requirement errors.
//--------------------------------------------------------------------------------------------------

// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-validate="profile=pro_int extension=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable strict-op-spec-alignment"

// -----
func.func @test_conv2d(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
  // expected-error@+1 {{'tosa.conv2d' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = true} : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----
func.func @test_avg_pool2d_f32(%arg0: tensor<1x7x7x9xf32>) -> tensor<1x7x7x9xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.avg_pool2d %arg0 {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xf32>) -> tensor<1x7x7x9xf32>
  return %0 : tensor<1x7x7x9xf32>
}

// -----
func.func @test_matmul(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>) -> tensor<1x14x28xf32> {
  // expected-error@+1 {{'tosa.matmul' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.matmul %arg0, %arg1 : (tensor<1x14x19xf32>, tensor<1x19x28xf32>) -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----
func.func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.sigmoid' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.sigmoid %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----
func.func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.add' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.add %arg0, %arg1 : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_concat(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<26x21x3xf32> {
  // expected-error@+1 {{'tosa.concat' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<26x21x3xf32>
  return %0 : tensor<26x21x3xf32>
}

// -----
func.func @test_cast_i32_f32(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.cast' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

