//--------------------------------------------------------------------------------------------------
// Enable all supported extensions to focus the verification of expected profile requirement errors.
//--------------------------------------------------------------------------------------------------

// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-validate="profile=pro_int extension=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,dynamic,doubleround,inexactround strict-op-spec-alignment"

// -----
func.func @test_const_f16() -> tensor<3x11x11x3xf16> {
  // expected-error@+1 {{'tosa.const' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = "tosa.const"() {values = dense<2.0> : tensor<3x11x11x3xf16>} : () -> tensor<3x11x11x3xf16>
  return %0 : tensor<3x11x11x3xf16>
}

// -----
func.func @test_const_f32() -> tensor<3x11x11x3xf32> {
  // expected-error@+1 {{'tosa.const' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = "tosa.const"() {values = dense<3.0> : tensor<3x11x11x3xf32>} : () -> tensor<3x11x11x3xf32>
  return %0 : tensor<3x11x11x3xf32>
}

// -----
func.func @test_avg_pool2d(%arg0: tensor<1x7x7x9xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1x7x7x9xf32> {
  // expected-error@+1 {{'tosa.avg_pool2d' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.avg_pool2d %arg0, %arg1, %arg2 {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x7x7x9xf32>
  return %0 : tensor<1x7x7x9xf32>
}

// -----
func.func @test_conv2d(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>, %arg2: tensor<8xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x4x4x8xf32> {
  // expected-error@+1 {{'tosa.conv2d' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = true} : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----
func.func @test_conv3d(%arg0: tensor<1x4x8x21x17xf16>, %arg1: tensor<34x1x1x1x17xf16>, %arg2: tensor<34xf16>, %arg3: tensor<1xf16>, %arg4: tensor<1xf16>) -> tensor<1x4x8x21x34xf16> {
  // expected-error@+1 {{'tosa.conv3d' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<1x4x8x21x17xf16>, tensor<34x1x1x1x17xf16>, tensor<34xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x4x8x21x34xf16>
  return %0 : tensor<1x4x8x21x34xf16>
}

// -----
func.func @test_depthwise_conv2d(%arg0: tensor<1x4x4x4xf16>, %arg1: tensor<1x1x4x2xf16>, %arg2: tensor<8xf16>, %arg3: tensor<1xf16>, %arg4: tensor<1xf16>) -> tensor<1x4x4x8xf16> {
  // expected-error@+1 {{'tosa.depthwise_conv2d' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f16, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xf16>, tensor<1x1x4x2xf16>, tensor<8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x4x4x8xf16>
  return %0 : tensor<1x4x4x8xf16>
}

// -----
func.func @test_matmul(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>) -> tensor<1x14x28xf32> {
  // expected-error@+1 {{'tosa.const' op illegal: requires [pro_fp] but not enabled in target}}
  %azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  // expected-error@+1 {{'tosa.const' op illegal: requires [pro_fp] but not enabled in target}}
  %bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  // expected-error@+1 {{'tosa.matmul' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x14x19xf32>, tensor<1x19x28xf32>, tensor<1xf32>, tensor<1xf32>)  -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----
func.func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.sigmoid' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.sigmoid %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----
func.func @test_max_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  // expected-error@+1 {{'tosa.max_pool2d' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----
func.func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xf16>, %arg1: tensor<16x1x1x8xf16>, %arg2: tensor<16xf16>, %arg3: tensor<1xf16>, %arg4: tensor<1xf16>) -> tensor<1x32x32x16xf16> {
  // expected-error@+1 {{'tosa.transpose_conv2d' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f16, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf16>, tensor<16x1x1x8xf16>, tensor<16xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x32x32x16xf16>
  return %0 : tensor<1x32x32x16xf16>
}

// -----
func.func @test_clamp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.clamp' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.clamp %arg0 {min_val = 0.0 : f32, max_val = 1.0 : f32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
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

// -----
func.func @test_max(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.maximum' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.maximum %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_mul(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>, %shift: tensor<1xi8>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.mul' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xf32>, tensor<13x1x3xf32>, tensor<1xi8>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_pow(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.pow' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.pow %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_sub(%arg0: tensor<1x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.sub' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.sub %arg0, %arg1 : (tensor<1x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.abs' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.abs %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.ceil' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.ceil %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_cos(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.cos' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.cos %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.exp' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.exp %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_floor(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.floor' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.floor %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_log(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.log' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.log %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_negate(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.negate' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.negate %arg0, %arg1, %arg2 : (tensor<13x21x3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_reciprocal(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.reciprocal' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.reciprocal %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.rsqrt' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.rsqrt %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_select(%arg0: tensor<1x1x1xi1>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.select' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<1x1x1xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_sin(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.sin' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.sin %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.equal' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.equal %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
func.func @test_greater(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.greater' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.greater %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
func.func @test_reduce_max(%arg0: tensor<13x21x3xf16>) -> tensor<1x21x3xf16> {
  // expected-error@+1 {{'tosa.reduce_max' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<13x21x3xf16>) -> tensor<1x21x3xf16>
  return %0 : tensor<1x21x3xf16>
}

// -----
func.func @test_reduce_sum(%arg0: tensor<13x21x3xf32>) -> tensor<1x21x3xf32> {
  // expected-error@+1 {{'tosa.reduce_sum' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
  return %0 : tensor<1x21x3xf32>
}

// -----
func.func @test_concat(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<26x21x3xf32> {
  // expected-error@+1 {{'tosa.concat' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<26x21x3xf32>
  return %0 : tensor<26x21x3xf32>
}

// -----
func.func @test_pad(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %padding = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
  // expected-error@+1 {{'tosa.const' op illegal: requires [pro_fp] but not enabled in target}}
  %pad_const = "tosa.const"() {values = dense<3.14> : tensor<1xf32>} : () -> tensor<1xf32>
  // expected-error@+1 {{'tosa.pad' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.pad %arg0, %padding, %pad_const : (tensor<13x21x3xf32>, !tosa.shape<6>, tensor<1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x819xf32> {
  %1 = tosa.const_shape {values = dense<[1, 819]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.reshape' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.reshape %arg0, %1 : (tensor<13x21x3xf32>, !tosa.shape<2>) -> tensor<1x819xf32>
  return %0 : tensor<1x819xf32>
}

// -----
func.func @test_reverse(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.reverse' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
func.func @test_slice(%arg0: tensor<13x21x3xf32>) -> tensor<4x11x1xf32> {
  %0 = tosa.const_shape {values = dense<[4, 11, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %1 = tosa.const_shape {values = dense<[6, 8, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  // expected-error@+1 {{'tosa.slice' op illegal: requires [pro_fp] but not enabled in target}}
  %2 = tosa.slice %arg0, %0, %1 : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xf32>
  return %2 : tensor<4x11x1xf32>
}

// -----
func.func @test_tile(%arg0: tensor<13x21x3xf32>) -> tensor<39x21x6xf32> {
  %cst = tosa.const_shape { values = dense<[3, 1, 2]> : tensor<3xindex> } : () -> !tosa.shape<3>
  // expected-error@+1 {{'tosa.tile' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.tile %arg0, %cst: (tensor<13x21x3xf32>, !tosa.shape<3>) -> tensor<39x21x6xf32>
  return %0 : tensor<39x21x6xf32>
}

// -----
func.func @test_transpose(%arg0: tensor<13x21x3xf32>, %arg1: tensor<3xi32>) -> tensor<3x13x21xf32> {
  // expected-error@+1 {{'tosa.transpose' op illegal: requires [pro_fp] but not enabled in target}}
  %1 = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>}: (tensor<13x21x3xf32>) -> tensor<3x13x21xf32>
  return %1 : tensor<3x13x21xf32>
}

// -----
func.func @test_gather(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x26xi32>) -> tensor<13x26x3xf32> {
  // expected-error@+1 {{'tosa.gather' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.gather %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x26xi32>) -> tensor<13x26x3xf32>
  return %0 : tensor<13x26x3xf32>
}

// -----
func.func @test_scatter(%arg0: tensor<13x28x3xf32>, %arg1: tensor<13x26xi32>, %arg2: tensor<13x26x3xf32>) -> tensor<13x28x3xf32> {
  // expected-error@+1 {{'tosa.scatter' op illegal: requires [pro_fp] but not enabled in target}}
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x28x3xf32>, tensor<13x26xi32>, tensor<13x26x3xf32>) -> tensor<13x28x3xf32>
  return %0 : tensor<13x28x3xf32>
}

// -----
func.func @test_resize(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x64x64x8xf32> {
  %scale = tosa.const_shape { values = dense<[4, 2, 4, 2]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<[-1, -1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<[1, 1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.resize' op illegal: requires [pro_fp] but not enabled in target}}
  %1 = tosa.resize %arg0, %scale, %offset, %border { mode = #tosa.resize_mode<BILINEAR> } : (tensor<1x32x32x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x64x64x8xf32>
  return %1 : tensor<1x64x64x8xf32>
}
