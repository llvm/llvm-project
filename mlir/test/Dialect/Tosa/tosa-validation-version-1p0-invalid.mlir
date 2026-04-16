// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="specification_version=1.0 profiles=pro_int,pro_fp extensions=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,dynamic,doubleround,inexactround" -tosa-validate="strict-op-spec-alignment"

// -----

func.func @test_matmul_fp8_mixed_precision_operands(%arg0: tensor<1x14x19xf8E4M3FN>, %arg1: tensor<1x19x28xf8E5M2>) -> tensor<1x14x28xf16> {
  %azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  %bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  // expected-error@+1 {{'tosa.matmul' op illegal: requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x14x19xf8E4M3FN>, tensor<1x19x28xf8E5M2>, tensor<1xf8E4M3FN>, tensor<1xf8E5M2>)  -> tensor<1x14x28xf16>
  return %0 : tensor<1x14x28xf16>
}

// -----

func.func @test_matmul_fp8_input_fp32_acc_type(%arg0: tensor<1x14x19xf8E4M3FN>, %arg1: tensor<1x19x28xf8E4M3FN>) -> tensor<1x14x28xf32> {
  %azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  %bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  // expected-error@+1 {{'tosa.matmul' op illegal: requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x14x19xf8E4M3FN>, tensor<1x19x28xf8E4M3FN>, tensor<1xf8E4M3FN>, tensor<1xf8E4M3FN>)  -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----

func.func @test_conv2d_fp8_acc32(%arg0: tensor<1x4x4x4xf8E5M2>, %arg1: tensor<8x1x1x4xf8E5M2>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  // expected-error@+1 {{'tosa.conv2d' op illegal: requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xf8E5M2>, tensor<8x1x1x4xf8E5M2>, tensor<8xf32>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----

func.func @test_conv3d_fp8_acc32(%arg0: tensor<1x4x8x21x17xf8E5M2>, %arg1: tensor<34x1x1x1x17xf8E5M2>, %arg2: tensor<34xf32>) -> tensor<1x4x8x21x34xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  // expected-error@+1 {{'tosa.conv3d' op illegal: requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<1x4x8x21x17xf8E5M2>, tensor<34x1x1x1x17xf8E5M2>, tensor<34xf32>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x8x21x34xf32>
  return %0 : tensor<1x4x8x21x34xf32>
}

// -----

func.func @test_depthwise_conv2d_fp8_acc32(%arg0: tensor<1x4x4x4xf8E5M2>, %arg1: tensor<1x1x4x2xf8E5M2>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  // expected-error@+1 {{'tosa.depthwise_conv2d' op illegal: requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xf8E5M2>, tensor<1x1x4x2xf8E5M2>, tensor<8xf32>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----

func.func @test_transpose_conv2d_fp8_acc32(%arg0: tensor<1x32x32x8xf8E5M2>, %arg1: tensor<16x1x1x8xf8E5M2>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  // expected-error@+1 {{'tosa.transpose_conv2d' op illegal: requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf8E5M2>, tensor<16x1x1x8xf8E5M2>, tensor<16xf32>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @test_gather_bool_i64(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x26xi64>) -> tensor<13x26x3xi1> {
  // expected-error@+1 {{'tosa.gather' op illegal: requires specification version compatible with 1.1 (got 1.0) and requires any of [int64] profiles/extensions to be specified in the target environment}}
  %0 = tosa.gather %arg0, %arg1 : (tensor<13x21x3xi1>, tensor<13x26xi64>) -> tensor<13x26x3xi1>
  return %0 : tensor<13x26x3xi1>
}

// -----

func.func @test_gather_bool_i32(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x26xi32>) -> tensor<13x26x3xi1> {
  // expected-error@+1 {{'tosa.gather' op illegal: requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %0 = tosa.gather %arg0, %arg1 : (tensor<13x21x3xi1>, tensor<13x26xi32>) -> tensor<13x26x3xi1>
  return %0 : tensor<13x26x3xi1>
}

// -----

func.func @test_row_gather_block_scaled_i8_i32(%arg0: tensor<13x21x3xi8>, %arg1: tensor<13x26xi32>) -> tensor<13x52x3xi8> {
  %row_count = "tosa.const"() {values = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error@+1 {{'tosa.row_gather_block_scaled' op illegal: requires specification version compatible with 1.1 (got 1.0) OR requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %0 = tosa.row_gather_block_scaled %arg0, %arg1, %row_count {block_size = #tosa.block_size<BLOCK_SIZE_1>} : (tensor<13x21x3xi8>, tensor<13x26xi32>, tensor<1xi32>) -> (tensor<13x52x3xi8>)
  return %0 : tensor<13x52x3xi8>
}

// -----

func.func @test_scatter_bool_i64(%arg0: tensor<13x52x3xi1>, %arg1: tensor<13x26xi64>, %arg2: tensor<13x26x3xi1>) -> tensor<13x52x3xi1> {
  // expected-error@+1 {{'tosa.scatter' op illegal: requires specification version compatible with 1.1 (got 1.0) and requires any of [int64] profiles/extensions to be specified in the target environment}}
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x52x3xi1>, tensor<13x26xi64>, tensor<13x26x3xi1>) -> tensor<13x52x3xi1>
  return %0 : tensor<13x52x3xi1>
}

// -----

func.func @test_scatter_bool_i32(%arg0: tensor<13x52x3xi1>, %arg1: tensor<13x26xi32>, %arg2: tensor<13x26x3xi1>) -> tensor<13x52x3xi1> {
  // expected-error@+1 {{'tosa.scatter' op illegal: requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x52x3xi1>, tensor<13x26xi32>, tensor<13x26x3xi1>) -> tensor<13x52x3xi1>
  return %0 : tensor<13x52x3xi1>
}

// -----

func.func @test_cast_bool_fp32(%arg0: tensor<13x21x3xi1>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.cast' op illegal: requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi1>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

func.func @test_cast_bool_i64(%arg0: tensor<13x21x3xi1>) -> tensor<13x21x3xi64> {
  // expected-error@+1 {{'tosa.cast' op illegal: requires specification version compatible with 1.1 (got 1.0) and requires any of [int64] profiles/extensions to be specified in the target environment}}
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi1>) -> tensor<13x21x3xi64>
  return %0 : tensor<13x21x3xi64>
}

// -----

func.func @test_cast_fp32_bool(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.cast' op illegal: requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %0 = tosa.cast %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

func.func @test_cast_i64_bool(%arg0: tensor<13x21x3xi64>) -> tensor<13x21x3xi1> {
  // expected-error@+1 {{'tosa.cast' op illegal: requires specification version compatible with 1.1 (got 1.0) and requires any of [int64] profiles/extensions to be specified in the target environment}}
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi64>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

func.func @test_dyanmic_dims(%arg0: tensor<?x8x16xi8>) -> tensor<?x16xi32> {
  // expected-error@+1 {{'tosa.argmax' op failed level check: operand shape dimension cannot be dynamic when targeting TOSA specification version 1.0 or below}}
  %0 = tosa.argmax %arg0 { axis = 1 : i32 } : (tensor<?x8x16xi8>) -> tensor<?x16xi32>
  return %0 : tensor<?x16xi32>
}

// -----

func.func @test_matmul_t_block_scaled(%arg0: tensor<4x8x32xf8E4M3FN>, %arg1: tensor<4x8x1xf8E8M0FNU>, %arg2: tensor<4x16x32xf8E4M3FN>, %arg3: tensor<4x16x1xf8E8M0FNU>) -> tensor<4x8x16xf32> {
  // expected-error@+1 {{'tosa.matmul_t_block_scaled' op illegal: requires specification version compatible with 1.1 (got 1.0) and requires any of [mxfp] profiles/extensions to be specified in the target environment}}
  %0 = tosa.matmul_t_block_scaled %arg0, %arg1, %arg2, %arg3 {block_size = #tosa.block_size<BLOCK_SIZE_32>} : (tensor<4x8x32xf8E4M3FN>, tensor<4x8x1xf8E8M0FNU>, tensor<4x16x32xf8E4M3FN>, tensor<4x16x1xf8E8M0FNU>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// -----

func.func @test_argmax_int64(%arg0: tensor<1x13x13x5xf32>) -> tensor<1x13x13xi64> {
  // expected-error@+1 {{'tosa.argmax' op illegal: requires specification version compatible with 1.1 (got 1.0) and requires any of [int64] profiles/extensions to be specified in the target environment}}
  %0 = tosa.argmax %arg0 {axis = 3 : i32} : (tensor<1x13x13x5xf32>) -> tensor<1x13x13xi64>
  return %0 : tensor<1x13x13xi64>
}

// -----
func.func @test_const_fp6e3m2(%arg0 : index) -> tensor<4xf6E3M2FN> {
  // expected-error@+1 {{'tosa.const' op illegal: requires specification version compatible with 1.1 (got 1.0) and requires any of [mxfp] profiles/extensions to be specified in the target environment}}
    %0 = "tosa.const"() {values = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf6E3M2FN>} : () -> tensor<4xf6E3M2FN>
    return %0 : tensor<4xf6E3M2FN>
}

// -----

func.func @test_cast_from_block_scaled(%arg0: tensor<4x32xf8E5M2>, %arg1: tensor<4x1xf8E8M0FNU>) -> tensor<4x32xf32> {
  // expected-error@+1 {{'tosa.cast_from_block_scaled' op illegal: requires specification version compatible with 1.1 (got 1.0) and requires any of [mxfp] profiles/extensions to be specified in the target environment}}
  %0 = tosa.cast_from_block_scaled %arg0, %arg1 {block_size = #tosa.block_size<BLOCK_SIZE_32> : i32} : (tensor<4x32xf8E5M2>, tensor<4x1xf8E8M0FNU>) -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// -----

func.func @test_cast_to_block_scaled(%arg0: tensor<4x32xf32>) -> (tensor<4x32xf6E3M2FN>, tensor<4x1xf8E8M0FNU>) {
  // expected-error@+1 {{'tosa.cast_to_block_scaled' op illegal: requires specification version compatible with 1.1 (got 1.0) and requires any of [mxfp] profiles/extensions to be specified in the target environment}}
  %0:2 = tosa.cast_to_block_scaled %arg0 {block_size = #tosa.block_size<BLOCK_SIZE_32>} : (tensor<4x32xf32>) -> (tensor<4x32xf6E3M2FN>, tensor<4x1xf8E8M0FNU>)
  return %0#0, %0#1 : tensor<4x32xf6E3M2FN>, tensor<4x1xf8E8M0FNU>
}

// -----

func.func @test_conv2d_block_scaled(%arg0: tensor<*xf4E2M1FN>, %arg1: tensor<*xf8E8M0FNU>, %arg2: tensor<*xf4E2M1FN>, %arg3: tensor<*xf8E8M0FNU>, %arg4: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tosa.const_shape {values = dense<[0, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %1 = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %2 = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.conv2d_block_scaled' op illegal: requires specification version compatible with 1.1 (got 1.0) and requires any of [mxfp_conv] profiles/extensions to be specified in the target environment}}
  %3 = tosa.conv2d_block_scaled %arg0, %arg1, %arg2, %arg3, %arg4, %0, %1, %2 {block_size = BLOCK_SIZE_32} : (tensor<*xf4E2M1FN>, tensor<*xf8E8M0FNU>, tensor<*xf4E2M1FN>, tensor<*xf8E8M0FNU>, tensor<*xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<*xf32>
  return %3 : tensor<*xf32>
}

// -----
func.func @test_maxpool2d_adaptive(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> { 
  %kernel = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %stride = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %pad = tosa.const_shape {values = dense<[0, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // expected-error@+1 {{'tosa.max_pool2d_adaptive' op illegal: requires specification version compatible with 1.1 (got 1.0) to be specified in the target environment}}
  %0 = tosa.max_pool2d_adaptive %arg0, %kernel, %stride, %pad :
         (tensor<1x32x32x8xf32>, !tosa.shape<2>, !tosa.shape<2>, !tosa.shape<4>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}
