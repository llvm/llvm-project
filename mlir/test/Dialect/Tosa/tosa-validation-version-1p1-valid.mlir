// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="specification_version=1.1.draft profiles=pro_int,pro_fp extensions=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,doubleround,inexactround,mxfp,int64,mxfp_conv,shape" -tosa-validate="strict-op-spec-alignment" | FileCheck %s

// -----

// CHECK-LABEL: test_matmul_fp8_mixed_precision_operands
func.func @test_matmul_fp8_mixed_precision_operands(%arg0: tensor<1x14x19xf8E4M3FN>, %arg1: tensor<1x19x28xf8E5M2>) -> tensor<1x14x28xf16> {
  %azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  %bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x14x19xf8E4M3FN>, tensor<1x19x28xf8E5M2>, tensor<1xf8E4M3FN>, tensor<1xf8E5M2>)  -> tensor<1x14x28xf16>
  return %0 : tensor<1x14x28xf16>
}

// -----

// CHECK-LABEL: test_matmul_fp8_input_fp32_acc_type
func.func @test_matmul_fp8_input_fp32_acc_type(%arg0: tensor<1x14x19xf8E4M3FN>, %arg1: tensor<1x19x28xf8E4M3FN>) -> tensor<1x14x28xf32> {
  %azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  %bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x14x19xf8E4M3FN>, tensor<1x19x28xf8E4M3FN>, tensor<1xf8E4M3FN>, tensor<1xf8E4M3FN>)  -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----

// CHECK-LABEL: test_conv2d_fp8_acc32
func.func @test_conv2d_fp8_acc32(%arg0: tensor<1x4x4x4xf8E5M2>, %arg1: tensor<8x1x1x4xf8E5M2>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xf8E5M2>, tensor<8x1x1x4xf8E5M2>, tensor<8xf32>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----

// CHECK-LABEL: test_conv3d_fp8_acc32
func.func @test_conv3d_fp8_acc32(%arg0: tensor<1x4x8x21x17xf8E5M2>, %arg1: tensor<34x1x1x1x17xf8E5M2>, %arg2: tensor<34xf32>) -> tensor<1x4x8x21x34xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<1x4x8x21x17xf8E5M2>, tensor<34x1x1x1x17xf8E5M2>, tensor<34xf32>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x8x21x34xf32>
  return %0 : tensor<1x4x8x21x34xf32>
}

// -----

// CHECK-LABEL: test_depthwise_conv2d_fp8_acc32
func.func @test_depthwise_conv2d_fp8_acc32(%arg0: tensor<1x4x4x4xf8E5M2>, %arg1: tensor<1x1x4x2xf8E5M2>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xf8E5M2>, tensor<1x1x4x2xf8E5M2>, tensor<8xf32>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----

// CHECK-LABEL: test_transpose_conv2d_fp8_acc32
func.func @test_transpose_conv2d_fp8_acc32(%arg0: tensor<1x32x32x8xf8E5M2>, %arg1: tensor<16x1x1x8xf8E5M2>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf8E5M2>, tensor<16x1x1x8xf8E5M2>, tensor<16xf32>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_matmul_t_block_scaled_fp6e2m3
func.func @test_matmul_t_block_scaled_fp6e2m3(%arg0: tensor<4x8x32xf6E2M3FN>, %arg1: tensor<4x8x1xf8E8M0FNU>, %arg2: tensor<4x16x32xf6E2M3FN>, %arg3: tensor<4x16x1xf8E8M0FNU>) -> tensor<4x8x16xf32> {
  %0 = tosa.matmul_t_block_scaled %arg0, %arg1, %arg2, %arg3 {block_size = BLOCK_SIZE_32} : (tensor<4x8x32xf6E2M3FN>, tensor<4x8x1xf8E8M0FNU>, tensor<4x16x32xf6E2M3FN>, tensor<4x16x1xf8E8M0FNU>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: test_argmax_int64
func.func @test_argmax_int64(%arg0: tensor<1x13x13x5xf32>) -> tensor<1x13x13xi64> {
  %0 = tosa.argmax %arg0 {axis = 3 : i32} : (tensor<1x13x13x5xf32>) -> tensor<1x13x13xi64>
  return %0 : tensor<1x13x13xi64>
}

// -----

// CHECK-LABEL: test_const_i64
func.func @test_const_i64() -> tensor<4xi64> {
    %0 = "tosa.const"() {values = dense<[3, 0, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
    return %0 : tensor<4xi64>
}

// -----

// CHECK-LABEL: test_const_fp6e3m2
func.func @test_const_fp6e3m2() -> tensor<4xf6E3M2FN> {
    %0 = "tosa.const"() {values = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf6E3M2FN>} : () -> tensor<4xf6E3M2FN>
    return %0 : tensor<4xf6E3M2FN>
}

// -----

// CHECK-LABEL: test_cast_from_block_scaled_fp8e5m2_fp32
func.func @test_cast_from_block_scaled_fp8e5m2_fp32(%arg0: tensor<4x32xf8E5M2>, %arg1: tensor<4x1xf8E8M0FNU>) -> tensor<4x32xf32> {
  %0 = tosa.cast_from_block_scaled %arg0, %arg1 {block_size = #tosa.block_size<BLOCK_SIZE_32> : i32} : (tensor<4x32xf8E5M2>, tensor<4x1xf8E8M0FNU>) -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// -----

// CHECK-LABEL: test_cast_from_block_scaled_fp8e5m2_bf16
func.func @test_cast_from_block_scaled_fp8e5m2_bf16(%arg0: tensor<4x32xf8E5M2>, %arg1: tensor<4x1xf8E8M0FNU>) -> tensor<4x32xbf16> {
  %0 = tosa.cast_from_block_scaled %arg0, %arg1 {block_size = #tosa.block_size<BLOCK_SIZE_32> : i32} : (tensor<4x32xf8E5M2>, tensor<4x1xf8E8M0FNU>) -> tensor<4x32xbf16>
  return %0 : tensor<4x32xbf16>
}

// -----

// CHECK-LABEL: test_cast_to_block_scaled_static
func.func @test_cast_to_block_scaled_static(%arg0: tensor<4x32xf32>) -> (tensor<4x32xf6E3M2FN>, tensor<4x1xf8E8M0FNU>) {
  %0:2 = tosa.cast_to_block_scaled %arg0 {block_size = #tosa.block_size<BLOCK_SIZE_32>} : (tensor<4x32xf32>) -> (tensor<4x32xf6E3M2FN>, tensor<4x1xf8E8M0FNU>)
  return %0#0, %0#1 : tensor<4x32xf6E3M2FN>, tensor<4x1xf8E8M0FNU>
}

// -----

// CHECK-LABEL: test_cast_to_block_scaled_mxint8
func.func @test_cast_to_block_scaled_mxint8(%arg0: tensor<4x32xf32>) -> (tensor<4x32x!tosa.mxint8>, tensor<4x1xf8E8M0FNU>) {
  %0:2 = tosa.cast_to_block_scaled %arg0 {block_size = #tosa.block_size<BLOCK_SIZE_32> : i32, stochastic_round = false} : (tensor<4x32xf32>) -> (tensor<4x32x!tosa.mxint8>, tensor<4x1xf8E8M0FNU>)
  return %0#0, %0#1 : tensor<4x32x!tosa.mxint8>, tensor<4x1xf8E8M0FNU>
}

// -----

// CHECK-LABEL: test_const_fp6e3m2
func.func @test_const_fp6e3m2() -> tensor<4xf6E3M2FN> {
    %0 = "tosa.const"() {values = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf6E3M2FN>} : () -> tensor<4xf6E3M2FN>
    return %0 : tensor<4xf6E3M2FN>
}

// -----

// CHECK-LABEL: test_const_mxint8
func.func @test_const_mxint8() -> tensor<2x!tosa.mxint8> {
    %0 = "tosa.const"() {values = dense<["0x00", "0x7F"]> : tensor<2x!tosa.mxint8>} : () -> tensor<2x!tosa.mxint8>
    return %0 : tensor<2x!tosa.mxint8>
}

// -----

// CHECK-LABEL: test_matmul_t_block_scaled_mxint8
func.func @test_matmul_t_block_scaled_mxint8(%arg0: tensor<4x8x32x!tosa.mxint8>, %arg1: tensor<4x8x1xf8E8M0FNU>, %arg2: tensor<4x16x32x!tosa.mxint8>, %arg3: tensor<4x16x1xf8E8M0FNU>) -> tensor<4x8x16xf32> {
  %0 = tosa.matmul_t_block_scaled %arg0, %arg1, %arg2, %arg3 {block_size = #tosa.block_size<BLOCK_SIZE_32>} : (tensor<4x8x32x!tosa.mxint8>, tensor<4x8x1xf8E8M0FNU>, tensor<4x16x32x!tosa.mxint8>, tensor<4x16x1xf8E8M0FNU>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: test_cast_to_block_scaled_mxint8
func.func @test_cast_to_block_scaled_mxint8(%arg0: tensor<4x32xf32>) -> (tensor<4x32x!tosa.mxint8>, tensor<4x1xf8E8M0FNU>) {
  %0:2 = tosa.cast_to_block_scaled %arg0 {block_size = #tosa.block_size<BLOCK_SIZE_32> : i32, stochastic_round = false} : (tensor<4x32xf32>) -> (tensor<4x32x!tosa.mxint8>, tensor<4x1xf8E8M0FNU>)
  return %0#0, %0#1 : tensor<4x32x!tosa.mxint8>, tensor<4x1xf8E8M0FNU>
}

// -----

// CHECK-LABEL: test_argmax_fp8_i64
func.func @test_argmax_fp8_i64(%arg0: tensor<12x8x16xf8E5M2>) -> tensor<12x16xi64> {
  %0 = tosa.argmax %arg0 { axis = 1 : i32 } : (tensor<12x8x16xf8E5M2>) -> tensor<12x16xi64>
  return %0 : tensor<12x16xi64>
}

// -----

// CHECK-LABEL: test_argmax_bf16_i64
func.func @test_argmax_bf16_i64(%arg0: tensor<12x8x16xbf16>) -> tensor<12x16xi64> {
  %0 = tosa.argmax %arg0 { axis = 1 : i32 } : (tensor<12x8x16xbf16>) -> tensor<12x16xi64>
  return %0 : tensor<12x16xi64>
}

// -----

// CHECK-LABEL: test_scatter_const_indices_int64
func.func @test_scatter_const_indices_int64(%arg0: tensor<2x52x3xf32>, %arg2: tensor<2x12x3xf32>) -> tensor<2x52x3xf32> {
  %indices = "tosa.const"() { values = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]> : tensor<2x12xi64> } : () -> tensor<2x12xi64>
  %0 = tosa.scatter %arg0, %indices, %arg2 : (tensor<2x52x3xf32>, tensor<2x12xi64>, tensor<2x12x3xf32>) -> tensor<2x52x3xf32>
  return %0 : tensor<2x52x3xf32>
}

// -----

// CHECK-LABEL: test_dynamic_dims
func.func @test_dynamic_dims(%arg0: tensor<?x8x16xi8>) -> tensor<?x16xi32> {
  %0 = tosa.argmax %arg0 { axis = 1 : i32 } : (tensor<?x8x16xi8>) -> tensor<?x16xi32>
  return %0 : tensor<?x16xi32>
}

// -----

// CHECK-LABEL: test_add_shape
func.func @test_add_shape() -> !tosa.shape<4> {
  %a = tosa.const_shape {values = dense<[1, 2, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %b = tosa.const_shape {values = dense<[5, 6, 7, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %c = tosa.add_shape %a, %b : (!tosa.shape<4>, !tosa.shape<4>) -> !tosa.shape<4>
  return %c : !tosa.shape<4>
}

// -----

// CHECK-LABEL: test_dim
func.func @test_dim(%arg0: tensor<1x2x3x4xi32>) -> !tosa.shape<1> {
  %0 = tosa.dim %arg0 {axis = 2 : i32} : (tensor<1x2x3x4xi32>) -> !tosa.shape<1>
  return %0 : !tosa.shape<1>
}

// -----
// CHECK-LABEL: test_exp2_shape
func.func @test_exp2_shape() -> !tosa.shape<4> {
  %a = tosa.const_shape {values = dense<[5, 7, 10, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %b = tosa.exp2_shape %a : (!tosa.shape<4>) -> !tosa.shape<4>
  return %b : !tosa.shape<4>
}

// -----
// CHECK-LABEL: test_log2_ceil_shape
func.func @test_log2_ceil_shape() -> !tosa.shape<4> {
  %a = tosa.const_shape {values = dense<[5, 7, 10, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %b = tosa.log2_ceil_shape %a : (!tosa.shape<4>) -> !tosa.shape<4>
  return %b : !tosa.shape<4>
}

// -----

// CHECK-LABEL: test_mod_shape
func.func @test_mod_shape() -> !tosa.shape<3> {
  %a = tosa.const_shape {values = dense<[10, 11, 12]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %b = tosa.const_shape {values = dense<[3, 5, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %c = tosa.mod_shape %a, %b : (!tosa.shape<3>, !tosa.shape<3>) -> !tosa.shape<3>
  return %c : !tosa.shape<3>
}

// -----

// CHECK-LABEL: test_conv2d_block_scaled
func.func @test_conv2d_block_scaled(%arg0: tensor<1x4x4x64xf4E2M1FN>, %arg1: tensor<1x4x4x2xf8E8M0FNU>, %arg2: tensor<8x1x1x64xf4E2M1FN>, %arg3: tensor<8x1x1x2xf8E8M0FNU>, %arg4: tensor<1xf32>) -> tensor<1x4x4x8xf32> {
  %pad = tosa.const_shape {values = dense<[0, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %stride = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %dilation = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %0 = tosa.conv2d_block_scaled %arg0, %arg1, %arg2, %arg3, %arg4, %pad, %stride, %dilation {block_size = BLOCK_SIZE_32} : (tensor<1x4x4x64xf4E2M1FN>, tensor<1x4x4x2xf8E8M0FNU>, tensor<8x1x1x64xf4E2M1FN>, tensor<8x1x1x2xf8E8M0FNU>, tensor<1xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// CHECK-LABEL: test_assert_equal_shape
func.func @test_assert_equal_shape() {
  %0 = tosa.const_shape {values = dense<[10, 15]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.const_shape {values = dense<[5, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
  tosa.assert_equal_shape %0, %1 {allow_broadcast = true} : (!tosa.shape<2>, !tosa.shape<2>) -> ()
  return
}