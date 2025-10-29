// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="specification_version=1.1.draft profiles=pro_int,pro_fp extensions=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,doubleround,inexactround,mxfp,int64" -tosa-validate="strict-op-spec-alignment" | FileCheck %s

// -----

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
func.func @test_const_i64(%arg0 : index) -> tensor<4xi64> {
    %0 = "tosa.const"() {values = dense<[3, 0, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
    return %0 : tensor<4xi64>
}

// -----

// CHECK-LABEL: test_const_fp6e3m2
func.func @test_const_fp6e3m2(%arg0 : index) -> tensor<4xf6E3M2FN> {
    %0 = "tosa.const"() {values = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf6E3M2FN>} : () -> tensor<4xf6E3M2FN>
    return %0 : tensor<4xf6E3M2FN>
}

// -----

// CHECK-LABEL: test_cast_f4e2m1
func.func @test_cast_f4e2m1(%arg0: tensor<13x21x3xf4E2M1FN>) -> tensor<13x21x3xbf16> {
  %0 = tosa.cast %arg0 : (tensor<13x21x3xf4E2M1FN>) -> tensor<13x21x3xbf16>
  return %0 : tensor<13x21x3xbf16>
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
