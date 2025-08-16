// RUN: mlir-opt %s --verify-each | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s


// -----
// CHECK-LABEL: argmax
func.func @test_argmax(%arg0: tensor<14x19xf32>) -> tensor<14xi32> {
  %0 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<14x19xf32>) -> tensor<14xi32>
  return %0 : tensor<14xi32>
}

// -----
// CHECK-LABEL: avg_pool2d_f32
func.func @test_avg_pool2d_f32(%arg0: tensor<1x7x7x9xf32>) -> tensor<1x7x7x9xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %output_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.avg_pool2d %arg0, %input_zp, %output_zp {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x7x7x9xf32>
  return %0 : tensor<1x7x7x9xf32>
}

// -----
// CHECK-LABEL: avg_pool2d_f16
func.func @test_avg_pool2d_f16(%arg0: tensor<1x7x7x9xf16>) -> tensor<1x7x7x9xf16> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %output_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %0 = tosa.avg_pool2d %arg0, %input_zp, %output_zp {acc_type = f16, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x7x7x9xf16>
  return %0 : tensor<1x7x7x9xf16>
}

// -----
// CHECK-LABEL: avg_pool2d_f16_accumf32
func.func @test_avg_pool2d_f16_accumf32(%arg0: tensor<1x7x7x9xf16>) -> tensor<1x7x7x9xf16> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %output_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %0 = tosa.avg_pool2d %arg0, %input_zp, %output_zp {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x7x7x9xf16>
  return %0 : tensor<1x7x7x9xf16>
}

// -----
// CHECK-LABEL: avg_pool2d_i8
func.func @test_avg_pool2d_i8(%arg0: tensor<1x7x7x9xi8>) -> tensor<1x7x7x9xi8> {
  %input_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %output_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.avg_pool2d %arg0, %input_zp, %output_zp {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x7x7x9xi8>
  return %0 : tensor<1x7x7x9xi8>
}

// -----
// CHECK-LABEL: avg_pool2d_i16
func.func @test_avg_pool2d_i16(%arg0: tensor<1x7x7x9xi16>) -> tensor<1x7x7x9xi16> {
  %input_zp = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
  %output_zp = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
  %0 = tosa.avg_pool2d %arg0, %input_zp, %output_zp {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<1x7x7x9xi16>
  return %0 : tensor<1x7x7x9xi16>
}

// -----
// CHECK-LABEL: avg_pool2d_q8
func.func @test_avg_pool2d_q8(%arg0: tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>) -> tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>> {
  %input_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %output_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.avg_pool2d %arg0, %input_zp, %output_zp {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>
  return %0 : tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>
}

// -----
// CHECK-LABEL: conv2d
func.func @test_conv2d(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>, %arg2: tensor<8xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x4x4x8xf32> {
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = true} : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----
// CHECK-LABEL: conv2d_unranked_input
func.func @test_conv2d_unranked_input(%arg0: tensor<*xf32>, %arg1: tensor<8x1x1x4xf32>, %arg2: tensor<8xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x4x4x8xf32> {
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = true} : (tensor<*xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----
// CHECK-LABEL: conv2d_quant_uniform
func.func @test_conv2d_quant_uniform(%arg0: tensor<1x4x4x4x!quant.uniform<i8:f32, 0.01>>, %arg1: tensor<8x1x1x4x!quant.uniform<i8:f32, 0.01>>, %arg2: tensor<8x!quant.uniform<i8:f32, 0.01>>) -> tensor<1x4x4x8x!quant.uniform<i32:f32, 0.01>> {
  %zp = "tosa.const" () { values = dense<0> : tensor<1xi8> } : () -> tensor<1xi8>
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %zp, %zp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = true} : (tensor<1x4x4x4x!quant.uniform<i8:f32, 0.01>>, tensor<8x1x1x4x!quant.uniform<i8:f32, 0.01>>, tensor<8x!quant.uniform<i8:f32, 0.01>>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x4x8x!quant.uniform<i32:f32, 0.01>>
  return %0 : tensor<1x4x4x8x!quant.uniform<i32:f32, 0.01>>
}

// -----
// CHECK-LABEL: conv2d_quant_any
func.func @test_conv2d_quant_any(%arg0: tensor<1x4x4x4x!quant.any<i8<-8:7>>>, %arg1: tensor<8x1x1x4x!quant.any<i8<-8:7>>>, %arg2: tensor<8x!quant.any<i8<-8:7>>>) -> tensor<1x4x4x8x!quant.any<i32<-8:7>>> {
  %zp = "tosa.const" () { values = dense<0> : tensor<1xi8> } : () -> tensor<1xi8>
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %zp, %zp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = true} : (tensor<1x4x4x4x!quant.any<i8<-8:7>>>, tensor<8x1x1x4x!quant.any<i8<-8:7>>>, tensor<8x!quant.any<i8<-8:7>>>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x4x8x!quant.any<i32<-8:7>>>
  return %0 : tensor<1x4x4x8x!quant.any<i32<-8:7>>>
}

// -----
// CHECK-LABEL: conv2d_q8xi4
func.func @test_conv2d_q8xi4(%arg0: tensor<1x11x11x3xi8>) -> tensor<1x1x1x3xi8> {
  %0 = "tosa.const"() {values = dense<0> : tensor<3x11x11x3xi4>} : () -> tensor<3x11x11x3xi4>
  %1 = "tosa.const"() {values = dense<[12, 23, 55]> : tensor<3xi32>} : () -> tensor<3xi32>
  %izp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %wzp = "tosa.const"() {values = dense<0> : tensor<1xi4>} : () -> tensor<1xi4>
  %2 = "tosa.conv2d"(%arg0, %0, %1, %izp, %wzp) {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x11x11x3xi8>, tensor<3x11x11x3xi4>, tensor<3xi32>, tensor<1xi8>, tensor<1xi4>) -> tensor<1x1x1x3xi32>
  %multiplier = "tosa.const"() {values = dense<[2026291432, 1079222024, 1693132724]> : tensor<3xi32>} : () -> tensor<3xi32>
  %shift = "tosa.const"() {values = dense<[37, 36, 37]> : tensor<3xi8>} : () -> tensor<3xi8>
  %rescale_input_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %rescale_output_zp = "tosa.const"() <{values = dense<27> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = tosa.rescale %2, %multiplier, %shift, %rescale_input_zp, %rescale_output_zp {rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>, scale32 = true, per_channel = true, input_unsigned = false, output_unsigned = false} : (tensor<1x1x1x3xi32>, tensor<3xi32>, tensor<3xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x1x3xi8>
  return %3 : tensor<1x1x1x3xi8>
}

// -----
// CHECK-LABEL: conv3d
func.func @test_conv3d(%arg0: tensor<1x4x8x21x17xf32>, %arg1: tensor<34x1x1x1x17xf32>, %arg2: tensor<34xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x4x8x21x34xf32> {
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<1x4x8x21x17xf32>, tensor<34x1x1x1x17xf32>, tensor<34xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x8x21x34xf32>
  return %0 : tensor<1x4x8x21x34xf32>
}

// -----
// CHECK-LABEL: conv3d_with_local_bound
func.func @test_conv3d_with_local_bound(%arg0: tensor<1x4x8x21x17xf32>, %arg1: tensor<34x1x1x1x17xf32>, %arg2: tensor<34xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x4x8x21x34xf32> {
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>, local_bound = true} : (tensor<1x4x8x21x17xf32>, tensor<34x1x1x1x17xf32>, tensor<34xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x8x21x34xf32>
  return %0 : tensor<1x4x8x21x34xf32>
}

// -----
// CHECK-LABEL: depthwise_conv2d
func.func @test_depthwise_conv2d(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<1x1x4x2xf32>, %arg2: tensor<8xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x4x4x8xf32> {
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xf32>, tensor<1x1x4x2xf32>, tensor<8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----
// CHECK-LABEL: depthwise_conv2d_with_local_bound
func.func @test_depthwise_conv2d_with_local_bound(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<1x1x4x2xf32>, %arg2: tensor<8xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x4x4x8xf32> {
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = true} : (tensor<1x4x4x4xf32>, tensor<1x1x4x2xf32>, tensor<8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----
// CHECK-LABEL: fft2d
func.func @test_fft2d(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
  %0, %1 = tosa.fft2d %arg0, %arg1 {inverse = false} : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>)
  return %0, %1 : tensor<1x4x8xf32>, tensor<1x4x8xf32>
}

// -----
// CHECK-LABEL: fft2d_with_local_bound
func.func @test_fft2d_with_local_bound(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
  %0, %1 = tosa.fft2d %arg0, %arg1 {inverse = false, local_bound = true} : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>)
  return %0, %1 : tensor<1x4x8xf32>, tensor<1x4x8xf32>
}

// -----
// CHECK-LABEL: test_matmul
func.func @test_matmul(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>) -> tensor<1x14x28xf32> {
%azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
%bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
%0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x14x19xf32>, tensor<1x19x28xf32>, tensor<1xf32>, tensor<1xf32>)  -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----
// CHECK-LABEL: max_pool2d_f32
func.func @test_max_pool2d_f32(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----
// CHECK-LABEL: max_pool2d_bf16
func.func @test_max_pool2d_bf16(%arg0: tensor<1x32x32x8xbf16>) -> tensor<1x32x32x8xbf16> {
  %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xbf16>) -> tensor<1x32x32x8xbf16>
  return %0 : tensor<1x32x32x8xbf16>
}

// -----
// CHECK-LABEL: max_pool2d_f16
func.func @test_max_pool2d_f16(%arg0: tensor<1x32x32x8xf16>) -> tensor<1x32x32x8xf16> {
  %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf16>) -> tensor<1x32x32x8xf16>
  return %0 : tensor<1x32x32x8xf16>
}

// -----
// CHECK-LABEL: rfft2d
func.func @test_rfft2d(%arg0: tensor<13x8x16xf32>) -> (tensor<13x8x9xf32>, tensor<13x8x9xf32>) {
  %0, %1 = tosa.rfft2d %arg0 : (tensor<13x8x16xf32>) -> (tensor<13x8x9xf32>, tensor<13x8x9xf32>)
  return %0, %1 : tensor<13x8x9xf32>, tensor<13x8x9xf32>
}

// -----
// CHECK-LABEL: rfft2d_width1
func.func @test_rfft2d_width1(%arg0: tensor<1x1x1xf32>) -> (tensor<1x1x1xf32>, tensor<1x1x1xf32>) {
  %0, %1 = tosa.rfft2d %arg0 : (tensor<1x1x1xf32>) -> (tensor<1x1x1xf32>, tensor<1x1x1xf32>)
  return %0, %1 : tensor<1x1x1xf32>, tensor<1x1x1xf32>
}

// -----
// CHECK-LABEL: rfft2d_with_local_bound
func.func @test_rfft2d_with_local_bound(%arg0: tensor<13x8x16xf32>) -> (tensor<13x8x9xf32>, tensor<13x8x9xf32>) {
  %0, %1 = tosa.rfft2d %arg0 {local_bound = true} : (tensor<13x8x16xf32>) -> (tensor<13x8x9xf32>, tensor<13x8x9xf32>)
  return %0, %1 : tensor<13x8x9xf32>, tensor<13x8x9xf32>
}

// -----
// CHECK-LABEL: transpose_conv2d
func.func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----
// CHECK-LABEL: transpose_conv2d_unranked_input
func.func @test_transpose_conv2d_unranked_input(%arg0: tensor<*xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} : (tensor<*xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----
// CHECK-LABEL: transpose_conv2d_unranked_weight
func.func @test_transpose_conv2d_unranked_weight(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<*xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>, tensor<*xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----
// CHECK-LABEL: transpose_conv2d_with_local_bound
func.func @test_transpose_conv2d_with_local_bound(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x32x32x16xf32> {
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = false} : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----
// CHECK-LABEL: clamp
func.func @test_clamp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.clamp %arg0 {min_val = 0.0 : f32, max_val = 1.0: f32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: clamp_propagate
func.func @test_clamp_propagate(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.clamp %arg0 {min_val = 0.0 : f32, max_val = 1.0: f32, nan_mode = #tosa.nan_mode<PROPAGATE>} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: clamp_ignore
func.func @test_clamp_ignore(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.clamp %arg0 {min_val = 0.0 : f32, max_val = 1.0: f32, nan_mode = #tosa.nan_mode<IGNORE>} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: clamp_f16
func.func @test_clamp_f16(%arg0: tensor<13x21x3xf16>) -> tensor<13x21x3xf16> {
  %0 = tosa.clamp %arg0 {min_val = 0.0 : f16, max_val = 1.0: f16} : (tensor<13x21x3xf16>) -> tensor<13x21x3xf16>
  return %0 : tensor<13x21x3xf16>
}

// -----
// CHECK-LABEL: clamp_bf16
func.func @test_clamp_bf16(%arg0: tensor<13x21x3xbf16>) -> tensor<13x21x3xbf16> {
  %0 = tosa.clamp %arg0 {min_val = 0.0 : bf16, max_val = 1.0: bf16} : (tensor<13x21x3xbf16>) -> tensor<13x21x3xbf16>
  return %0 : tensor<13x21x3xbf16>
}

// -----
// CHECK-LABEL: clamp_quantized
func.func @test_clamp_quantized(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 1.000000e-01:-127>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 1.000000e-01:-127>> {
  %0 = tosa.clamp %arg0 {min_val = 0 : i8, max_val = 1 : i8} : (tensor<13x21x3x!quant.uniform<i8:f32, 1.000000e-01:-127>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 1.000000e-01:-127>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 1.000000e-01:-127>>
}

// -----
// CHECK-LABEL: sigmoid
func.func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.sigmoid %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: tanh
func.func @test_tanh(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.tanh %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: erf
func.func @test_erf(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.erf %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: add
func.func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: arithmetic_right_shift
func.func @test_arithmetic_right_shift(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: bitwise_and
func.func @test_bitwise_and(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x3xi32> {
  %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: bitwise_or
func.func @test_bitwise_or(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>) -> tensor<13x21x3xi32> {
  %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<13x21x3xi32>, tensor<13x1x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: bitwise_xor
func.func @test_bitwise_xor(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: intdiv
func.func @test_intdiv(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  %0 = tosa.intdiv %arg0, %arg1 : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: logical_and
func.func @test_logical_and(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<13x21x3xi1> {
  %0 = tosa.logical_and %arg0, %arg1 : (tensor<13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
// CHECK-LABEL: logical_left_shift
func.func @test_logical_left_shift(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x3xi32> {
  %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: logical_right_shift
func.func @test_logical_right_shift(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x3xi32> {
  %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: logical_or
func.func @test_logical_or(%arg0: tensor<13x1x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  %0 = tosa.logical_or %arg0, %arg1 : (tensor<13x1x3xi1>, tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
// CHECK-LABEL: logical_xor
func.func @test_logical_xor(%arg0: tensor<13x1x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  %0 = tosa.logical_xor %arg0, %arg1 : (tensor<13x1x3xi1>, tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
// CHECK-LABEL: maximum
func.func @test_max(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.maximum %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: minimum
func.func @test_min(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.minimum %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: test_mul_scalar_with_unranked_output
func.func @test_mul_scalar_with_unranked_output(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<*xf32> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<f32>, tensor<f32>, tensor<1xi8>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----
// CHECK-LABEL: test_mul_scalar
func.func @test_mul_scalar(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<f32>, tensor<f32>, tensor<1xi8>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----
// CHECK-LABEL: test_mul
func.func @test_mul(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xf32> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xf32>, tensor<13x1x3xf32>, tensor<1xi8>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: i32_mul
func.func @test_i32_mul(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>) -> tensor<13x21x3xi32> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xi32>, tensor<13x1x3xi32>, tensor<1xi8>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: mul
func.func @test_mul_relaxed_result_type(%arg0: tensor<13x21x3xi16>, %arg1: tensor<13x1x3xi16>) -> tensor<13x21x3xi16> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xi16>, tensor<13x1x3xi16>, tensor<1xi8>) -> tensor<13x21x3xi16>
  return %0 : tensor<13x21x3xi16>
}

// -----
// CHECK-LABEL: test_mul_unranked_b
func.func @test_mul_unranked_b(%arg0: tensor<13x21x3xf32>, %arg1: tensor<*xf32>) -> tensor<13x21x3xf32> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xf32>, tensor<*xf32>, tensor<1xi8>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: test_mul_unranked_a_and_b
func.func @test_mul_unranked_a_and_b(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<13x21x3xf32> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<*xf32>, tensor<*xf32>, tensor<1xi8>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: pow
func.func @test_pow(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.pow %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: sub
func.func @test_sub(%arg0: tensor<1x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.sub %arg0, %arg1 : (tensor<1x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: table
func.func @main(%arg0: tensor<64xi32>, %arg1: tensor<513x!quant.uniform<i16:f32, 1.0:0>>) -> tensor<64x!quant.uniform<i16:f32, 1.0:0>> {
    %0 = tosa.table %arg0, %arg1 : (tensor<64xi32>, tensor<513x!quant.uniform<i16:f32, 1.000000e+00>>) -> tensor<64x!quant.uniform<i16:f32, 1.000000e+00>>
    return %0 : tensor<64x!quant.uniform<i16:f32, 1.0:0>>
}

// -----
// CHECK-LABEL: abs
func.func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.abs %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: bitwise_not
func.func @test_bitwise_not(%arg0: tensor<13x21x1xi32>) -> tensor<13x21x1xi32> {
  %0 = tosa.bitwise_not %arg0 : (tensor<13x21x1xi32>) -> tensor<13x21x1xi32>
  return %0 : tensor<13x21x1xi32>
}

// -----
// CHECK-LABEL: ceil
func.func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.ceil %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: clz
func.func @test_clz(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  %0 = tosa.clz %arg0 : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: cos
func.func @test_cos(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.cos %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: exp
func.func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.exp %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: floor
func.func @test_floor(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.floor %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: log
func.func @test_log(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.log %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: logical_not
func.func @test_logical_not(%arg0: tensor<1x21x3xi1>) -> tensor<1x21x3xi1> {
  %0 = tosa.logical_not %arg0 : (tensor<1x21x3xi1>) -> tensor<1x21x3xi1>
  return %0 : tensor<1x21x3xi1>
}

// -----
// CHECK-LABEL: negate
func.func @test_negate(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.negate %arg0, %arg1, %arg2 : (tensor<13x21x3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: reciprocal
func.func @test_reciprocal(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.reciprocal %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: rsqrt
func.func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.rsqrt %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: sin
func.func @test_sin(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.sin %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: select
func.func @test_select(%arg0: tensor<1x1x1xi1>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<1x1x1xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}


// -----
// CHECK-LABEL: equal
func.func @test_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xi1> {
  %0 = tosa.equal %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
// CHECK-LABEL: greater
func.func @test_greater(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %0 = tosa.greater %arg0, %arg1 : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
// CHECK-LABEL: greater_equal
func.func @test_greater_equal(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %0 = tosa.greater_equal %arg0, %arg1 : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
// CHECK-LABEL: reduce_all
func.func @test_reduce_all(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
  %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<13x21x3xi1>) -> tensor<1x21x3xi1>
  %2 = tosa.const_shape {values = dense<[21, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.reshape %0, %2 : (tensor<1x21x3xi1>, !tosa.shape<2>) -> tensor<21x3xi1>
  return %1 : tensor<21x3xi1>
}

// -----
// CHECK-LABEL: reduce_any
func.func @test_reduce_any(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
  %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<13x21x3xi1>) -> tensor<1x21x3xi1>
  %2 = tosa.const_shape {values = dense<[21, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.reshape %0, %2 : (tensor<1x21x3xi1>, !tosa.shape<2>) -> tensor<21x3xi1>
  return %1 : tensor<21x3xi1>
}

// -----
// CHECK-LABEL: reduce_max
func.func @test_reduce_max(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
  %2 = tosa.const_shape {values = dense<[21, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.reshape %0, %2 : (tensor<1x21x3xf32>, !tosa.shape<2>) -> tensor<21x3xf32>
  return %1 : tensor<21x3xf32>
}

// -----
// CHECK-LABEL: reduce_min
func.func @test_reduce_min(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
  %2 = tosa.const_shape {values = dense<[21, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.reshape %0, %2 : (tensor<1x21x3xf32>, !tosa.shape<2>) -> tensor<21x3xf32>
  return %1 : tensor<21x3xf32>
}

// -----
// CHECK-LABEL: reduce_product
func.func @test_reduce_product(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
  %2 = tosa.const_shape {values = dense<[21, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.reshape %0, %2 : (tensor<1x21x3xf32>, !tosa.shape<2>) -> tensor<21x3xf32>
  return %1 : tensor<21x3xf32>
}

// -----
// CHECK-LABEL: reduce_sum
func.func @test_reduce_sum(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
  %2 = tosa.const_shape {values = dense<[21, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.reshape %0, %2 : (tensor<1x21x3xf32>, !tosa.shape<2>) -> tensor<21x3xf32>
  return %1 : tensor<21x3xf32>
}

// -----
// CHECK-LABEL: concat
func.func @test_concat(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<26x21x3xf32> {
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<26x21x3xf32>
  return %0 : tensor<26x21x3xf32>
}

// -----
// CHECK-LABEL: pad_explicit_value
func.func @test_pad_explicit_value(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.const"() {values = dense<3.14> : tensor<1xf32>} : () -> tensor<1xf32>
  %padding = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
  %1 = tosa.pad %arg0, %padding, %0 : (tensor<13x21x3xf32>, !tosa.shape<6>, tensor<1xf32>) -> tensor<13x21x3xf32>
  return %1 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: reshape
func.func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x819xf32> {
  %1 = tosa.const_shape {values = dense<[1, 819]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %0 = tosa.reshape %arg0, %1 : (tensor<13x21x3xf32>, !tosa.shape<2>) -> tensor<1x819xf32>
  return %0 : tensor<1x819xf32>
}

// -----
// CHECK-LABEL: reshape_unranked_output
func.func @test_reshape_unranked_output(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %1 = tosa.const_shape {values = dense<[21, 13, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %0 = tosa.reshape %arg0, %1 : (tensor<13x21x3xf32>, !tosa.shape<3>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----
// CHECK-LABEL: reverse
func.func @test_reverse(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: slice
func.func @test_slice(%arg0: tensor<13x21x3xf32>) -> tensor<4x11x1xf32> {
  %0 = tosa.const_shape {values = dense<[4, 11, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %1 = tosa.const_shape {values = dense<[6, 8, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %2 = tosa.slice %arg0, %0, %1 : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xf32>
  return %2 : tensor<4x11x1xf32>
}

// -----
// CHECK-LABEL: slice_size
func.func @test_slice_size(%arg0: tensor<13x21x3xf32>) -> tensor<7x11x1xf32> {
  %0 = tosa.const_shape {values = dense<[-1, 11, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %1 = tosa.const_shape {values = dense<[6, 8, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %2 = tosa.slice %arg0, %0, %1 : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<7x11x1xf32>
  return %2 : tensor<7x11x1xf32>
}

// -----
// CHECK-LABEL: tile
func.func @test_tile(%arg0: tensor<13x21x3xf32>) -> tensor<39x21x6xf32> {
  %cst = tosa.const_shape { values = dense<[3, 1, 2]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = tosa.tile %arg0, %cst: (tensor<13x21x3xf32>, !tosa.shape<3>) -> tensor<39x21x6xf32>
  return %0 : tensor<39x21x6xf32>
}

// -----
// CHECK-LABEL: transpose
func.func @test_transpose(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21xf32> {
  %1 = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>} : (tensor<13x21x3xf32>) -> tensor<3x13x21xf32>
  return %1 : tensor<3x13x21xf32>
}

// -----
// CHECK-LABEL: transpose_dynamic_dim
func.func @test_transpose_dynamic_dim(%arg0: tensor<13x?x3xf32>) -> tensor<3x13x?xf32> {
  %1 = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>} : (tensor<13x?x3xf32>) -> tensor<3x13x?xf32>
  return %1 : tensor<3x13x?xf32>
}

// -----
// CHECK-LABEL: transpose_half_dynamic_dim
func.func @test_transpose_half_dynamic_dim(%arg0: tensor<13x3x3xf32>) -> tensor<3x13x?xf32> {
  %1 = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>} : (tensor<13x3x3xf32>) -> tensor<3x13x?xf32>
  return %1 : tensor<3x13x?xf32>
}

// -----
// CHECK-LABEL: gather
func.func @test_gather(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x26xi32>) -> tensor<13x26x3xf32> {
  %0 = tosa.gather %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x26xi32>) -> tensor<13x26x3xf32>
  return %0 : tensor<13x26x3xf32>
}

// -----
// CHECK-LABEL: scatter
func.func @test_scatter(%arg0: tensor<13x52x3xf32>, %arg1: tensor<13x26xi32>, %arg2: tensor<13x26x3xf32>) -> tensor<13x52x3xf32> {
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x52x3xf32>, tensor<13x26xi32>, tensor<13x26x3xf32>) -> tensor<13x52x3xf32>
  return %0 : tensor<13x52x3xf32>
}

// -----
// CHECK-LABEL: gather_unranked_indices
func.func @test_gather_unranked_indices(%arg0: tensor<13x21x3xf32>, %arg1: tensor<*xi32>) -> tensor<13x26x3xf32> {
  %0 = tosa.gather %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<*xi32>) -> tensor<13x26x3xf32>
  return %0 : tensor<13x26x3xf32>
}

// -----
// CHECK-LABEL: scatter_unranked_indices
func.func @test_scatter_unranked_indices(%arg0: tensor<13x21x3xf32>, %arg1: tensor<*xi32>, %arg2: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x21x3xf32>, tensor<*xi32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: resize
func.func @test_resize(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x64x64x8xf32> {
  %scale = tosa.const_shape { values = dense<[4, 2, 4, 2]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<[-1, -1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<[1, 1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %1 = tosa.resize %arg0, %scale, %offset, %border { mode = #tosa.resize_mode<BILINEAR> } : (tensor<1x32x32x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x64x64x8xf32>
  return %1 : tensor<1x64x64x8xf32>
}

// -----
// CHECK-LABEL: resize_unranked_output
func.func @test_resize_unranked_output(%arg0: tensor<1x32x32x8xf32>) -> tensor<*xf32> {
  %scale = tosa.const_shape { values = dense<[4, 2, 4, 2]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<[-1, -1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<[1, 1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %1 = tosa.resize %arg0, %scale, %offset, %border { mode = #tosa.resize_mode<BILINEAR> } : (tensor<1x32x32x8xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}

// -----
// CHECK-LABEL: resize_unranked_input
func.func @test_resize_unranked_input(%arg0: tensor<*xf32>) -> tensor<1x64x64x8xf32> {
  %scale = tosa.const_shape { values = dense<[4, 2, 4, 2]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<[-1, -1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<[1, 1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %1 = tosa.resize %arg0, %scale, %offset, %border { mode = #tosa.resize_mode<BILINEAR> } : (tensor<*xf32>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x64x64x8xf32>
  return %1 : tensor<1x64x64x8xf32>
}

// -----
// CHECK-LABEL: cast
func.func @test_cast1(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xf32> {
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: cast2
func.func @test_cast2(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3x!quant.uniform<u8:f32, 0.078431375324726104:128>> {
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi32>) -> tensor<13x21x3x!quant.uniform<u8:f32, 0.078431375324726104:128>>
  return %0 : tensor<13x21x3x!quant.uniform<u8:f32, 0.078431375324726104:128>>
}

// -----
// CHECK-LABEL: cast3
func.func @test_cast3(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3x!quant.uniform<i16:f32, 0.078431375324726104:128>> {
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi32>) -> tensor<13x21x3x!quant.uniform<i16:f32, 0.078431375324726104:128>>
  return %0 : tensor<13x21x3x!quant.uniform<i16:f32, 0.078431375324726104:128>>
}

// -----
// CHECK-LABEL: rescale
func.func @test_rescale(%arg0: tensor<13x21x3x!quant.uniform<u8:f32, 0.015655439347028732:127>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>> {
   %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32> } : () -> tensor<1xi32>
   %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
   %input_zp = "tosa.const"() <{values = dense<127> : tensor<1xi8>}> : () -> tensor<1xi8>
   %output_zp = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
   %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true, per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<13x21x3x!quant.uniform<u8:f32, 0.015655439347028732:127>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>
    return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>
}

// -----
// CHECK-LABEL: rescale_i16_zp32768
func.func @test_rescale_i16_zp32768(%arg0 : tensor<2xi8>) -> tensor<2xi16> {
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16>} : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<15> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<17> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<32768> : tensor<1xi16>} : () -> tensor<1xi16>
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = true} : (tensor<2xi8>, tensor<1xi16>, tensor<1xi8>, tensor<1xi8>, tensor<1xi16>) -> tensor<2xi16>
  return %0 : tensor<2xi16>
}

// -----
// CHECK-LABEL: const
func.func @test_const(%arg0 : index) -> tensor<4xi32> {
    %0 = "tosa.const"() {values = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: identity
func.func @test_identity(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  %0 = tosa.identity %arg0 : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: cond_if
func.func @test_cond_if(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> tensor<f32> {
  %0 = tosa.cond_if %arg2 : tensor<i1> -> tensor<f32> {
    %1 = tosa.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tosa.yield %1 : tensor<f32>
  } else {
    %1 = tosa.sub %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tosa.yield %1 : tensor<f32>
  }
  return %0 : tensor<f32>
}

// -----
// CHECK-LABEL: while_loop
func.func @test_while_loop(%arg0: tensor<10xi32>, %arg1: tensor<i32>) {
  %0 = "tosa.const"() {values = dense<0> : tensor<i32>} : () -> tensor<i32>
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
// CHECK-LABEL: custom
func.func @test_custom(%arg0: tensor<10xi32>) -> tensor<10xi32> {
  %0 = tosa.custom %arg0 {operator_name="custom_test", domain_name="tosa.mlir_test", implementation_attrs="" } : (tensor<10xi32>) -> (tensor<10xi32>)
  return %0 : tensor<10xi32>
}

// -----
// CHECK-LABEL: const_shape
func.func @test_const_shape() -> !tosa.shape<4> {
  %cst = tosa.const_shape {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
  return %cst : !tosa.shape<4>
}

// F8 support tests

// -----
// CHECK-LABEL: argmax_f8E5M2
func.func @test_argmax_f8E5M2(%arg0: tensor<12x8x16xf8E5M2>) -> tensor<12x16xi32> {
  %0 = tosa.argmax %arg0 { axis = 1 : i32 } : (tensor<12x8x16xf8E5M2>) -> tensor<12x16xi32>
  return %0 : tensor<12x16xi32>
}

// -----
// CHECK-LABEL: avg_pool2d_f8E5M2
func.func @test_avg_pool2d_f8E5M2(%arg0: tensor<1x7x7x9xf8E5M2>) -> tensor<1x7x7x9xf8E5M2> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %output_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %0 = tosa.avg_pool2d %arg0, %input_zp, %output_zp {acc_type = f16, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xf8E5M2>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x7x7x9xf8E5M2>
  return %0 : tensor<1x7x7x9xf8E5M2>
}

// -----
// CHECK-LABEL: conv2d_f8E5M2
func.func @test_conv2d_f8E5M2(%arg0: tensor<1x4x4x4xf8E5M2>, %arg1: tensor<8x1x1x4xf8E5M2>, %arg2: tensor<8xf16>) -> tensor<1x4x4x8xf16> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f16, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = true} : (tensor<1x4x4x4xf8E5M2>, tensor<8x1x1x4xf8E5M2>, tensor<8xf16>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x4x8xf16>
  return %0 : tensor<1x4x4x8xf16>
}

// -----
// CHECK-LABEL: conv3d_f8E5M2
func.func @test_conv3d_f8E5M2(%arg0: tensor<1x4x8x21x17xf8E5M2>, %arg1: tensor<34x1x1x1x17xf8E5M2>, %arg2: tensor<34xf16>, %arg3: tensor<1xf8E5M2>, %arg4: tensor<1xf8E5M2>) -> tensor<1x4x8x21x34xf16> {
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f16, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<1x4x8x21x17xf8E5M2>, tensor<34x1x1x1x17xf8E5M2>, tensor<34xf16>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x8x21x34xf16>
  return %0 : tensor<1x4x8x21x34xf16>
}

// -----
// CHECK-LABEL: depthwise_conv2d_f8E5M2
func.func @test_depthwise_conv2d_f8E5M2(%arg0: tensor<1x4x4x4xf8E5M2>, %arg1: tensor<1x1x4x2xf8E5M2>, %arg2: tensor<8xf16>, %arg3: tensor<1xf8E5M2>, %arg4: tensor<1xf8E5M2>) -> tensor<1x4x4x8xf16> {
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f16, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xf8E5M2>, tensor<1x1x4x2xf8E5M2>, tensor<8xf16>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x4x8xf16>
  return %0 : tensor<1x4x4x8xf16>
}

// -----
// CHECK-LABEL: test_matmul_f8E5M2
func.func @test_matmul_f8E5M2(%arg0: tensor<1x14x19xf8E5M2>, %arg1: tensor<1x19x28xf8E5M2>) -> tensor<1x14x28xf16> {
  %azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E5M2>}> : () -> tensor<1xf8E5M2>
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x14x19xf8E5M2>, tensor<1x19x28xf8E5M2>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x14x28xf16>
  return %0 : tensor<1x14x28xf16>
}

// -----
// CHECK-LABEL: max_pool2d_f8E5M2
func.func @test_max_pool2d_f8E5M2(%arg0: tensor<1x32x32x8xf8E5M2>) -> tensor<1x32x32x8xf8E5M2> {
  %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf8E5M2>) -> tensor<1x32x32x8xf8E5M2>
  return %0 : tensor<1x32x32x8xf8E5M2>
}

// -----
// CHECK-LABEL: transpose_conv2d_f8E5M2
func.func @test_transpose_conv2d_f8E5M2(%arg0: tensor<1x32x32x8xf8E5M2>, %arg1: tensor<16x1x1x8xf8E5M2>, %arg2: tensor<16xf16>, %arg3: tensor<1xf8E5M2>, %arg4: tensor<1xf8E5M2>) -> tensor<1x32x32x16xf16> {
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f16, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf8E5M2>, tensor<16x1x1x8xf8E5M2>, tensor<16xf16>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x32x32x16xf16>
  return %0 : tensor<1x32x32x16xf16>
}

// -----
// CHECK-LABEL: const_f8E5M2
func.func @test_const_f8E5M2(%arg0 : index) -> tensor<4xf8E5M2> {
    %0 = "tosa.const"() {values = dense<[3.0, -0.0, -1.0, 2.0]> : tensor<4xf8E5M2>} : () -> tensor<4xf8E5M2>
    return %0 : tensor<4xf8E5M2>
}

// -----
// CHECK-LABEL: cast_f8E5M2
func.func @test_cast_f8E5M2(%arg0: tensor<13x21x3xf8E5M2>) -> tensor<13x21x3xf16> {
  %0 = tosa.cast %arg0 : (tensor<13x21x3xf8E5M2>) -> tensor<13x21x3xf16>
  return %0 : tensor<13x21x3xf16>
}

// -----
// CHECK-LABEL: concat_f8E5M2
func.func @test_concat_f8E5M2(%arg0: tensor<13x21x3xf8E5M2>, %arg1: tensor<13x21x3xf8E5M2>) -> tensor<26x21x3xf8E5M2> {
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13x21x3xf8E5M2>, tensor<13x21x3xf8E5M2>) -> tensor<26x21x3xf8E5M2>
  return %0 : tensor<26x21x3xf8E5M2>
}

// -----
// CHECK-LABEL: pad_f8E5M2
func.func @test_pad_f8E5M2(%arg0: tensor<13x21x3xf8E5M2>) -> tensor<13x21x3xf8E5M2> {
  %padding = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
  %cst = "tosa.const"() { values = dense<-0.0> : tensor<1xf8E5M2> } : () -> tensor<1xf8E5M2>
  %0 = tosa.pad %arg0, %padding, %cst : (tensor<13x21x3xf8E5M2>, !tosa.shape<6>, tensor<1xf8E5M2>) -> tensor<13x21x3xf8E5M2>
  return %0 : tensor<13x21x3xf8E5M2>
}

// -----
// CHECK-LABEL: reshape_f8E5M2
func.func @test_reshape_f8E5M2(%arg0: tensor<13x21x3xf8E5M2>) -> tensor<1x819xf8E5M2> {
  %1 = tosa.const_shape {values = dense<[1, 819]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %0 = tosa.reshape %arg0, %1 : (tensor<13x21x3xf8E5M2>, !tosa.shape<2>) -> tensor<1x819xf8E5M2>
  return %0 : tensor<1x819xf8E5M2>
}

// -----
// CHECK-LABEL: reverse_f8E5M2
func.func @test_reverse_f8E5M2(%arg0: tensor<13x21x3xf8E5M2>) -> tensor<13x21x3xf8E5M2> {
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<13x21x3xf8E5M2>) -> tensor<13x21x3xf8E5M2>
  return %0 : tensor<13x21x3xf8E5M2>
}

// -----
// CHECK-LABEL: slice_f8E5M2
func.func @test_slice_f8E5M2(%arg0: tensor<13x21x3xf8E5M2>) -> tensor<4x11x1xf8E5M2> {
  %0 = tosa.const_shape {values = dense<[4, 11, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %1 = tosa.const_shape {values = dense<[6, 8, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %2 = tosa.slice %arg0, %0, %1 : (tensor<13x21x3xf8E5M2>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xf8E5M2>
  return %2 : tensor<4x11x1xf8E5M2>
}

// -----
// CHECK-LABEL: tile_f8E5M2
func.func @test_tile_f8E5M2(%arg0: tensor<13x21x3xf8E5M2>) -> tensor<39x21x6xf8E5M2> {
  %cst = tosa.const_shape { values = dense<[3, 1, 2]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = tosa.tile %arg0, %cst: (tensor<13x21x3xf8E5M2>, !tosa.shape<3>) -> tensor<39x21x6xf8E5M2>
  return %0 : tensor<39x21x6xf8E5M2>
}

// -----
// CHECK-LABEL: transpose_f8E5M2
func.func @test_transpose_f8E5M2(%arg0: tensor<13x21x3xf8E5M2>) -> tensor<3x13x21xf8E5M2> {
  %1 = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>} : (tensor<13x21x3xf8E5M2>) -> tensor<3x13x21xf8E5M2>
  return %1 : tensor<3x13x21xf8E5M2>
}

// -----
// CHECK-LABEL: gather_f8E5M2
func.func @test_gather_f8E5M2(%arg0: tensor<13x21x3xf8E5M2>, %arg1: tensor<13x26xi32>) -> tensor<13x26x3xf8E5M2> {
  %0 = tosa.gather %arg0, %arg1 : (tensor<13x21x3xf8E5M2>, tensor<13x26xi32>) -> tensor<13x26x3xf8E5M2>
  return %0 : tensor<13x26x3xf8E5M2>
}

// -----
// CHECK-LABEL: scatter_f8E5M2
func.func @test_scatter_f8E5M2(%arg0: tensor<13x52x3xf8E5M2>, %arg1: tensor<13x26xi32>, %arg2: tensor<13x26x3xf8E5M2>) -> tensor<13x52x3xf8E5M2> {
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x52x3xf8E5M2>, tensor<13x26xi32>, tensor<13x26x3xf8E5M2>) -> tensor<13x52x3xf8E5M2>
  return %0 : tensor<13x52x3xf8E5M2>
}

// -----
// CHECK-LABEL: argmax_f8E4M3FN
func.func @test_argmax_f8E4M3FN(%arg0: tensor<12x8x16xf8E4M3FN>) -> tensor<12x16xi32> {
  %0 = tosa.argmax %arg0 { axis = 1 : i32 } : (tensor<12x8x16xf8E4M3FN>) -> tensor<12x16xi32>
  return %0 : tensor<12x16xi32>
}

// -----
// CHECK-LABEL: avg_pool2d_f8E4M3FN
func.func @test_avg_pool2d_f8E4M3FN(%arg0: tensor<1x7x7x9xf8E4M3FN>) -> tensor<1x7x7x9xf8E4M3FN> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  %output_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  %0 = tosa.avg_pool2d %arg0, %input_zp, %output_zp {acc_type = f16, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xf8E4M3FN>, tensor<1xf8E4M3FN>, tensor<1xf8E4M3FN>) -> tensor<1x7x7x9xf8E4M3FN>
  return %0 : tensor<1x7x7x9xf8E4M3FN>
}

// -----
// CHECK-LABEL: conv2d_f8E4M3FN
func.func @test_conv2d_f8E4M3FN(%arg0: tensor<1x4x4x4xf8E4M3FN>, %arg1: tensor<8x1x1x4xf8E4M3FN>, %arg2: tensor<8xf16>) -> tensor<1x4x4x8xf16> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %input_zp, %weight_zp {acc_type = f16, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, local_bound = true} : (tensor<1x4x4x4xf8E4M3FN>, tensor<8x1x1x4xf8E4M3FN>, tensor<8xf16>, tensor<1xf8E4M3FN>, tensor<1xf8E4M3FN>) -> tensor<1x4x4x8xf16>
  return %0 : tensor<1x4x4x8xf16>
}

// -----
// CHECK-LABEL: conv3d_f8E4M3FN
func.func @test_conv3d_f8E4M3FN(%arg0: tensor<1x4x8x21x17xf8E4M3FN>, %arg1: tensor<34x1x1x1x17xf8E4M3FN>, %arg2: tensor<34xf16>, %arg3: tensor<1xf8E4M3FN>, %arg4: tensor<1xf8E4M3FN>) -> tensor<1x4x8x21x34xf16> {
  %0 = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f16, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<1x4x8x21x17xf8E4M3FN>, tensor<34x1x1x1x17xf8E4M3FN>, tensor<34xf16>, tensor<1xf8E4M3FN>, tensor<1xf8E4M3FN>) -> tensor<1x4x8x21x34xf16>
  return %0 : tensor<1x4x8x21x34xf16>
}

// -----
// CHECK-LABEL: depthwise_conv2d_f8E4M3FN
func.func @test_depthwise_conv2d_f8E4M3FN(%arg0: tensor<1x4x4x4xf8E4M3FN>, %arg1: tensor<1x1x4x2xf8E4M3FN>, %arg2: tensor<8xf16>, %arg3: tensor<1xf8E4M3FN>, %arg4: tensor<1xf8E4M3FN>) -> tensor<1x4x4x8xf16> {
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f16, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xf8E4M3FN>, tensor<1x1x4x2xf8E4M3FN>, tensor<8xf16>, tensor<1xf8E4M3FN>, tensor<1xf8E4M3FN>) -> tensor<1x4x4x8xf16>
  return %0 : tensor<1x4x4x8xf16>
}

// -----
// CHECK-LABEL: matmul_f8E4M3FN
func.func @test_matmul_f8E4M3FN(%arg0: tensor<1x14x19xf8E4M3FN>, %arg1: tensor<1x19x28xf8E4M3FN>) -> tensor<1x14x28xf16> {
  %azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  %bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FN>}> : () -> tensor<1xf8E4M3FN>
  %0 = tosa.matmul %arg0, %arg1, %azp0, %bzp0 : (tensor<1x14x19xf8E4M3FN>, tensor<1x19x28xf8E4M3FN>, tensor<1xf8E4M3FN>, tensor<1xf8E4M3FN>) -> tensor<1x14x28xf16>
  return %0 : tensor<1x14x28xf16>
}

// -----
// CHECK-LABEL: max_pool2d_f8E4M3FN
func.func @test_max_pool2d_f8E4M3FN(%arg0: tensor<1x32x32x8xf8E4M3FN>) -> tensor<1x32x32x8xf8E4M3FN> {
  %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf8E4M3FN>) -> tensor<1x32x32x8xf8E4M3FN>
  return %0 : tensor<1x32x32x8xf8E4M3FN>
}

// -----
// CHECK-LABEL: transpose_conv2d_f8E4M3FN
func.func @test_transpose_conv2d_f8E4M3FN(%arg0: tensor<1x32x32x8xf8E4M3FN>, %arg1: tensor<16x1x1x8xf8E4M3FN>, %arg2: tensor<16xf16>, %arg3: tensor<1xf8E4M3FN>, %arg4: tensor<1xf8E4M3FN>) -> tensor<1x32x32x16xf16> {
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f16, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf8E4M3FN>, tensor<16x1x1x8xf8E4M3FN>, tensor<16xf16>, tensor<1xf8E4M3FN>, tensor<1xf8E4M3FN>) -> tensor<1x32x32x16xf16>
  return %0 : tensor<1x32x32x16xf16>
}

// -----
// CHECK-LABEL: const_f8E4M3FN
func.func @test_const_f8E4M3FN(%arg0 : index) -> tensor<4xf8E4M3FN> {
    %0 = "tosa.const"() {values = dense<[3.0, -0.0, -1.0, 2.0]> : tensor<4xf8E4M3FN>} : () -> tensor<4xf8E4M3FN>
    return %0 : tensor<4xf8E4M3FN>
}

// -----
// CHECK-LABEL: cast_f8E4M3FN
func.func @test_cast_f8E4M3FN(%arg0: tensor<13x21x3xf8E4M3FN>) -> tensor<13x21x3xf16> {
  %0 = tosa.cast %arg0 : (tensor<13x21x3xf8E4M3FN>) -> tensor<13x21x3xf16>
  return %0 : tensor<13x21x3xf16>
}

// -----
// CHECK-LABEL: concat_f8E4M3FN
func.func @test_concat_f8E4M3FN(%arg0: tensor<13x21x3xf8E4M3FN>, %arg1: tensor<13x21x3xf8E4M3FN>) -> tensor<26x21x3xf8E4M3FN> {
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13x21x3xf8E4M3FN>, tensor<13x21x3xf8E4M3FN>) -> tensor<26x21x3xf8E4M3FN>
  return %0 : tensor<26x21x3xf8E4M3FN>
}

// -----
// CHECK-LABEL: pad_f8E4M3FN
func.func @test_pad_f8E4M3FN(%arg0: tensor<13x21x3xf8E4M3FN>) -> tensor<13x21x3xf8E4M3FN> {
  %padding = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
  %cst = "tosa.const"() { values = dense<-0.0> : tensor<1xf8E4M3FN> } : () -> tensor<1xf8E4M3FN>
  %0 = tosa.pad %arg0, %padding, %cst : (tensor<13x21x3xf8E4M3FN>, !tosa.shape<6>, tensor<1xf8E4M3FN>) -> tensor<13x21x3xf8E4M3FN>
  return %0 : tensor<13x21x3xf8E4M3FN>
}

// -----
// CHECK-LABEL: reshape_f8E4M3FN
func.func @test_reshape_f8E4M3FN(%arg0: tensor<13x21x3xf8E4M3FN>) -> tensor<1x819xf8E4M3FN> {
  %1 = tosa.const_shape {values = dense<[1, 819]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %0 = tosa.reshape %arg0, %1 : (tensor<13x21x3xf8E4M3FN>, !tosa.shape<2>) -> tensor<1x819xf8E4M3FN>
  return %0 : tensor<1x819xf8E4M3FN>
}

// -----
// CHECK-LABEL: reverse_f8E4M3FN
func.func @test_reverse_f8E4M3FN(%arg0: tensor<13x21x3xf8E4M3FN>) -> tensor<13x21x3xf8E4M3FN> {
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<13x21x3xf8E4M3FN>) -> tensor<13x21x3xf8E4M3FN>
  return %0 : tensor<13x21x3xf8E4M3FN>
}

// -----
// CHECK-LABEL: slice_f8E4M3FN
func.func @test_slice_f8E4M3FN(%arg0: tensor<13x21x3xf8E4M3FN>) -> tensor<4x11x1xf8E4M3FN> {
  %0 = tosa.const_shape {values = dense<[4, 11, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %1 = tosa.const_shape {values = dense<[6, 8, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %2 = tosa.slice %arg0, %0, %1 : (tensor<13x21x3xf8E4M3FN>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xf8E4M3FN>
  return %2 : tensor<4x11x1xf8E4M3FN>
}

// -----
// CHECK-LABEL: tile_f8E4M3FN
func.func @test_tile_f8E4M3FN(%arg0: tensor<13x21x3xf8E4M3FN>) -> tensor<39x21x6xf8E4M3FN> {
  %cst = tosa.const_shape { values = dense<[3, 1, 2]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = tosa.tile %arg0, %cst: (tensor<13x21x3xf8E4M3FN>, !tosa.shape<3>) -> tensor<39x21x6xf8E4M3FN>
  return %0 : tensor<39x21x6xf8E4M3FN>
}

// -----
// CHECK-LABEL: transpose_f8E4M3FN
func.func @test_transpose_f8E4M3FN(%arg0: tensor<13x21x3xf8E4M3FN>) -> tensor<3x13x21xf8E4M3FN> {
  %1 = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>} : (tensor<13x21x3xf8E4M3FN>) -> tensor<3x13x21xf8E4M3FN>
  return %1 : tensor<3x13x21xf8E4M3FN>
}

// -----
// CHECK-LABEL: gather_f8E4M3FN
func.func @test_gather_f8E4M3FN(%arg0: tensor<13x21x3xf8E4M3FN>, %arg1: tensor<13x26xi32>) -> tensor<13x26x3xf8E4M3FN> {
  %0 = tosa.gather %arg0, %arg1 : (tensor<13x21x3xf8E4M3FN>, tensor<13x26xi32>) -> tensor<13x26x3xf8E4M3FN>
  return %0 : tensor<13x26x3xf8E4M3FN>
}

// -----
// CHECK-LABEL: scatter_f8E4M3FN
func.func @test_scatter_f8E4M3FN(%arg0: tensor<13x29x3xf8E4M3FN>, %arg1: tensor<13x26xi32>, %arg2: tensor<13x26x3xf8E4M3FN>) -> tensor<13x29x3xf8E4M3FN> {
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x29x3xf8E4M3FN>, tensor<13x26xi32>, tensor<13x26x3xf8E4M3FN>) -> tensor<13x29x3xf8E4M3FN>
  return %0 : tensor<13x29x3xf8E4M3FN>
}
