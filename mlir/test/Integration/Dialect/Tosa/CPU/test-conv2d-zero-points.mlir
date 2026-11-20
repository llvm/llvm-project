// DEFINE: %{tosa-to-linalg-pipeline} = -pass-pipeline="builtin.module(func.func(tosa-infer-shapes,tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith))"

// RUN:   mlir-opt %s \
// RUN:     %{tosa-to-linalg-pipeline} \
// RUN: | mlir-opt \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -buffer-deallocation-pipeline \
// RUN:     -test-lower-to-llvm \
// RUN: | mlir-runner \
// RUN:     -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN: | FileCheck %s

// Validate that the TOSA lowering for tosa.conv2d produces the expected results
// with non-constant zero points and combinations of constant and non-constant
// zero points.

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @conv_non_const_zps(%input: tensor<1x2x2x1xi8>,
                              %weights: tensor<1x2x2x1xi8>,
                              %bias: tensor<1xi32>,
                              %input_zp: tensor<1xi8>,
                              %weight_zp: tensor<1xi8>) -> tensor<1x3x3x1xi32> {
  %result = tosa.conv2d %input, %weights, %bias, %input_zp, %weight_zp
      pad([1, 1, 1, 1]) stride([1, 1]) dilation([1, 1]) acc_type(i32)
      : (tensor<1x2x2x1xi8>, tensor<1x2x2x1xi8>, tensor<1xi32>,
         tensor<1xi8>, tensor<1xi8>) -> tensor<1x3x3x1xi32>
  return %result : tensor<1x3x3x1xi32>
}

func.func @conv_const_input_zp(%input: tensor<1x2x2x1xi8>,
                               %weights: tensor<1x2x2x1xi8>,
                               %bias: tensor<1xi32>,
                               %weight_zp: tensor<1xi8>) -> tensor<1x3x3x1xi32> {
  %input_zp = "tosa.const"() <{values = dense<-3> : tensor<1xi8>}>
      : () -> tensor<1xi8>
  %result = tosa.conv2d %input, %weights, %bias, %input_zp, %weight_zp
      pad([1, 1, 1, 1]) stride([1, 1]) dilation([1, 1]) acc_type(i32)
      : (tensor<1x2x2x1xi8>, tensor<1x2x2x1xi8>, tensor<1xi32>,
         tensor<1xi8>, tensor<1xi8>) -> tensor<1x3x3x1xi32>
  return %result : tensor<1x3x3x1xi32>
}

func.func @conv_const_weight_zp(%input: tensor<1x2x2x1xi8>,
                                %weights: tensor<1x2x2x1xi8>,
                                %bias: tensor<1xi32>,
                                %input_zp: tensor<1xi8>) -> tensor<1x3x3x1xi32> {
  %weight_zp = "tosa.const"() <{values = dense<2> : tensor<1xi8>}>
      : () -> tensor<1xi8>
  %result = tosa.conv2d %input, %weights, %bias, %input_zp, %weight_zp
      pad([1, 1, 1, 1]) stride([1, 1]) dilation([1, 1]) acc_type(i32)
      : (tensor<1x2x2x1xi8>, tensor<1x2x2x1xi8>, tensor<1xi32>,
         tensor<1xi8>, tensor<1xi8>) -> tensor<1x3x3x1xi32>
  return %result : tensor<1x3x3x1xi32>
}

func.func @main() {
  %input = arith.constant dense<[[
    [[1], [2]],
    [[3], [4]]
  ]]> : tensor<1x2x2x1xi8>
  %weights = arith.constant dense<[[
    [[1], [2]],
    [[4], [8]]
  ]]> : tensor<1x2x2x1xi8>
  %bias = arith.constant dense<[7]> : tensor<1xi32>
  %negative_zp = arith.constant dense<[-3]> : tensor<1xi8>
  %positive_zp = arith.constant dense<[2]> : tensor<1xi8>

  // All three cases use input zero point -3 and weight zero point 2.
  // Verify the same expected result with both zero points non-constant,
  // only the input zero point constant, and only the weight zero point constant.
  %result_non_const_zps = func.call @conv_non_const_zps(
      %input,
      %weights,
      %bias,
      %negative_zp,
      %positive_zp)
      : (tensor<1x2x2x1xi8>, tensor<1x2x2x1xi8>, tensor<1xi32>,
         tensor<1xi8>, tensor<1xi8>) -> tensor<1x3x3x1xi32>
  %result_const_input_zp = func.call @conv_const_input_zp(
      %input,
      %weights,
      %bias,
      %positive_zp)
      : (tensor<1x2x2x1xi8>, tensor<1x2x2x1xi8>, tensor<1xi32>,
         tensor<1xi8>) -> tensor<1x3x3x1xi32>
  %result_const_weight_zp = func.call @conv_const_weight_zp(
      %input,
      %weights,
      %bias,
      %negative_zp)
      : (tensor<1x2x2x1xi8>, tensor<1x2x2x1xi8>, tensor<1xi32>,
         tensor<1xi8>) -> tensor<1x3x3x1xi32>
  // Also verify a negative weight zero point, using input zero point 2
  // and weight zero point -3.
  %result_negative_weight_zp = func.call @conv_non_const_zps(
      %input,
      %weights,
      %bias,
      %positive_zp,
      %negative_zp)
      : (tensor<1x2x2x1xi8>, tensor<1x2x2x1xi8>, tensor<1xi32>,
         tensor<1xi8>, tensor<1xi8>) -> tensor<1x3x3x1xi32>

  %buffer_non_const_zps = bufferization.to_buffer %result_non_const_zps
      : tensor<1x3x3x1xi32> to memref<1x3x3x1xi32>
  %unranked_non_const_zps = memref.cast %buffer_non_const_zps
      : memref<1x3x3x1xi32> to memref<*xi32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 3, 3, 1] strides = [9, 3, 1, 1] data =
  // CHECK-NEXT: {{^ *\[\[\[\[31\], *$}}
  // CHECK-NEXT: {{^ *\[45\], *$}}
  // CHECK-NEXT: {{^ *\[17\]\], *$}}

  // CHECK-NEXT: {{^ *\[\[43\], *$}}
  // CHECK-NEXT: {{^ *\[57\], *$}}
  // CHECK-NEXT: {{^ *\[16\]\], *$}}

  // CHECK-NEXT: {{^ *\[\[7\], *$}}
  // CHECK-NEXT: {{^ *\[1\], *$}}
  // CHECK-NEXT: {{^ *\[0\]\]\]\] *$}}
  func.call @printMemrefI32(%unranked_non_const_zps) : (memref<*xi32>) -> ()

  %buffer_const_input_zp = bufferization.to_buffer %result_const_input_zp
      : tensor<1x3x3x1xi32> to memref<1x3x3x1xi32>
  %unranked_const_input_zp = memref.cast %buffer_const_input_zp
      : memref<1x3x3x1xi32> to memref<*xi32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 3, 3, 1] strides = [9, 3, 1, 1] data =
  // CHECK-NEXT: {{^ *\[\[\[\[31\], *$}}
  // CHECK-NEXT: {{^ *\[45\], *$}}
  // CHECK-NEXT: {{^ *\[17\]\], *$}}

  // CHECK-NEXT: {{^ *\[\[43\], *$}}
  // CHECK-NEXT: {{^ *\[57\], *$}}
  // CHECK-NEXT: {{^ *\[16\]\], *$}}

  // CHECK-NEXT: {{^ *\[\[7\], *$}}
  // CHECK-NEXT: {{^ *\[1\], *$}}
  // CHECK-NEXT: {{^ *\[0\]\]\]\] *$}}
  func.call @printMemrefI32(%unranked_const_input_zp) : (memref<*xi32>) -> ()

  %buffer_const_weight_zp = bufferization.to_buffer %result_const_weight_zp
      : tensor<1x3x3x1xi32> to memref<1x3x3x1xi32>
  %unranked_const_weight_zp = memref.cast %buffer_const_weight_zp
      : memref<1x3x3x1xi32> to memref<*xi32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 3, 3, 1] strides = [9, 3, 1, 1] data =
  // CHECK-NEXT: {{^ *\[\[\[\[31\], *$}}
  // CHECK-NEXT: {{^ *\[45\], *$}}
  // CHECK-NEXT: {{^ *\[17\]\], *$}}

  // CHECK-NEXT: {{^ *\[\[43\], *$}}
  // CHECK-NEXT: {{^ *\[57\], *$}}
  // CHECK-NEXT: {{^ *\[16\]\], *$}}

  // CHECK-NEXT: {{^ *\[\[7\], *$}}
  // CHECK-NEXT: {{^ *\[1\], *$}}
  // CHECK-NEXT: {{^ *\[0\]\]\]\] *$}}
  func.call @printMemrefI32(%unranked_const_weight_zp) : (memref<*xi32>) -> ()

  %buffer_negative_weight_zp = bufferization.to_buffer %result_negative_weight_zp
      : tensor<1x3x3x1xi32> to memref<1x3x3x1xi32>
  %unranked_negative_weight_zp = memref.cast %buffer_negative_weight_zp
      : memref<1x3x3x1xi32> to memref<*xi32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 3, 3, 1] strides = [9, 3, 1, 1] data =
  // CHECK-NEXT: {{^ *\[\[\[\[-4\], *$}}
  // CHECK-NEXT: {{^ *\[0\], *$}}
  // CHECK-NEXT: {{^ *\[7\]\], *$}}

  // CHECK-NEXT: {{^ *\[\[13\], *$}}
  // CHECK-NEXT: {{^ *\[32\], *$}}
  // CHECK-NEXT: {{^ *\[21\]\], *$}}

  // CHECK-NEXT: {{^ *\[\[12\], *$}}
  // CHECK-NEXT: {{^ *\[21\], *$}}
  // CHECK-NEXT: {{^ *\[15\]\]\]\] *$}}
  func.call @printMemrefI32(%unranked_negative_weight_zp) : (memref<*xi32>) -> ()

  return
}
