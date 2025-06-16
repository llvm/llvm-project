// RUN: mlir-opt %s -transform-interpreter -canonicalize --split-input-file -verify-diagnostics| FileCheck %s

func.func @conv2d(%arg0: tensor<2x10x10x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>, %arg3: tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x10x10x5xf32>, tensor<2x3x3x5xf32>) outs(%arg3 : tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_fhwc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.winograd_conv2d %0 { m = 4, r = 3 } : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @conv2d
// CHECK: linalg.winograd_filter_transform m(4) r(3)
// CHECK: linalg.winograd_input_transform m(4) r(3)
// CHECK: linalg.batch_matmul
// CHECK: linalg.winograd_output_transform m(4) r(3)

// -----

func.func @conv2d_unaligned(%arg0: tensor<2x11x11x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>, %arg3: tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x11x11x5xf32>, tensor<2x3x3x5xf32>) outs(%arg3 : tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32>
  return %0 : tensor<2x9x9x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_fhwc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.winograd_conv2d %0 { m = 4, r = 3 } : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @conv2d_unaligned
// CHECK:       linalg.winograd_filter_transform m(4) r(3)
// CHECK:       tensor.pad
// CHECK-SAME:  low[0, 0, 0, 0] high[0, 3, 3, 0]
// CHECK:       linalg.winograd_input_transform m(4) r(3)
// CHECK:       tensor.pad
// CHECK-SAME:  low[0, 0, 0, 0] high[0, 3, 3, 0]
// CHECK:       linalg.winograd_output_transform m(4) r(3)

// -----

func.func @conv2d_unsupported(%arg0: tensor<2x10x10x5xf32>, %arg1: tensor<3x3x5x2xf32>, %arg2: tensor<1xf32>, %arg3: tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x10x10x5xf32>, tensor<3x3x5x2xf32>) outs(%arg3 : tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{this operation is not supported to convert to Winograd Conv2D}}
    %1 = transform.structured.winograd_conv2d %0 { m = 4, r = 3 } : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

func.func @conv2d_unsupported_type(%arg0: memref<2x10x10x5xf32>, %arg1: memref<2x3x3x5xf32>, %arg2: memref<2x8x8x2xf32>) {
  linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : memref<2x10x10x5xf32>, memref<2x3x3x5xf32>) outs(%arg2 : memref<2x8x8x2xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_fhwc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{apply Winograd Conv2D failed}}
    %1 = transform.structured.winograd_conv2d %0 { m = 4, r = 3 } : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

func.func @conv2d(%arg0: tensor<2x?x?x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>, %arg3: tensor<2x?x?x2xf32>) -> tensor<2x?x?x2xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x?x?x5xf32>, tensor<2x3x3x5xf32>) outs(%arg3 : tensor<2x?x?x2xf32>) -> tensor<2x?x?x2xf32>
  return %0 : tensor<2x?x?x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_fhwc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{apply Winograd Conv2D failed}}
    %1 = transform.structured.winograd_conv2d %0 { m = 4, r = 3 } : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
