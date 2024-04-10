// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics

func.func @conv1d_nwc_wcf_dyn_ch_dim(%input: memref<4x6x?xf32>, %filter: memref<1x?x8xf32>, %output: memref<4x2x8xf32>) {
  // expected-error @+1 {{Attempted to vectorize, but failed}}
  linalg.conv_1d_nwc_wcf
    {dilations = dense<1> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x6x?xf32>, memref<1x?x8xf32>)
    outs(%output : memref<4x2x8xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_1d_nwc_wcf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
}

// -----

// Masked vectorisation of 1D depthwise CW convs is not yet supported

func.func @depthwise_conv1d_ncw_cw(%input: memref<3x?x4xf32>, %filter: memref<?x1xf32>, %output: memref<3x?x4xf32>) {
  // expected-error @+1 {{Attempted to vectorize, but failed}}
  linalg.depthwise_conv_1d_ncw_cw
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<3x?x4xf32>, memref<?x1xf32>)
    outs(%output : memref<3x?x4xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_ncw_cw"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [3, 4, 5, 1] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @depthwise_conv1d_nwc_wc_dyn_w_dim(%input: memref<3x?x4xf32>, %filter: memref<?x4xf32>, %output: memref<3x?x4xf32>) {
  // expected-error @+1 {{Attempted to vectorize, but failed}}
  linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<3x?x4xf32>, memref<?x4xf32>)
    outs(%output : memref<3x?x4xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [3, 2, 4, 2] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @depthwise_conv1d_nwc_wc_dyn_ch_dim(%input: memref<3x5x?xf32>, %filter: memref<2x?xf32>, %output: memref<3x2x?xf32>) {
  // expected-error @+1 {{Attempted to vectorize, but failed}}
  linalg.depthwise_conv_1d_nwc_wc
    ins(%input, %filter : memref<3x5x?xf32>, memref<2x?xf32>)
    outs(%output : memref<3x2x?xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
}

// -----

func.func @depthwise_conv1d_nwc_wc_dyn_w_dim(%input: memref<3x?x3xf32>, %filter: memref<2x3xf32>, %output: memref<3x?x3xf32>) {
  // expected-error @+1 {{Attempted to vectorize, but failed}}
  linalg.depthwise_conv_1d_nwc_wc
    ins(%input, %filter : memref<3x?x3xf32>, memref<2x3xf32>)
    outs(%output : memref<3x?x3xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
}

// -----

func.func @conv1d_dyn_w_dim(%input: tensor<?xf32>, %filter: tensor<4xf32>, %output: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{Attempted to vectorize, but failed}}
  %0 = linalg.conv_1d ins(%input, %filter : tensor<?xf32>, tensor<4xf32>)
                     outs(%output : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_1d"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
}
