// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @depthwise_conv_2d_input_nhwc_filter_hwc_tensor
func @depthwise_conv_2d_input_nhwc_filter_hwc_tensor(%input: tensor<1x113x113x96xf32>, %filter: tensor<3x3x96xf32>) -> tensor<1x56x56x96xf32> {
  %init = linalg.init_tensor [1, 56, 56, 96] : tensor<1x56x56x96xf32>
  // CHECK:      %{{.+}} = linalg.depthwise_conv_2d_input_nhwc_filter_hwc
  // CHECK-SAME:   {strides = dense<2> : vector<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  %0 = linalg.depthwise_conv_2d_input_nhwc_filter_hwc {strides = dense<2> : vector<2xi64>}
         ins(%input, %filter: tensor<1x113x113x96xf32>, tensor<3x3x96xf32>)
         outs(%init: tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  return %0: tensor<1x56x56x96xf32>
}

// CHECK-LABEL: func @depthwise_conv_2d_input_nhwc_filter_hwc_memref
func @depthwise_conv_2d_input_nhwc_filter_hwc_memref(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // CHECK:      linalg.depthwise_conv_2d_input_nhwc_filter_hwc
  // CHECK-SAME:   {strides = dense<2> : vector<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<1x113x113x96xf32>, memref<3x3x96xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<1x56x56x96xf32>)
  linalg.depthwise_conv_2d_input_nhwc_filter_hwc {strides = dense<2> : vector<2xi64>}
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

// -----

func @depthwise_conv_2d_input_nhwc_filter_missing_stride(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // expected-error @+1 {{missing indexing map required attribute 'strides'}}
  linalg.depthwise_conv_2d_input_nhwc_filter_hwc
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

// -----

func @depthwise_conv_2d_input_nhwc_filter_wrong_stride_element_type(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // expected-error @+1 {{incorrect element type for indexing map required attribute 'strides'}}
  linalg.depthwise_conv_2d_input_nhwc_filter_hwc {strides = dense<2.0> : vector<2xf32>}
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

// -----

func @depthwise_conv_2d_input_nhwc_filter_wrong_stride_size(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // expected-error @+1 {{incorrect shape for indexing map required attribute 'strides'}}
  linalg.depthwise_conv_2d_input_nhwc_filter_hwc {strides = dense<2> : vector<3xi64> }
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}
