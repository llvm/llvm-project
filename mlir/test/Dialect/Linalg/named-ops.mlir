// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @depthwise_conv_1d_nwc_wcm
func.func @depthwise_conv_1d_nwc_wcm(%input: tensor<1x12x8xf32>, %filter: tensor<3x8x8xf32>) -> tensor<1x10x8x8xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x10x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<1x10x8x8xf32>) -> tensor<1x10x8x8xf32>
  // CHECK: depthwise_conv_1d_nwc_wcm
  %0 = linalg.depthwise_conv_1d_nwc_wcm {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : tensor<1x12x8xf32>, tensor<3x8x8xf32>)
    outs(%fill : tensor<1x10x8x8xf32>) -> tensor<1x10x8x8xf32>
  return %0 : tensor<1x10x8x8xf32>
}

// -----

// CHECK-LABEL: func @depthwise_conv_1d_nwc_wc
func.func @depthwise_conv_1d_nwc_wc(%input: tensor<1x12x8xf32>, %filter: tensor<3x8xf32>) -> tensor<1x10x8xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x10x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<1x10x8xf32>) -> tensor<1x10x8xf32>
  // CHECK: depthwise_conv_1d_nwc_wc
  %0 = linalg.depthwise_conv_1d_nwc_wc {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : tensor<1x12x8xf32>, tensor<3x8xf32>)
    outs(%fill : tensor<1x10x8xf32>) -> tensor<1x10x8xf32>
  return %0 : tensor<1x10x8xf32>
}

// -----

// CHECK-LABEL: func @depthwise_conv_1d_ncw_cw
func.func @depthwise_conv_1d_ncw_cw(%input: tensor<1x8x12xf32>, %filter: tensor<8x3xf32>) -> tensor<1x8x10xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x8x10xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<1x8x10xf32>) -> tensor<1x8x10xf32>
  // CHECK: depthwise_conv_1d_ncw_cw
  %0 = linalg.depthwise_conv_1d_ncw_cw {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : tensor<1x8x12xf32>, tensor<8x3xf32>)
    outs(%fill : tensor<1x8x10xf32>) -> tensor<1x8x10xf32>
  return %0 : tensor<1x8x10xf32>
}

// -----

// CHECK-LABEL: func @depthwise_conv_2d_nhwc_hwcm_tensor
func.func @depthwise_conv_2d_nhwc_hwcm_tensor(%input: tensor<2x4x5x2xf32>, %filter: tensor<2x2x2x3xf32>) -> tensor<2x3x4x2x3xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2x3x4x2x3xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<2x3x4x2x3xf32>) -> tensor<2x3x4x2x3xf32>
  // CHECK:      %{{.+}} = linalg.depthwise_conv_2d_nhwc_hwcm
  // CHECK-SAME:   {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<2x4x5x2xf32>, tensor<2x2x2x3xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<2x3x4x2x3xf32>)
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm
     { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%input, %filter : tensor<2x4x5x2xf32>, tensor<2x2x2x3xf32>)
    outs(%fill : tensor<2x3x4x2x3xf32>) -> tensor<2x3x4x2x3xf32>
  return %0 : tensor<2x3x4x2x3xf32>
}

// CHECK-LABEL: func @depthwise_conv_2d_nhwc_hwcm_memref
func.func @depthwise_conv_2d_nhwc_hwcm_memref(%input: memref<2x4x5x2xf32>, %filter: memref<2x2x2x3xf32>, %output: memref<2x3x4x2x3xf32>) {
  // CHECK:      linalg.depthwise_conv_2d_nhwc_hwcm
  // CHECK-SAME:   {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<2x4x5x2xf32>, memref<2x2x2x3xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<2x3x4x2x3xf32>)
  linalg.depthwise_conv_2d_nhwc_hwcm
     { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%input, %filter : memref<2x4x5x2xf32>, memref<2x2x2x3xf32>)
    outs(%output : memref<2x3x4x2x3xf32>)
  return
}

// CHECK-LABEL: func @depthwise_conv_1d_nw_tensor
func.func @depthwise_conv_1d_nw_tensor(%input: tensor<1x113x96xf32>, %filter: tensor<3x96xf32>) -> tensor<1x56x96xf32> {
  %init = tensor.empty() : tensor<1x56x96xf32>
  // CHECK:      %{{.+}} = linalg.depthwise_conv_1d_nw
  // CHECK-SAME:   {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<1x113x96xf32>, tensor<3x96xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<1x56x96xf32>) -> tensor<1x56x96xf32>
  %0 = linalg.depthwise_conv_1d_nwc_wc {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>}
         ins(%input, %filter: tensor<1x113x96xf32>, tensor<3x96xf32>)
         outs(%init: tensor<1x56x96xf32>) -> tensor<1x56x96xf32>
  return %0: tensor<1x56x96xf32>
}

// CHECK-LABEL: func @depthwise_conv_2d_nhwc_hwc_tensor
func.func @depthwise_conv_2d_nhwc_hwc_tensor(%input: tensor<1x113x113x96xf32>, %filter: tensor<3x3x96xf32>) -> tensor<1x56x56x96xf32> {
  %init = tensor.empty() : tensor<1x56x56x96xf32>
  // CHECK:      %{{.+}} = linalg.depthwise_conv_2d_nhwc_hwc
  // CHECK-SAME:   {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
         ins(%input, %filter: tensor<1x113x113x96xf32>, tensor<3x3x96xf32>)
         outs(%init: tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  return %0: tensor<1x56x56x96xf32>
}

// CHECK-LABEL: func @depthwise_conv_2d_nhwc_hwc_memref
func.func @depthwise_conv_2d_nhwc_hwc_memref(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // CHECK:      linalg.depthwise_conv_2d_nhwc_hwc
  // CHECK-SAME:   {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<1x113x113x96xf32>, memref<3x3x96xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<1x56x56x96xf32>)
  linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

// CHECK-LABEL: func @depthwise_conv_2d_nchw_chw_tensor
func.func @depthwise_conv_2d_nchw_chw_tensor(%input: tensor<1x96x113x113xf32>, %filter: tensor<96x3x3xf32>) -> tensor<1x96x56x56xf32> {
  %init = tensor.empty() : tensor<1x96x56x56xf32>
  // CHECK:      %{{.+}} = linalg.depthwise_conv_2d_nchw_chw
  // CHECK-SAME:   {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<1x96x113x113xf32>, tensor<96x3x3xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
  %0 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
         ins(%input, %filter: tensor<1x96x113x113xf32>, tensor<96x3x3xf32>)
         outs(%init: tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
  return %0: tensor<1x96x56x56xf32>
}

// CHECK-LABEL: func @depthwise_conv_2d_nchw_chw_memref
func.func @depthwise_conv_2d_nchw_chw_memref(%input: memref<1x96x113x113xf32>, %filter: memref<96x3x3xf32>, %output: memref<1x96x56x56xf32>) {
  // CHECK:      linalg.depthwise_conv_2d_nchw_chw
  // CHECK-SAME:   {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<1x96x113x113xf32>, memref<96x3x3xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<1x96x56x56xf32>)
  linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%input, %filter: memref<1x96x113x113xf32>, memref<96x3x3xf32>)
    outs(%output: memref<1x96x56x56xf32>)
  return
}

func.func @depthwise_conv_2d_nhwc_hwcm_tensor_dilated(%input: tensor<2x8x9x2xf32>, %filter: tensor<2x2x2x3xf32>) -> tensor<2x6x7x2x3xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2x6x7x2x3xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<2x6x7x2x3xf32>) -> tensor<2x6x7x2x3xf32>
  // CHECK:      %{{.+}} = linalg.depthwise_conv_2d_nhwc_hwcm
  // CHECK-SAME:   {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<2x8x9x2xf32>, tensor<2x2x2x3xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<2x6x7x2x3xf32>)
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm
     { dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%input, %filter : tensor<2x8x9x2xf32>, tensor<2x2x2x3xf32>)
    outs(%fill : tensor<2x6x7x2x3xf32>) -> tensor<2x6x7x2x3xf32>
  return %0 : tensor<2x6x7x2x3xf32>
}

// CHECK-LABEL: func @depthwise_conv_2d_nhwc_hwcm_memref_dilated
func.func @depthwise_conv_2d_nhwc_hwcm_memref_dilated(%input: memref<2x8x9x2xf32>, %filter: memref<2x2x2x3xf32>, %output: memref<2x6x7x2x3xf32>) {
  // CHECK:      linalg.depthwise_conv_2d_nhwc_hwcm
  // CHECK-SAME:   {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<2x8x9x2xf32>, memref<2x2x2x3xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<2x6x7x2x3xf32>)
  linalg.depthwise_conv_2d_nhwc_hwcm
     { dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%input, %filter : memref<2x8x9x2xf32>, memref<2x2x2x3xf32>)
    outs(%output : memref<2x6x7x2x3xf32>)
  return
}

// -----

// CHECK-LABEL: func @depthwise_conv_2d_input_nhwc_filter_default_attributes
func.func @depthwise_conv_2d_input_nhwc_filter_default_attributes(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // CHECK:      linalg.depthwise_conv_2d_nhwc_hwc
  // CHECK-NOT:  strides =
  // CHECK-NOT:  dilations =
  linalg.depthwise_conv_2d_nhwc_hwc
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

// -----

func.func @depthwise_conv_2d_input_nhwc_filter_wrong_stride_element_type_properties(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // expected-error @+1 {{invalid properties {dilations = dense<1> : vector<2xi64>, operandSegmentSizes = array<i32: 2, 1>, strides = dense<2.000000e+00> : vector<2xf32>} for op linalg.depthwise_conv_2d_nhwc_hwc: Invalid attribute `strides` in property conversion: dense<2.000000e+00> : vector<2xf32>}}
  linalg.depthwise_conv_2d_nhwc_hwc <{dilations = dense<1> : vector<2xi64>, strides = dense<2.0> : vector<2xf32>}>
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

// -----

func.func @depthwise_conv_2d_input_nhwc_filter_wrong_stride_element_type(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // expected-error @+1 {{op attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]}}
  linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2.0> : vector<2xf32>}
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

// -----

func.func @depthwise_conv_2d_input_nhwc_filter_wrong_stride_size(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // expected-error @+1 {{op attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]}}
  linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<3xi64> }
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

// -----

// CHECK-LABEL: func @depthwise_conv_3d_ndhwc_dhwcm
func.func @depthwise_conv_3d_ndhwc_dhwcm(%input: tensor<2x6x13x12x6xf32>, %filter: tensor<2x1x3x6x6xf32>) -> tensor<2x3x13x4x6x6xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2x3x13x4x6x6xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<2x3x13x4x6x6xf32>) -> tensor<2x3x13x4x6x6xf32>
  // CHECK: depthwise_conv_3d_ndhwc_dhwcm
  %0 = linalg.depthwise_conv_3d_ndhwc_dhwcm {dilations = dense<1> : tensor<3xi64>, strides = dense<[2, 1, 3]> : tensor<3xi64>}
    ins(%input, %filter : tensor<2x6x13x12x6xf32>, tensor<2x1x3x6x6xf32>)
    outs(%fill : tensor<2x3x13x4x6x6xf32>) -> tensor<2x3x13x4x6x6xf32>
  return %0 : tensor<2x3x13x4x6x6xf32>
}

// -----

// CHECK-LABEL: func @depthwise_conv_3d_ndhwc_dhwc
func.func @depthwise_conv_3d_ndhwc_dhwc(%input: tensor<2x6x13x12x6xf32>, %filter: tensor<2x1x3x6xf32>) -> tensor<2x3x13x4x6xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2x3x13x4x6xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<2x3x13x4x6xf32>) -> tensor<2x3x13x4x6xf32>
  // CHECK: depthwise_conv_3d_ndhwc_dhwc
  %0 = linalg.depthwise_conv_3d_ndhwc_dhwc {dilations = dense<1> : tensor<3xi64>, strides = dense<[2, 1, 3]> : tensor<3xi64>}
    ins(%input, %filter : tensor<2x6x13x12x6xf32>, tensor<2x1x3x6xf32>)
    outs(%fill : tensor<2x3x13x4x6xf32>) -> tensor<2x3x13x4x6xf32>
  return %0 : tensor<2x3x13x4x6xf32>
}

// -----

// CHECK-LABEL: func @depthwise_conv_3d_ncdhw_cdhw
func.func @depthwise_conv_3d_ncdhw_cdhw(%input: tensor<2x6x6x13x12xf32>, %filter: tensor<6x2x1x3xf32>) -> tensor<2x6x3x13x4xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2x6x3x13x4xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<2x6x3x13x4xf32>) -> tensor<2x6x3x13x4xf32>
  // CHECK: depthwise_conv_3d_ncdhw_cdhw
  %0 = linalg.depthwise_conv_3d_ncdhw_cdhw {dilations = dense<1> : tensor<3xi64>, strides = dense<[2, 1, 3]> : tensor<3xi64>}
    ins(%input, %filter : tensor<2x6x6x13x12xf32>, tensor<6x2x1x3xf32>)
    outs(%fill : tensor<2x6x3x13x4xf32>) -> tensor<2x6x3x13x4xf32>
  return %0 : tensor<2x6x3x13x4xf32>
}

// -----

// CHECK-LABEL: func @conv_1d_nwc_wcf
func.func @conv_1d_nwc_wcf(%input: tensor<?x?x?xf32>, %filter: tensor<?x?x?xf32>, %init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK:      %{{.+}} = linalg.conv_1d_nwc_wcf
  // CHECK-SAME:   dilations = dense<1> : tensor<1xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<1xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %0 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>,
                                            strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs (%init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @conv_1d_nwc_wcf
func.func @conv_1d_nwc_wcf(%input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?xf32>) {
  // CHECK:      linalg.conv_1d_nwc_wcf
  // CHECK-SAME:   dilations = dense<1> : tensor<1xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<1xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<?x?x?xf32>, memref<?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>,
                                       strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: memref<?x?x?xf32>, memref<?x?x?xf32>)
    outs (%output: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @conv_1d_ncw_fcw
func.func @conv_1d_ncw_fcw(%input: tensor<?x?x?xf32>, %filter: tensor<?x?x?xf32>, %init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK:      %{{.+}} = linalg.conv_1d_ncw_fcw
  // CHECK-SAME:   dilations = dense<1> : tensor<1xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<1xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %0 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : tensor<1xi64>,
                                            strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs (%init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @conv_1d_ncw_fcw
func.func @conv_1d_ncw_fcw(%input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?xf32>) {
  // CHECK:      linalg.conv_1d_ncw_fcw
  // CHECK-SAME:   dilations = dense<1> : tensor<1xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<1xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<?x?x?xf32>, memref<?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.conv_1d_ncw_fcw {dilations = dense<1> : tensor<1xi64>,
                                       strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: memref<?x?x?xf32>, memref<?x?x?xf32>)
    outs (%output: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @conv_2d_nhwc_hwcf
func.func @conv_2d_nhwc_hwcf(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK:      %{{.+}} = linalg.conv_2d_nhwc_hwcf
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @conv_2d_ngchw_fgchw
func.func @conv_2d_ngchw_fgchw(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  // CHECK:      %{{.+}} = linalg.conv_2d_ngchw_fgchw
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  %0 = linalg.conv_2d_ngchw_fgchw {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @conv_2d_nhwc_fhwc
func.func @conv_2d_nhwc_fhwc(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK:      %{{.+}} = linalg.conv_2d_nhwc_fhwc
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @conv_2d_nhwc_fhwc_static
func.func @conv_2d_nhwc_fhwc_static(%input: tensor<?x128x128x32xf32>, %filter: tensor<64x3x3x32xf32>, %init: tensor<?x126x126x64xf32>) -> tensor<?x126x126x64xf32> {
  // CHECK:      %{{.+}} = linalg.conv_2d_nhwc_fhwc
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x128x128x32xf32>, tensor<64x3x3x32xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x126x126x64xf32>) -> tensor<?x126x126x64xf32>
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x128x128x32xf32>, tensor<64x3x3x32xf32>)
    outs (%init: tensor<?x126x126x64xf32>) -> tensor<?x126x126x64xf32>
  return %0 : tensor<?x126x126x64xf32>
}

// -----

// CHECK-LABEL: func @conv_2d_nhwc_hwcf
func.func @conv_2d_nhwc_hwcf(%input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32>) {
  // CHECK:      linalg.conv_2d_nhwc_hwcf
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<?x?x?x?xf32>)
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                         strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
    outs (%output: memref<?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @conv_2d_ngchw_fgchw
func.func @conv_2d_ngchw_fgchw(%input: memref<?x?x?x?x?xf32>, %filter: memref<?x?x?x?x?xf32>, %output: memref<?x?x?x?x?xf32>) {
  // CHECK:      linalg.conv_2d_ngchw_fgchw
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<?x?x?x?x?xf32>)
  linalg.conv_2d_ngchw_fgchw {dilations = dense<1> : tensor<2xi64>,
                                         strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
    outs (%output: memref<?x?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @conv_2d_nhwgc_gfhwc
func.func @conv_2d_nhwgc_gfhwc(%input: memref<?x?x?x?x?xf32>, %filter: memref<?x?x?x?x?xf32>, %output: memref<?x?x?x?x?xf32>) {
  // CHECK:      linalg.conv_2d_nhwgc_gfhwc
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<?x?x?x?x?xf32>)
  linalg.conv_2d_nhwgc_gfhwc {dilations = dense<1> : tensor<2xi64>,
                                         strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
    outs (%output: memref<?x?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @conv_2d_nhwgc_gfhwc_tensor
func.func @conv_2d_nhwgc_gfhwc_tensor(%input: tensor<1x28x28x2x3xf32>, %filter: tensor<2x8x3x3x3xf32>, %output: tensor<1x26x26x2x8xf32>) -> tensor<1x26x26x2x8xf32>  {
  // CHECK:      linalg.conv_2d_nhwgc_gfhwc
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<1x28x28x2x3xf32>, tensor<2x8x3x3x3xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<1x26x26x2x8xf32>) -> tensor<1x26x26x2x8xf32>
  %0 = linalg.conv_2d_nhwgc_gfhwc {dilations = dense<1> : tensor<2xi64>,
                                         strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x28x28x2x3xf32>, tensor<2x8x3x3x3xf32>)
    outs (%output: tensor<1x26x26x2x8xf32>) -> tensor<1x26x26x2x8xf32>
  return  %0 : tensor<1x26x26x2x8xf32>
}

// -----

// CHECK-LABEL: func @conv_2d_ngchw_fgchw_dimensions
func.func @conv_2d_ngchw_fgchw_dimensions(%input: tensor<1x5x3x32x32xf32>, %filter: tensor<2x5x3x3x3xf32>, %init: tensor<1x5x2x30x30xf32>) -> tensor<1x5x2x30x30xf32> {
  // CHECK:      linalg.conv_2d_ngchw_fgchw
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<1x5x3x32x32xf32>, tensor<2x5x3x3x3xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<1x5x2x30x30xf32>) -> tensor<1x5x2x30x30xf32>
  %0 = linalg.conv_2d_ngchw_fgchw {dilations = dense<1> : tensor<2xi64>,
                                         strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x5x3x32x32xf32>, tensor<2x5x3x3x3xf32>)
    outs (%init: tensor<1x5x2x30x30xf32>) -> tensor<1x5x2x30x30xf32>
  return %0 : tensor<1x5x2x30x30xf32>
}

// -----

// CHECK-LABEL: func @conv_2d_ngchw_gfchw
func.func @conv_2d_ngchw_gfchw(%input: tensor<1x5x3x32x32xf32>, %filter: tensor<5x2x3x3x3xf32>, %init: tensor<1x5x2x30x30xf32>) -> tensor<1x5x2x30x30xf32> {
  // CHECK:      linalg.conv_2d_ngchw_gfchw
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<1x5x3x32x32xf32>, tensor<5x2x3x3x3xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<1x5x2x30x30xf32>) -> tensor<1x5x2x30x30xf32>
  %0 = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : tensor<2xi64>,
                                         strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x5x3x32x32xf32>, tensor<5x2x3x3x3xf32>)
    outs (%init: tensor<1x5x2x30x30xf32>) -> tensor<1x5x2x30x30xf32>
  return %0 : tensor<1x5x2x30x30xf32>
}

// -----

// CHECK-LABEL: func @conv_2d_ngchw_gfchw_q
func.func @conv_2d_ngchw_gfchw_q(%input: tensor<1x5x3x32x32xi8>, %filter: tensor<5x2x3x3x3xi8>, %inputzp: i32, %filterzp: i32, %init: tensor<1x5x2x30x30xi32>) -> tensor<1x5x2x30x30xi32> {
  // CHECK:      linalg.conv_2d_ngchw_gfchw_q
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<1x5x3x32x32xi8>, tensor<5x2x3x3x3xi8>, i32, i32)
  // CHECK-SAME:   outs(%{{.+}} : tensor<1x5x2x30x30xi32>) -> tensor<1x5x2x30x30xi32>
  %0 = linalg.conv_2d_ngchw_gfchw_q {dilations = dense<1> : tensor<2xi64>,
                                         strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter, %inputzp, %filterzp: tensor<1x5x3x32x32xi8>, tensor<5x2x3x3x3xi8>, i32, i32)
    outs (%init: tensor<1x5x2x30x30xi32>) -> tensor<1x5x2x30x30xi32>
  return %0 : tensor<1x5x2x30x30xi32>
}
// -----

// CHECK-LABEL: func @conv_3d_ndhwc_dhwcf
func.func @conv_3d_ndhwc_dhwcf(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  // CHECK:      %{{.+}} = linalg.conv_3d_ndhwc_dhwcf
  // CHECK-SAME:   dilations = dense<1> : tensor<3xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<3xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  %0 = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @conv_3d_ndhwc_dhwcf
func.func @conv_3d_ndhwc_dhwcf(%input: memref<?x?x?x?x?xf32>, %filter: memref<?x?x?x?x?xf32>, %output: memref<?x?x?x?x?xf32>) {
  // CHECK:      linalg.conv_3d_ndhwc_dhwcf
  // CHECK-SAME:   dilations = dense<1> : tensor<3xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<3xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<?x?x?x?x?xf32>)
  linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                           strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
    outs (%output: memref<?x?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @conv_3d_ncdhw_fcdhw
func.func @conv_3d_ncdhw_fcdhw(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  // CHECK:      %{{.+}} = linalg.conv_3d_ncdhw_fcdhw
  // CHECK-SAME:   dilations = dense<1> : tensor<3xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<3xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  %0 = linalg.conv_3d_ncdhw_fcdhw {dilations = dense<1> : tensor<3xi64>,
                                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @conv_3d_ncdhw_fcdhw
func.func @conv_3d_ncdhw_fcdhw(%input: memref<?x?x?x?x?xf32>, %filter: memref<?x?x?x?x?xf32>, %output: memref<?x?x?x?x?xf32>) {
  // CHECK:      linalg.conv_3d_ncdhw_fcdhw
  // CHECK-SAME:   dilations = dense<1> : tensor<3xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<3xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<?x?x?x?x?xf32>)
  linalg.conv_3d_ncdhw_fcdhw {dilations = dense<1> : tensor<3xi64>,
                                           strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
    outs (%output: memref<?x?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nhwc_sum_tensor
// CHECK:         %{{.+}} = linalg.pooling_nhwc_sum
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x1xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
func.func @pooling_nhwc_sum_tensor(%input: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %fake = tensor.empty() : tensor<3x3xf32>
  %init = tensor.empty() : tensor<1x2x2x1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  %res = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xf32>, tensor<3x3xf32>)
    outs(%fill: tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  return %res : tensor<1x2x2x1xf32>
}

// -----

// CHECK-LABEL: func @pooling_nwc_sum_tensor
// CHECK:         %{{.+}} = linalg.pooling_nwc_sum
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x1xf32>, tensor<3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
func.func @pooling_nwc_sum_tensor(%input: tensor<1x4x1xf32>) -> tensor<1x2x1xf32> {
  %fake = tensor.empty() : tensor<3xf32>
  %init = tensor.empty() : tensor<1x2x1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
  %res = linalg.pooling_nwc_sum {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: tensor<1x4x1xf32>, tensor<3xf32>)
    outs(%fill: tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
  return %res : tensor<1x2x1xf32>
}

// -----

// CHECK-LABEL: func @pooling_nhwc_sum
// CHECK:         linalg.pooling_nhwc_sum
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x1xf32>, memref<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x1xf32>)
func.func @pooling_nhwc_sum(%input: memref<1x4x4x1xf32>, %fake: memref<3x3xf32>, %output: memref<1x2x2x1xf32>) {
  linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x4x4x1xf32>, memref<3x3xf32>)
    outs(%output: memref<1x2x2x1xf32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nwc_sum
// CHECK:         linalg.pooling_nwc_sum
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x1xf32>, memref<3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x1xf32>)
func.func @pooling_nwc_sum(%input: memref<1x4x1xf32>, %fake: memref<3xf32>, %output: memref<1x2x1xf32>) {
  linalg.pooling_nwc_sum {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: memref<1x4x1xf32>, memref<3xf32>)
    outs(%output: memref<1x2x1xf32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nchw_sum_tensor
// CHECK:         %{{.+}} = linalg.pooling_nchw_sum
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x1x4x4xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
func.func @pooling_nchw_sum_tensor(%input: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
  %fake = tensor.empty() : tensor<3x3xf32>
  %init = tensor.empty() : tensor<1x1x2x2xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  %res = linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x1x4x4xf32>, tensor<3x3xf32>)
    outs(%fill: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  return %res : tensor<1x1x2x2xf32>
}

// -----

// CHECK-LABEL: func @pooling_ncw_sum_tensor
// CHECK:         %{{.+}} = linalg.pooling_ncw_sum
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x1x4xf32>, tensor<3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
func.func @pooling_ncw_sum_tensor(%input: tensor<1x1x4xf32>) -> tensor<1x1x2xf32> {
  %fake = tensor.empty() : tensor<3xf32>
  %init = tensor.empty() : tensor<1x1x2xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
  %res = linalg.pooling_ncw_sum {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: tensor<1x1x4xf32>, tensor<3xf32>)
    outs(%fill: tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
  return %res : tensor<1x1x2xf32>
}

// -----

// CHECK-LABEL: func @pooling_nchw_sum
// CHECK:         linalg.pooling_nchw_sum
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x1x4x4xf32>, memref<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x1x2x2xf32>)
func.func @pooling_nchw_sum(%input: memref<1x1x4x4xf32>, %fake: memref<3x3xf32>, %output: memref<1x1x2x2xf32>) {
  linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x1x4x4xf32>, memref<3x3xf32>)
    outs(%output: memref<1x1x2x2xf32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_ncw_sum
// CHECK:         linalg.pooling_ncw_sum
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x1x4xf32>, memref<3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x1x2xf32>)
func.func @pooling_ncw_sum(%input: memref<1x1x4xf32>, %fake: memref<3xf32>, %output: memref<1x1x2xf32>) {
  linalg.pooling_ncw_sum {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: memref<1x1x4xf32>, memref<3xf32>)
    outs(%output: memref<1x1x2xf32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nhwc_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_nhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x1xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
func.func @pooling_nhwc_max_tensor(%input: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %fake = tensor.empty() : tensor<3x3xf32>
  %init = tensor.empty() : tensor<1x2x2x1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  %res = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xf32>, tensor<3x3xf32>)
    outs(%fill: tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  return %res : tensor<1x2x2x1xf32>
}

// -----
// CHECK-LABEL: func @pooling_nwc_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_nwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x1xf32>, tensor<3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
func.func @pooling_nwc_max_tensor(%input: tensor<1x4x1xf32>) -> tensor<1x2x1xf32> {
  %fake = tensor.empty() : tensor<3xf32>
  %init = tensor.empty() : tensor<1x2x1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
  %res = linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: tensor<1x4x1xf32>, tensor<3xf32>)
    outs(%fill: tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
  return %res : tensor<1x2x1xf32>
}

// -----
// CHECK-LABEL: func @pooling_nchw_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_nchw_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x1x4x4xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>

func.func @pooling_nchw_max_tensor(%input: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
  %fake = tensor.empty() : tensor<3x3xf32>
  %init = tensor.empty() : tensor<1x1x2x2xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  %res = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x1x4x4xf32>, tensor<3x3xf32>)
    outs(%fill: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  return %res : tensor<1x1x2x2xf32>
}

// -----
// CHECK-LABEL: func @pooling_ncw_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_ncw_max
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x1x4xf32>, tensor<3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x1x2xf32>) -> tensor<1x1x2xf32>

func.func @pooling_ncw_max_tensor(%input: tensor<1x1x4xf32>) -> tensor<1x1x2xf32> {
  %fake = tensor.empty() : tensor<3xf32>
  %init = tensor.empty() : tensor<1x1x2xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
  %res = linalg.pooling_ncw_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: tensor<1x1x4xf32>, tensor<3xf32>)
    outs(%fill: tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
  return %res : tensor<1x1x2xf32>
}

// -----

// CHECK-LABEL: func @pooling_nhwc_max
// CHECK:         linalg.pooling_nhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x1xf32>, memref<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x1xf32>)
func.func @pooling_nhwc_max(%input: memref<1x4x4x1xf32>, %fake: memref<3x3xf32>, %output: memref<1x2x2x1xf32>) {
  linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x4x4x1xf32>, memref<3x3xf32>)
    outs(%output: memref<1x2x2x1xf32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nwc_max
// CHECK:         linalg.pooling_nwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x1xf32>, memref<3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x1xf32>)
func.func @pooling_nwc_max(%input: memref<1x4x1xf32>, %fake: memref<3xf32>, %output: memref<1x2x1xf32>) {
  linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: memref<1x4x1xf32>, memref<3xf32>)
    outs(%output: memref<1x2x1xf32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nhwc_i8_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_nhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x1xi8>, tensor<3x3xi8>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x1xi8>) -> tensor<1x2x2x1xi8>
func.func @pooling_nhwc_i8_max_tensor(%input: tensor<1x4x4x1xi8>) -> tensor<1x2x2x1xi8> {
  %fake = tensor.empty() : tensor<3x3xi8>
  %init = tensor.empty() : tensor<1x2x2x1xi8>
  %cst = arith.constant 0 : i8
  %fill = linalg.fill ins(%cst : i8) outs(%init : tensor<1x2x2x1xi8>) -> tensor<1x2x2x1xi8>
  %res = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xi8>, tensor<3x3xi8>)
    outs(%fill: tensor<1x2x2x1xi8>) -> tensor<1x2x2x1xi8>
  return %res : tensor<1x2x2x1xi8>
}

// -----

// CHECK-LABEL: func @pooling_nwc_i8_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_nwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x1xi8>, tensor<3xi8>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x1xi8>) -> tensor<1x2x1xi8>
func.func @pooling_nwc_i8_max_tensor(%input: tensor<1x4x1xi8>) -> tensor<1x2x1xi8> {
  %fake = tensor.empty() : tensor<3xi8>
  %init = tensor.empty() : tensor<1x2x1xi8>
  %cst = arith.constant 0 : i8
  %fill = linalg.fill ins(%cst : i8) outs(%init : tensor<1x2x1xi8>) -> tensor<1x2x1xi8>
  %res = linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: tensor<1x4x1xi8>, tensor<3xi8>)
    outs(%fill: tensor<1x2x1xi8>) -> tensor<1x2x1xi8>
  return %res : tensor<1x2x1xi8>
}

// -----

// CHECK-LABEL: func @pooling_nhwc_i8_max
// CHECK:         linalg.pooling_nhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x1xi8>, memref<3x3xi8>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x1xi8>)
func.func @pooling_nhwc_i8_max(%input: memref<1x4x4x1xi8>, %fake: memref<3x3xi8>, %output: memref<1x2x2x1xi8>) {
  linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x4x4x1xi8>, memref<3x3xi8>)
    outs(%output: memref<1x2x2x1xi8>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nwc_i8_max
// CHECK:         linalg.pooling_nwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x1xi8>, memref<3xi8>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x1xi8>)
func.func @pooling_nwc_i8_max(%input: memref<1x4x1xi8>, %fake: memref<3xi8>, %output: memref<1x2x1xi8>) {
  linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: memref<1x4x1xi8>, memref<3xi8>)
    outs(%output: memref<1x2x1xi8>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nhwc_i16_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_nhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x1xi16>, tensor<3x3xi16>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x1xi16>) -> tensor<1x2x2x1xi16>
func.func @pooling_nhwc_i16_max_tensor(%input: tensor<1x4x4x1xi16>) -> tensor<1x2x2x1xi16> {
  %fake = tensor.empty() : tensor<3x3xi16>
  %init = tensor.empty() : tensor<1x2x2x1xi16>
  %cst = arith.constant 0 : i16
  %fill = linalg.fill ins(%cst : i16) outs(%init : tensor<1x2x2x1xi16>) -> tensor<1x2x2x1xi16>
  %res = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xi16>, tensor<3x3xi16>)
    outs(%fill: tensor<1x2x2x1xi16>) -> tensor<1x2x2x1xi16>
  return %res : tensor<1x2x2x1xi16>
}

// -----

// CHECK-LABEL: func @pooling_nwc_i16_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_nwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x1xi16>, tensor<3xi16>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x1xi16>) -> tensor<1x2x1xi16>
func.func @pooling_nwc_i16_max_tensor(%input: tensor<1x4x1xi16>) -> tensor<1x2x1xi16> {
  %fake = tensor.empty() : tensor<3xi16>
  %init = tensor.empty() : tensor<1x2x1xi16>
  %cst = arith.constant 0 : i16
  %fill = linalg.fill ins(%cst : i16) outs(%init : tensor<1x2x1xi16>) -> tensor<1x2x1xi16>
  %res = linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: tensor<1x4x1xi16>, tensor<3xi16>)
    outs(%fill: tensor<1x2x1xi16>) -> tensor<1x2x1xi16>
  return %res : tensor<1x2x1xi16>
}

// -----

// CHECK-LABEL: func @pooling_nhwc_i16_max
// CHECK:         linalg.pooling_nhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x1xi16>, memref<3x3xi16>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x1xi16>)
func.func @pooling_nhwc_i16_max(%input: memref<1x4x4x1xi16>, %fake: memref<3x3xi16>, %output: memref<1x2x2x1xi16>) {
  linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x4x4x1xi16>, memref<3x3xi16>)
    outs(%output: memref<1x2x2x1xi16>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nwc_i16_max
// CHECK:         linalg.pooling_nwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x1xi16>, memref<3xi16>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x1xi16>)
func.func @pooling_nwc_i16_max(%input: memref<1x4x1xi16>, %fake: memref<3xi16>, %output: memref<1x2x1xi16>) {
  linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: memref<1x4x1xi16>, memref<3xi16>)
    outs(%output: memref<1x2x1xi16>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nhwc_i32_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_nhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x1xi32>, tensor<3x3xi32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x1xi32>) -> tensor<1x2x2x1xi32>
func.func @pooling_nhwc_i32_max_tensor(%input: tensor<1x4x4x1xi32>) -> tensor<1x2x2x1xi32> {
  %fake = tensor.empty() : tensor<3x3xi32>
  %init = tensor.empty() : tensor<1x2x2x1xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%init : tensor<1x2x2x1xi32>) -> tensor<1x2x2x1xi32>
  %res = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xi32>, tensor<3x3xi32>)
    outs(%fill: tensor<1x2x2x1xi32>) -> tensor<1x2x2x1xi32>
  return %res : tensor<1x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @pooling_nwc_i32_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_nwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x1xi32>, tensor<3xi32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x1xi32>) -> tensor<1x2x1xi32>
func.func @pooling_nwc_i32_max_tensor(%input: tensor<1x4x1xi32>) -> tensor<1x2x1xi32> {
  %fake = tensor.empty() : tensor<3xi32>
  %init = tensor.empty() : tensor<1x2x1xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%init : tensor<1x2x1xi32>) -> tensor<1x2x1xi32>
  %res = linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: tensor<1x4x1xi32>, tensor<3xi32>)
    outs(%fill: tensor<1x2x1xi32>) -> tensor<1x2x1xi32>
  return %res : tensor<1x2x1xi32>
}

// -----

// CHECK-LABEL: func @pooling_nhwc_i32_max
// CHECK:         linalg.pooling_nhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x1xi32>, memref<3x3xi32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x1xi32>)
func.func @pooling_nhwc_i32_max(%input: memref<1x4x4x1xi32>, %fake: memref<3x3xi32>, %output: memref<1x2x2x1xi32>) {
  linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x4x4x1xi32>, memref<3x3xi32>)
    outs(%output: memref<1x2x2x1xi32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nwc_i32_max
// CHECK:         linalg.pooling_nwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x1xi32>, memref<3xi32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x1xi32>)
func.func @pooling_nwc_i32_max(%input: memref<1x4x1xi32>, %fake: memref<3xi32>, %output: memref<1x2x1xi32>) {
  linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: memref<1x4x1xi32>, memref<3xi32>)
    outs(%output: memref<1x2x1xi32>)
  return
}


// -----

// CHECK-LABEL: func @pooling_nhwc_min_tensor
// CHECK:         %{{.+}} = linalg.pooling_nhwc_min
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x1xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
func.func @pooling_nhwc_min_tensor(%input: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %fake = tensor.empty() : tensor<3x3xf32>
  %init = tensor.empty() : tensor<1x2x2x1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  %res = linalg.pooling_nhwc_min {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xf32>, tensor<3x3xf32>)
    outs(%fill: tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  return %res : tensor<1x2x2x1xf32>
}

// -----

// CHECK-LABEL: func @pooling_nwc_min_tensor
// CHECK:         %{{.+}} = linalg.pooling_nwc_min
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x1xf32>, tensor<3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
func.func @pooling_nwc_min_tensor(%input: tensor<1x4x1xf32>) -> tensor<1x2x1xf32> {
  %fake = tensor.empty() : tensor<3xf32>
  %init = tensor.empty() : tensor<1x2x1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
  %res = linalg.pooling_nwc_min {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: tensor<1x4x1xf32>, tensor<3xf32>)
    outs(%fill: tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
  return %res : tensor<1x2x1xf32>
}

// -----

// CHECK-LABEL: func @pooling_nhwc_min
// CHECK:         linalg.pooling_nhwc_min
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x1xf32>, memref<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x1xf32>)
func.func @pooling_nhwc_min(%input: memref<1x4x4x1xf32>, %fake: memref<3x3xf32>, %output: memref<1x2x2x1xf32>) {
  linalg.pooling_nhwc_min {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x4x4x1xf32>, memref<3x3xf32>)
    outs(%output: memref<1x2x2x1xf32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nwc_min
// CHECK:         linalg.pooling_nwc_min
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:      strides = dense<1> : tensor<1xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x1xf32>, memref<3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x1xf32>)
func.func @pooling_nwc_min(%input: memref<1x4x1xf32>, %fake: memref<3xf32>, %output: memref<1x2x1xf32>) {
  linalg.pooling_nwc_min {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %fake: memref<1x4x1xf32>, memref<3xf32>)
    outs(%output: memref<1x2x1xf32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_ndhwc_sum_tensor
// CHECK:         %{{.+}} = linalg.pooling_ndhwc_sum
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>
// CHECK-SAME:      strides = dense<1> : tensor<3xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x4x1xf32>, tensor<3x3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
func.func @pooling_ndhwc_sum_tensor(%input: tensor<1x4x4x4x1xf32>) -> tensor<1x2x2x2x1xf32> {
  %fake = tensor.empty() : tensor<3x3x3xf32>
  %init = tensor.empty() : tensor<1x2x2x2x1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
  %res = linalg.pooling_ndhwc_sum {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
    ins(%input, %fake: tensor<1x4x4x4x1xf32>, tensor<3x3x3xf32>)
    outs(%fill: tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
  return %res : tensor<1x2x2x2x1xf32>
}

// -----

// CHECK-LABEL: func @pooling_ndhwc_sum
// CHECK:         linalg.pooling_ndhwc_sum
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>
// CHECK-SAME:      strides = dense<1> : tensor<3xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x4x1xf32>, memref<3x3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x2x1xf32>)
func.func @pooling_ndhwc_sum(%input: memref<1x4x4x4x1xf32>, %fake: memref<3x3x3xf32>, %output: memref<1x2x2x2x1xf32>) {
  linalg.pooling_ndhwc_sum {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
    ins(%input, %fake: memref<1x4x4x4x1xf32>, memref<3x3x3xf32>)
    outs(%output: memref<1x2x2x2x1xf32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_ndhwc_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_ndhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>
// CHECK-SAME:      strides = dense<1> : tensor<3xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x4x1xf32>, tensor<3x3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
func.func @pooling_ndhwc_max_tensor(%input: tensor<1x4x4x4x1xf32>) -> tensor<1x2x2x2x1xf32> {
  %fake = tensor.empty() : tensor<3x3x3xf32>
  %init = tensor.empty() : tensor<1x2x2x2x1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
  %res = linalg.pooling_ndhwc_max {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
    ins(%input, %fake: tensor<1x4x4x4x1xf32>, tensor<3x3x3xf32>)
    outs(%fill: tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
  return %res : tensor<1x2x2x2x1xf32>
}

// -----

// CHECK-LABEL: func @pooling_ndhwc_max
// CHECK:         linalg.pooling_ndhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>
// CHECK-SAME:      strides = dense<1> : tensor<3xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x4x1xf32>, memref<3x3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x2x1xf32>)
func.func @pooling_ndhwc_max(%input: memref<1x4x4x4x1xf32>, %fake: memref<3x3x3xf32>, %output: memref<1x2x2x2x1xf32>) {
  linalg.pooling_ndhwc_max {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
    ins(%input, %fake: memref<1x4x4x4x1xf32>, memref<3x3x3xf32>)
    outs(%output: memref<1x2x2x2x1xf32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_ndhwc_min_tensor
// CHECK:         %{{.+}} = linalg.pooling_ndhwc_min
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>
// CHECK-SAME:      strides = dense<1> : tensor<3xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x4x1xf32>, tensor<3x3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
func.func @pooling_ndhwc_min_tensor(%input: tensor<1x4x4x4x1xf32>) -> tensor<1x2x2x2x1xf32> {
  %fake = tensor.empty() : tensor<3x3x3xf32>
  %init = tensor.empty() : tensor<1x2x2x2x1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
  %res = linalg.pooling_ndhwc_min {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
    ins(%input, %fake: tensor<1x4x4x4x1xf32>, tensor<3x3x3xf32>)
    outs(%fill: tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
  return %res : tensor<1x2x2x2x1xf32>
}

// -----

// CHECK-LABEL: func @pooling_ndhwc_min
// CHECK:         linalg.pooling_ndhwc_min
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>
// CHECK-SAME:      strides = dense<1> : tensor<3xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x4x1xf32>, memref<3x3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x2x1xf32>)
func.func @pooling_ndhwc_min(%input: memref<1x4x4x4x1xf32>, %fake: memref<3x3x3xf32>, %output: memref<1x2x2x2x1xf32>) {
  linalg.pooling_ndhwc_min {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
    ins(%input, %fake: memref<1x4x4x4x1xf32>, memref<3x3x3xf32>)
    outs(%output: memref<1x2x2x2x1xf32>)
  return
}

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2, d2 * 2 + d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
func.func @conv_interface_wrong_input_indexing_map(
    %arg0 : tensor<?x?x?x?xf32>, %arg2 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // expected-error @+1 {{unexpected input index map for convolutions}}
  %0 = "linalg.conv_2d_nhwc_hwcf"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: f32, %arg4: f32, %arg5 : f32):
      %1 = "arith.mulf"(%arg3, %arg4) : (f32, f32) -> f32
      %2 = "arith.addf"(%arg5, %1) : (f32, f32) -> f32
      "linalg.yield"(%2) : (f32) -> ()
    }) {dilations = dense<1> : tensor<2xi64>, linalg.memoized_indexing_maps = [#map0, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>, strides = dense<2> : tensor<2xi64>} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3, d5 + 1)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
func.func @conv_interface_wrong_num_operands(
    %arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x?x?x?xf32>, %arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // expected-error @+1 {{expected output/filter indexing maps to be projected permutations}}
  %0 = "linalg.conv_2d_nhwc_hwcf"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: f32, %arg4: f32, %arg5 : f32):
      %1 = "arith.mulf"(%arg3, %arg4) : (f32, f32) -> f32
      %2 = "arith.addf"(%arg5, %1) : (f32, f32) -> f32
      "linalg.yield"(%2) : (f32) -> ()
    }) {dilations = dense<1> : tensor<2xi64>, linalg.memoized_indexing_maps = [#map0, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>, strides = dense<1> : tensor<2xi64>} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @batch_reduce_matmul(%arg0: tensor<8x128x256xf32>, %arg1: tensor<8x256x512xf32>, %arg2: tensor<128x512xf32>) -> tensor<128x512xf32> {
  // CHECK: %{{.+}} = linalg.batch_reduce_matmul
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<8x128x256xf32>, tensor<8x256x512xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<128x512xf32>) -> tensor<128x512xf32>
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<8x128x256xf32>, tensor<8x256x512xf32>) outs(%arg2: tensor<128x512xf32>) -> tensor<128x512xf32>
  return %0: tensor<128x512xf32>
}

// -----

func.func @batch_reduce_matmul(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?xf32>) {
  // CHECK: linalg.batch_reduce_matmul
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<?x?x?xf32>, memref<?x?x?xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<?x?xf32>)
  linalg.batch_reduce_matmul ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2: memref<?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @matmul_transpose_a
//       CHECK:   linalg.matmul_transpose_a
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : memref<5x3xf32>, memref<5x7xf32>)
//  CHECK-SAME:     outs(%{{.+}} : memref<3x7xf32>)
func.func @matmul_transpose_a(%arg0: memref<5x3xf32>, %arg1: memref<5x7xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul_transpose_a ins(%arg0, %arg1 : memref<5x3xf32>, memref<5x7xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// -----

// CHECK-LABEL: func @matmul_transpose_a_explicit
//       CHECK:   linalg.matmul
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : memref<5x3xf32>, memref<5x7xf32>)
//  CHECK-SAME:     outs(%{{.+}} : memref<3x7xf32>)
func.func @matmul_transpose_a_explicit(%arg0: memref<5x3xf32>, %arg1: memref<5x7xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
                      ins(%arg0, %arg1 : memref<5x3xf32>, memref<5x7xf32>)
                      outs(%arg2: memref<3x7xf32>)
  return
}

// -----

func.func @matmul_transpose_b_explicit(%arg0: memref<3x5xf32>, %arg1: memref<7x5xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
                      ins(%arg0, %arg1 : memref<3x5xf32>, memref<7x5xf32>)
                      outs(%arg2: memref<3x7xf32>)
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL:   func.func @matmul_transpose_b_explicit(
// CHECK-SAME:                                           %[[VAL_0:.*]]: memref<3x5xf32>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: memref<7x5xf32>,
// CHECK-SAME:                                           %[[VAL_2:.*]]: memref<3x7xf32>) {
// CHECK:           linalg.matmul ins(%[[VAL_0]], %[[VAL_1]] : memref<3x5xf32>, memref<7x5xf32>) outs(%[[VAL_2]] : memref<3x7xf32>) indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]]
// CHECK:           return
// CHECK:         }

// -----

func.func @matmul_transpose_a_b_explicit(%arg0: memref<5x3xf32>, %arg1: memref<7x5xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
                      ins(%arg0, %arg1 : memref<5x3xf32>, memref<7x5xf32>)
                      outs(%arg2: memref<3x7xf32>)
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL:   func.func @matmul_transpose_a_b_explicit(
// CHECK-SAME:                                             %[[VAL_0:.*]]: memref<5x3xf32>,
// CHECK-SAME:                                             %[[VAL_1:.*]]: memref<7x5xf32>,
// CHECK-SAME:                                             %[[VAL_2:.*]]: memref<3x7xf32>) {
// CHECK:           linalg.matmul ins(%[[VAL_0]], %[[VAL_1]] : memref<5x3xf32>, memref<7x5xf32>) outs(%[[VAL_2]] : memref<3x7xf32>) indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]]
// CHECK:           return
// CHECK:         }

// -----

func.func @matmul_bcast_a(%arg0: memref<5xf32>, %arg1: memref<5x7xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<5xf32>, memref<5x7xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @matmul_bcast_a
//       CHECK:   linalg.matmul
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : memref<5xf32>, memref<5x7xf32>)
//  CHECK-SAME:     outs(%{{.+}} : memref<3x7xf32>)

// -----

func.func @matmul_bcast_a_dim1(%arg0: memref<5xf32>, %arg1: memref<5x7xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<5xf32>, memref<5x7xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @matmul_bcast_a_dim1
//       CHECK:   linalg.matmul
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : memref<5xf32>, memref<5x7xf32>)
//  CHECK-SAME:     outs(%{{.+}} : memref<3x7xf32>)

// -----

func.func @matmul_bcast_b(%arg0: memref<3x5xf32>, %arg1: memref<5xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<3x5xf32>, memref<5xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @matmul_bcast_b
//       CHECK:   linalg.matmul
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : memref<3x5xf32>, memref<5xf32>)
//  CHECK-SAME:     outs(%{{.+}} : memref<3x7xf32>)

// -----

func.func @matmul_bcast_a_b(%arg0: memref<5xf32>, %arg1: memref<5xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<5xf32>, memref<5xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL:   func.func @matmul_bcast_a_b(
// CHECK-SAME:                                %[[VAL_0:.*]]: memref<5xf32>, %[[VAL_1:.*]]: memref<5xf32>,
// CHECK-SAME:                                %[[VAL_2:.*]]: memref<3x7xf32>) {
// CHECK:           linalg.matmul ins(%[[VAL_0]], %[[VAL_1]] : memref<5xf32>, memref<5xf32>) outs(%[[VAL_2]] : memref<3x7xf32>) indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_1]]]
// CHECK:           return
// CHECK:         }

// -----

func.func @matmul_bcast_b_dim1(%arg0: memref<3x5xf32>, %arg1: memref<5xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<3x5xf32>, memref<5xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @matmul_bcast_b_dim1
//       CHECK:   linalg.matmul
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : memref<3x5xf32>, memref<5xf32>)
//  CHECK-SAME:     outs(%{{.+}} : memref<3x7xf32>)

// -----

func.func @dynamic_matmul_bcast_a(%arg0: memref<?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<?xf32>, memref<?x?xf32>) outs(%arg2: memref<?x?xf32>)
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL:   func.func @dynamic_matmul_bcast_a(
// CHECK-SAME:                                      %[[VAL_0:.*]]: memref<?xf32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: memref<?x?xf32>,
// CHECK-SAME:                                      %[[VAL_2:.*]]: memref<?x?xf32>) {
// CHECK:           linalg.matmul ins(%[[VAL_0]], %[[VAL_1]] : memref<?xf32>, memref<?x?xf32>) outs(%[[VAL_2]] : memref<?x?xf32>) indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]]
// CHECK:           return
// CHECK:         }

// -----

func.func @matmul_bcast_a_transpose_b(%arg0: memref<5xf32>, %arg1: memref<7x5xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<5xf32>, memref<7x5xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL:   func.func @matmul_bcast_a_transpose_b(
// CHECK-SAME:                                  %[[VAL_0:.*]]: memref<5xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: memref<7x5xf32>,
// CHECK-SAME:                                  %[[VAL_2:.*]]: memref<3x7xf32>) {
// CHECK:           linalg.matmul ins(%[[VAL_0]], %[[VAL_1]] : memref<5xf32>, memref<7x5xf32>) outs(%[[VAL_2]] : memref<3x7xf32>) indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]]
// CHECK:           return
// CHECK:         }

// -----

func.func @matmul_bcast_b_transpose_a(%arg0: memref<5x3xf32>, %arg1: memref<5xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<5x3xf32>, memref<5xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL:   func.func @matmul_bcast_b_transpose_a(
// CHECK-SAME:                                          %[[VAL_0:.*]]: memref<5x3xf32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: memref<5xf32>,
// CHECK-SAME:                                          %[[VAL_2:.*]]: memref<3x7xf32>) {
// CHECK:           linalg.matmul ins(%[[VAL_0]], %[[VAL_1]] : memref<5x3xf32>, memref<5xf32>) outs(%[[VAL_2]] : memref<3x7xf32>) indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]]
// CHECK:           return
// CHECK:         }

// -----

// CHECK-LABEL: func @matmul_transpose_b
//       CHECK:   linalg.matmul_transpose_b
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : memref<3x5xf32>, memref<7x5xf32>)
//  CHECK-SAME:     outs(%{{.+}} : memref<3x7xf32>)
func.func @matmul_transpose_b(%arg0: memref<3x5xf32>, %arg1: memref<7x5xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul_transpose_b ins(%arg0, %arg1 : memref<3x5xf32>, memref<7x5xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// -----

// CHECK-LABEL: func @batchmatmul_transpose_a
//       CHECK:   linalg.batch_matmul_transpose_a
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : memref<2x5x3xf32>, memref<2x5x7xf32>)
//  CHECK-SAME:     outs(%{{.+}} : memref<2x3x7xf32>)
func.func @batchmatmul_transpose_a(%arg0: memref<2x5x3xf32>, %arg1: memref<2x5x7xf32>, %arg2: memref<2x3x7xf32>) {
  linalg.batch_matmul_transpose_a ins(%arg0, %arg1 : memref<2x5x3xf32>, memref<2x5x7xf32>) outs(%arg2: memref<2x3x7xf32>)
  return
}

// -----

// CHECK-LABEL: func @batchmatmul_transpose_b
//       CHECK:   linalg.batch_matmul_transpose_b
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : memref<2x3x5xf32>, memref<2x7x5xf32>)
//  CHECK-SAME:     outs(%{{.+}} : memref<2x3x7xf32>)
func.func @batchmatmul_transpose_b(%arg0: memref<2x3x5xf32>, %arg1: memref<2x7x5xf32>, %arg2: memref<2x3x7xf32>) {
  linalg.batch_matmul_transpose_b ins(%arg0, %arg1 : memref<2x3x5xf32>, memref<2x7x5xf32>) outs(%arg2: memref<2x3x7xf32>)
  return
}

// -----

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func @contract
//       CHECK:   linalg.contract
//  CHECK-SAME:     indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]]
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : memref<2x3x5xf32>, memref<2x5x7xf32>)
//  CHECK-SAME:     outs(%{{.+}} : memref<2x3x7xf32>)
func.func @contract(
    %A: memref<2x3x5xf32>, %B: memref<2x5x7xf32>, %C: memref<2x3x7xf32>) {
  linalg.contract
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
      ins(%A, %B : memref<2x3x5xf32>, memref<2x5x7xf32>)
      outs(%C: memref<2x3x7xf32>)
  return
}

// -----

// CHECK: #[[$ACCESS_A:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ACCESS_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$ACCESS_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @contract_matmul_bcast_a
func.func @contract_matmul_bcast_a(%A: memref<5xf32>, %B: memref<5x7xf32>, %C: memref<3x7xf32>) {
// CHECK:  linalg.contract
// CHECK-SAME: indexing_maps = [#[[$ACCESS_A]], #[[$ACCESS_B]], #[[$ACCESS_C]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<5xf32>, memref<5x7xf32>)
// CHECK-SAME: outs(%{{.+}} : memref<3x7xf32>)
  linalg.contract
      indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%A, %B : memref<5xf32>, memref<5x7xf32>)
      outs(%C: memref<3x7xf32>)
  return
}

// -----

// CHECK: #[[$ACCESS_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ACCESS_B:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ACCESS_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @contract_matmul_bcast_b
func.func @contract_matmul_bcast_b(%A: memref<3x5xf32>, %B: memref<5xf32>, %C: memref<3x7xf32>) {
// CHECK:  linalg.contract
// CHECK-SAME: indexing_maps = [#[[$ACCESS_A]], #[[$ACCESS_B]], #[[$ACCESS_C]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<3x5xf32>, memref<5xf32>)
// CHECK-SAME: outs(%{{.+}} : memref<3x7xf32>)
  linalg.contract
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%A, %B : memref<3x5xf32>, memref<5xf32>)
      outs(%C: memref<3x7xf32>)
  return
}

// -----

// CHECK: #[[$ACCESS_A:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ACCESS_B:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @contract_matmul_bcast_a_b
func.func @contract_matmul_bcast_a_b(
    %A: memref<5xf32>, %B: memref<5xf32>, %C: memref<3x7xf32>) {
// CHECK:  linalg.contract
// CHECK-SAME: indexing_maps = [#[[$ACCESS_A]], #[[$ACCESS_A]], #[[$ACCESS_B]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<5xf32>, memref<5xf32>)
// CHECK-SAME: outs(%{{.+}} : memref<3x7xf32>)
  linalg.contract
      indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%A, %B : memref<5xf32>, memref<5xf32>)
      outs(%C: memref<3x7xf32>)
  return
}

// -----

// CHECK: #[[$ACCESS_A:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ACCESS_B:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #[[$ACCESS_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @contract_matmul_bcast_a_transpose_b
func.func @contract_matmul_bcast_a_transpose_b(
    %A: memref<5xf32>, %B: memref<7x5xf32>, %C: memref<3x7xf32>) {
// CHECK:  linalg.contract
// CHECK-SAME: indexing_maps = [#[[$ACCESS_A]], #[[$ACCESS_B]], #[[$ACCESS_C]]]
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<5xf32>, memref<7x5xf32>)
// CHECK-SAME: outs(%{{.+}} : memref<3x7xf32>)
  linalg.contract
      indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%A, %B : memref<5xf32>, memref<7x5xf32>)
      outs(%C: memref<3x7xf32>)
  return
}

// -----

// CHECK: #[[$ACCESS_A:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK: #[[$ACCESS_B:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ACCESS_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL:   func.func @contract_matmul_bcast_b_transpose_a
func.func @contract_matmul_bcast_b_transpose_a(%A: memref<5x3xf32>, %B: memref<5xf32>, %C: memref<3x7xf32>) {
// CHECK:      linalg.contract
// CHECK-SAME:     indexing_maps = [#[[$ACCESS_A]], #[[$ACCESS_B]], #[[$ACCESS_C]]]
// CHECK-SAME:     ins(%{{.+}}, %{{.+}} : memref<5x3xf32>, memref<5xf32>)
// CHECK-SAME:     outs(%{{.+}} : memref<3x7xf32>)
  linalg.contract
      indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%A, %B : memref<5x3xf32>, memref<5xf32>)
      outs(%C: memref<3x7xf32>)
  return
}

// -----

// CHECK-LABEL: func @mmt4d
func.func @mmt4d(%A: tensor<10x32x8x1xf32>, %B: tensor<80x32x4x1xf32>, %C: tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32> {
  // CHECK: %{{.+}} = linalg.mmt4d
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<10x32x8x1xf32>, tensor<80x32x4x1xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
  %0 = linalg.mmt4d ins(%A, %B : tensor<10x32x8x1xf32>, tensor<80x32x4x1xf32>) outs(%C: tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
  return %0: tensor<10x80x8x4xf32>
}

// -----

// CHECK-LABEL: func @batch_mmt4d
func.func @batch_mmt4d(%arg0: tensor<128x10x32x8x1xf32>, %arg1: tensor<128x80x32x4x1xf32>, %arg2: tensor<128x10x80x8x4xf32>) -> tensor<128x10x80x8x4xf32> {
  // CHECK: %{{.+}} = linalg.batch_mmt4d
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<128x10x32x8x1xf32>, tensor<128x80x32x4x1xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<128x10x80x8x4xf32>) -> tensor<128x10x80x8x4xf32>
  %0 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<128x10x32x8x1xf32>, tensor<128x80x32x4x1xf32>) outs(%arg2 : tensor<128x10x80x8x4xf32>) -> tensor<128x10x80x8x4xf32>
  return %0: tensor<128x10x80x8x4xf32>
}

// -----

// CHECK-LABEL: func @add_dynamic
func.func @add_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  // CHECK: linalg.add
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<?x?x?xf32>, memref<?x?x?xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.add ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @add_static
func.func @add_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: linalg.add
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<4x8x16xf32>, memref<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.add ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @add_tensor
func.func @add_tensor(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.add
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<4x8x16xf32>, tensor<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.add ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @sub_dynamic
func.func @sub_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  // CHECK: linalg.sub
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<?x?x?xf32>, memref<?x?x?xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.sub ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @sub_static
func.func @sub_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: linalg.sub
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<4x8x16xf32>, memref<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.sub ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @sub_tensor
func.func @sub_tensor(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.sub
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<4x8x16xf32>, tensor<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.sub ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @mul_dynamic
func.func @mul_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  // CHECK: linalg.mul
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<?x?x?xf32>, memref<?x?x?xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.mul ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @mul_static
func.func @mul_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: linalg.mul
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<4x8x16xf32>, memref<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.mul ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @mul_tensor
func.func @mul_tensor(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.mul
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<4x8x16xf32>, tensor<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.mul ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @div_dynamic
func.func @div_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  // CHECK: linalg.div
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<?x?x?xf32>, memref<?x?x?xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.div ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @div_static
func.func @div_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: linalg.div
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<4x8x16xf32>, memref<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.div ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @div_tensor
func.func @div_tensor(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.div
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<4x8x16xf32>, tensor<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.div ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @div_unsigned_dynamic
func.func @div_unsigned_dynamic(%arg0: memref<?x?x?xi32>, %arg1: memref<?x?x?xi32>, %arg2: memref<?x?x?xi32>) {
  // CHECK: linalg.div_unsigned
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<?x?x?xi32>, memref<?x?x?xi32>)
  // CHECK-SAME: outs(%{{.+}} : memref<?x?x?xi32>)
  linalg.div_unsigned ins(%arg0, %arg1 : memref<?x?x?xi32>, memref<?x?x?xi32>) outs(%arg2: memref<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: func @div_unsigned_static
func.func @div_unsigned_static(%arg0: memref<4x8x16xi32>, %arg1: memref<4x8x16xi32>, %arg2: memref<4x8x16xi32>) {
  // CHECK: linalg.div_unsigned
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<4x8x16xi32>, memref<4x8x16xi32>)
  // CHECK-SAME: outs(%{{.+}} : memref<4x8x16xi32>)
  linalg.div_unsigned ins(%arg0, %arg1 : memref<4x8x16xi32>, memref<4x8x16xi32>) outs(%arg2: memref<4x8x16xi32>)
  return
}

// -----

// CHECK-LABEL: func @div_unsigned_tensor
func.func @div_unsigned_tensor(%arg0: tensor<4x8x16xi32>, %arg1: tensor<4x8x16xi32>) -> tensor<4x8x16xi32> {
  %0 = tensor.empty() : tensor<4x8x16xi32>
  // CHECK: linalg.div_unsigned
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<4x8x16xi32>, tensor<4x8x16xi32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<4x8x16xi32>)
  %1 = linalg.div_unsigned ins(%arg0, %arg1 : tensor<4x8x16xi32>, tensor<4x8x16xi32>) outs(%0: tensor<4x8x16xi32>) -> tensor<4x8x16xi32>
  return %1 : tensor<4x8x16xi32>
}

// -----

// CHECK-LABEL: func @exp_dynamic
func.func @exp_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
  // CHECK: linalg.exp
  // CHECK-SAME: ins(%{{.+}} : memref<?x?x?xf32>) outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.exp ins(%arg0 : memref<?x?x?xf32>) outs(%arg1: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @exp_static
func.func @exp_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
  // CHECK: linalg.exp
  // CHECK-SAME: ins(%{{.+}} : memref<4x8x16xf32>) outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.exp ins(%arg0 : memref<4x8x16xf32>) outs(%arg1: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @exp_tensor
func.func @exp_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.exp
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8x16xf32>) outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.exp ins(%arg0 : tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @log_dynamic
func.func @log_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
  // CHECK: linalg.log
  // CHECK-SAME: ins(%{{.+}} : memref<?x?x?xf32>) outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.log ins(%arg0 : memref<?x?x?xf32>) outs(%arg1: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @log_static
func.func @log_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
  // CHECK: linalg.log
  // CHECK-SAME: ins(%{{.+}} : memref<4x8x16xf32>) outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.log ins(%arg0 : memref<4x8x16xf32>) outs(%arg1: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @log_tensor
func.func @log_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.log
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8x16xf32>) outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.log ins(%arg0 : tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @abs_dynamic
func.func @abs_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
  // CHECK: linalg.abs
  // CHECK-SAME: ins(%{{.+}} : memref<?x?x?xf32>) outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.abs ins(%arg0 : memref<?x?x?xf32>) outs(%arg1: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @abs_static
func.func @abs_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
  // CHECK: linalg.abs
  // CHECK-SAME: ins(%{{.+}} : memref<4x8x16xf32>) outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.abs ins(%arg0 : memref<4x8x16xf32>) outs(%arg1: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @abs_tensor
func.func @abs_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.abs
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8x16xf32>) outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.abs ins(%arg0 : tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @ceil_dynamic
func.func @ceil_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
  // CHECK: linalg.ceil
  // CHECK-SAME: ins(%{{.+}} : memref<?x?x?xf32>) outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.ceil ins(%arg0 : memref<?x?x?xf32>) outs(%arg1: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @ceil_static
func.func @ceil_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
  // CHECK: linalg.ceil
  // CHECK-SAME: ins(%{{.+}} : memref<4x8x16xf32>) outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.ceil ins(%arg0 : memref<4x8x16xf32>) outs(%arg1: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @ceil_tensor
func.func @ceil_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.ceil
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8x16xf32>) outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.ceil ins(%arg0 : tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @floor_dynamic
func.func @floor_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
  // CHECK: linalg.floor
  // CHECK-SAME: ins(%{{.+}} : memref<?x?x?xf32>) outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.floor ins(%arg0 : memref<?x?x?xf32>) outs(%arg1: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @floor_static
func.func @floor_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
  // CHECK: linalg.floor
  // CHECK-SAME: ins(%{{.+}} : memref<4x8x16xf32>) outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.floor ins(%arg0 : memref<4x8x16xf32>) outs(%arg1: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @floor_tensor
func.func @floor_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.floor
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8x16xf32>) outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.floor ins(%arg0 : tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @negf_dynamic
func.func @negf_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
  // CHECK: linalg.negf
  // CHECK-SAME: ins(%{{.+}} : memref<?x?x?xf32>) outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.negf ins(%arg0 : memref<?x?x?xf32>) outs(%arg1: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @negf_static
func.func @negf_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
  // CHECK: linalg.negf
  // CHECK-SAME: ins(%{{.+}} : memref<4x8x16xf32>) outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.negf ins(%arg0 : memref<4x8x16xf32>) outs(%arg1: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @negf_tensor
func.func @negf_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.negf
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8x16xf32>) outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.negf ins(%arg0 : tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @reciprocal_dynamic
func.func @reciprocal_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
  // CHECK: linalg.reciprocal
  // CHECK-SAME: ins(%{{.+}} : memref<?x?x?xf32>) outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.reciprocal ins(%arg0 : memref<?x?x?xf32>) outs(%arg1: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @reciprocal_static
func.func @reciprocal_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
  // CHECK: linalg.reciprocal
  // CHECK-SAME: ins(%{{.+}} : memref<4x8x16xf32>) outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.reciprocal ins(%arg0 : memref<4x8x16xf32>) outs(%arg1: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @reciprocal_tensor
func.func @reciprocal_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.reciprocal
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8x16xf32>) outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.reciprocal ins(%arg0 : tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @round_dynamic
func.func @round_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
  // CHECK: linalg.round
  // CHECK-SAME: ins(%{{.+}} : memref<?x?x?xf32>) outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.round ins(%arg0 : memref<?x?x?xf32>) outs(%arg1: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @round_static
func.func @round_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
  // CHECK: linalg.round
  // CHECK-SAME: ins(%{{.+}} : memref<4x8x16xf32>) outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.round ins(%arg0 : memref<4x8x16xf32>) outs(%arg1: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @round_tensor
func.func @round_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.round
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8x16xf32>) outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.round ins(%arg0 : tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @sqrt_dynamic
func.func @sqrt_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
  // CHECK: linalg.sqrt
  // CHECK-SAME: ins(%{{.+}} : memref<?x?x?xf32>) outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.sqrt ins(%arg0 : memref<?x?x?xf32>) outs(%arg1: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @sqrt_static
func.func @sqrt_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
  // CHECK: linalg.sqrt
  // CHECK-SAME: ins(%{{.+}} : memref<4x8x16xf32>) outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.sqrt ins(%arg0 : memref<4x8x16xf32>) outs(%arg1: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @sqrt_tensor
func.func @sqrt_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.sqrt
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8x16xf32>) outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.sqrt ins(%arg0 : tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @rsqrt_dynamic
func.func @rsqrt_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
  // CHECK: linalg.rsqrt
  // CHECK-SAME: ins(%{{.+}} : memref<?x?x?xf32>) outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.rsqrt ins(%arg0 : memref<?x?x?xf32>) outs(%arg1: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @rsqrt_static
func.func @rsqrt_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
  // CHECK: linalg.rsqrt
  // CHECK-SAME: ins(%{{.+}} : memref<4x8x16xf32>) outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.rsqrt ins(%arg0 : memref<4x8x16xf32>) outs(%arg1: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @rsqrt_tensor
func.func @rsqrt_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.rsqrt
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8x16xf32>) outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.rsqrt ins(%arg0 : tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @square_dynamic
func.func @square_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
  // CHECK: linalg.square
  // CHECK-SAME: ins(%{{.+}} : memref<?x?x?xf32>) outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.square ins(%arg0 : memref<?x?x?xf32>) outs(%arg1: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @square_static
func.func @square_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
  // CHECK: linalg.square
  // CHECK-SAME: ins(%{{.+}} : memref<4x8x16xf32>) outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.square ins(%arg0 : memref<4x8x16xf32>) outs(%arg1: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @square_tensor
func.func @square_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.square
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8x16xf32>) outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.square ins(%arg0 : tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @tanh_dynamic
func.func @tanh_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
  // CHECK: linalg.tanh
  // CHECK-SAME: ins(%{{.+}} : memref<?x?x?xf32>) outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.tanh ins(%arg0 : memref<?x?x?xf32>) outs(%arg1: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @tanh_static
func.func @tanh_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
  // CHECK: linalg.tanh
  // CHECK-SAME: ins(%{{.+}} : memref<4x8x16xf32>) outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.tanh ins(%arg0 : memref<4x8x16xf32>) outs(%arg1: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @tanh_tensor
func.func @tanh_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.tanh
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8x16xf32>) outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.tanh ins(%arg0 : tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @erf_dynamic
func.func @erf_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
  // CHECK: linalg.erf
  // CHECK-SAME: ins(%{{.+}} : memref<?x?x?xf32>) outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.erf ins(%arg0 : memref<?x?x?xf32>) outs(%arg1: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @erf_static
func.func @erf_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
  // CHECK: linalg.erf
  // CHECK-SAME: ins(%{{.+}} : memref<4x8x16xf32>) outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.erf ins(%arg0 : memref<4x8x16xf32>) outs(%arg1: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @erf_tensor
func.func @erf_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.erf
  // CHECK-SAME: ins(%{{.+}} : tensor<4x8x16xf32>) outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.erf ins(%arg0 : tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @max_dynamic
func.func @max_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  // CHECK: linalg.max
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<?x?x?xf32>, memref<?x?x?xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.max ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @max_static
func.func @max_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: linalg.max
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<4x8x16xf32>, memref<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.max ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @max_tensor
func.func @max_tensor(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.max
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<4x8x16xf32>, tensor<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.max ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @min_dynamic
func.func @min_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  // CHECK: linalg.min
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<?x?x?xf32>, memref<?x?x?xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.min ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @min_static
func.func @min_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: linalg.min
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<4x8x16xf32>, memref<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.min ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @min_tensor
func.func @min_tensor(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.min
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<4x8x16xf32>, tensor<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.min ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @powf_dynamic
func.func @powf_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  // CHECK: linalg.powf
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<?x?x?xf32>, memref<?x?x?xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.powf ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @powf_static
func.func @powf_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
  // CHECK: linalg.powf
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<4x8x16xf32>, memref<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.powf ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @powf_tensor
func.func @powf_tensor(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.powf
  // CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<4x8x16xf32>, tensor<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.powf ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: func @fill_tensor
func.func @fill_tensor(%f: f32, %v: vector<2x4xf32>) -> (tensor<f32>, tensor<vector<2x4xf32>>) {
  %e0 = tensor.empty() : tensor<f32>
  %0 = linalg.fill ins(%f : f32) outs(%e0 : tensor<f32>) -> tensor<f32>
  %e1 = tensor.empty() : tensor<vector<2x4xf32>>
  %1 = linalg.fill ins(%v : vector<2x4xf32>) outs(%e1 : tensor<vector<2x4xf32>>) -> tensor<vector<2x4xf32>>
  return %0, %1: tensor<f32>, tensor<vector<2x4xf32>>
}

// -----

// CHECK-LABEL: func @select_dynamic
func.func @select_dynamic(%arg0: memref<?x?x?xi1>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>, %arg3: memref<?x?x?xf32>) {
  // CHECK: linalg.select
  // CHECK-SAME: ins(%{{.+}}, %{{.+}}, %{{.+}} : memref<?x?x?xi1>, memref<?x?x?xf32>, memref<?x?x?xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.select ins(%arg0, %arg1, %arg2 : memref<?x?x?xi1>, memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg3: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @select_static
func.func @select_static(%arg0: memref<4x8x16xi1>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>, %arg3: memref<4x8x16xf32>) {
  // CHECK: linalg.select
  // CHECK-SAME: ins(%{{.+}}, %{{.+}}, %{{.+}} : memref<4x8x16xi1>, memref<4x8x16xf32>, memref<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : memref<4x8x16xf32>)
  linalg.select ins(%arg0, %arg1, %arg2 : memref<4x8x16xi1>, memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg3: memref<4x8x16xf32>)
  return
}

// -----

// CHECK-LABEL: func @select_tensor
func.func @select_tensor(%arg0: tensor<4x8x16xi1>, %arg1: tensor<4x8x16xf32>, %arg2: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = tensor.empty() : tensor<4x8x16xf32>
  // CHECK: linalg.select
  // CHECK-SAME: ins(%{{.+}}, %{{.+}}, %{{.+}} : tensor<4x8x16xi1>, tensor<4x8x16xf32>, tensor<4x8x16xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<4x8x16xf32>)
  %1 = linalg.select ins(%arg0, %arg1, %arg2 : tensor<4x8x16xi1>, tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %1 : tensor<4x8x16xf32>
}
