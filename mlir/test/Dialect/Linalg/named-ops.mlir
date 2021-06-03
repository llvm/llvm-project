// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @depthwise_conv_2d_input_nhwc_filter_hwcf_tensor
func @depthwise_conv_2d_input_nhwc_filter_hwcf_tensor(%input: tensor<2x4x5x2xf32>, %filter: tensor<2x2x2x3xf32>) -> tensor<2x3x4x2x3xf32> {
  %zero = constant 0.000000e+00 : f32
  %init = linalg.init_tensor [2, 3, 4, 2, 3] : tensor<2x3x4x2x3xf32>
  %fill = linalg.fill(%init, %zero) : tensor<2x3x4x2x3xf32>, f32 -> tensor<2x3x4x2x3xf32>
  // CHECK:      %{{.+}} = linalg.depthwise_conv_2d_input_nhwc_filter_hwcf
  // CHECK-SAME:   {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<2x4x5x2xf32>, tensor<2x2x2x3xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<2x3x4x2x3xf32>)
  %0 = linalg.depthwise_conv_2d_input_nhwc_filter_hwcf
     { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%input, %filter : tensor<2x4x5x2xf32>, tensor<2x2x2x3xf32>)
    outs(%fill : tensor<2x3x4x2x3xf32>) -> tensor<2x3x4x2x3xf32>
  return %0 : tensor<2x3x4x2x3xf32>
}

// CHECK-LABEL: func @depthwise_conv_2d_input_nhwc_filter_hwcf_memref
func @depthwise_conv_2d_input_nhwc_filter_hwcf_memref(%input: memref<2x4x5x2xf32>, %filter: memref<2x2x2x3xf32>, %output: memref<2x3x4x2x3xf32>) {
  // CHECK:      linalg.depthwise_conv_2d_input_nhwc_filter_hwcf
  // CHECK-SAME:   {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<2x4x5x2xf32>, memref<2x2x2x3xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<2x3x4x2x3xf32>)
  linalg.depthwise_conv_2d_input_nhwc_filter_hwcf
     { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%input, %filter : memref<2x4x5x2xf32>, memref<2x2x2x3xf32>)
    outs(%output : memref<2x3x4x2x3xf32>)
  return
}

// CHECK-LABEL: func @depthwise_conv_2d_input_nhwc_filter_hwc_tensor
func @depthwise_conv_2d_input_nhwc_filter_hwc_tensor(%input: tensor<1x113x113x96xf32>, %filter: tensor<3x3x96xf32>) -> tensor<1x56x56x96xf32> {
  %init = linalg.init_tensor [1, 56, 56, 96] : tensor<1x56x56x96xf32>
  // CHECK:      %{{.+}} = linalg.depthwise_conv_2d_input_nhwc_filter_hwc
  // CHECK-SAME:   {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  %0 = linalg.depthwise_conv_2d_input_nhwc_filter_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
         ins(%input, %filter: tensor<1x113x113x96xf32>, tensor<3x3x96xf32>)
         outs(%init: tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  return %0: tensor<1x56x56x96xf32>
}

// CHECK-LABEL: func @depthwise_conv_2d_input_nhwc_filter_hwc_memref
func @depthwise_conv_2d_input_nhwc_filter_hwc_memref(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // CHECK:      linalg.depthwise_conv_2d_input_nhwc_filter_hwc
  // CHECK-SAME:   {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<1x113x113x96xf32>, memref<3x3x96xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<1x56x56x96xf32>)
  linalg.depthwise_conv_2d_input_nhwc_filter_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

func @depthwise_conv_2d_input_nhwc_filter_hwcf_tensor_dilated(%input: tensor<2x8x9x2xf32>, %filter: tensor<2x2x2x3xf32>) -> tensor<2x6x7x2x3xf32> {
  %zero = constant 0.000000e+00 : f32
  %init = linalg.init_tensor [2, 6, 7, 2, 3] : tensor<2x6x7x2x3xf32>
  %fill = linalg.fill(%init, %zero) : tensor<2x6x7x2x3xf32>, f32 -> tensor<2x6x7x2x3xf32>
  // CHECK:      %{{.+}} = linalg.depthwise_conv_2d_input_nhwc_filter_hwcf
  // CHECK-SAME:   {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<2x8x9x2xf32>, tensor<2x2x2x3xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<2x6x7x2x3xf32>)
  %0 = linalg.depthwise_conv_2d_input_nhwc_filter_hwcf
     { dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%input, %filter : tensor<2x8x9x2xf32>, tensor<2x2x2x3xf32>)
    outs(%fill : tensor<2x6x7x2x3xf32>) -> tensor<2x6x7x2x3xf32>
  return %0 : tensor<2x6x7x2x3xf32>
}

// CHECK-LABEL: func @depthwise_conv_2d_input_nhwc_filter_hwcf_memref_dilated
func @depthwise_conv_2d_input_nhwc_filter_hwcf_memref_dilated(%input: memref<2x8x9x2xf32>, %filter: memref<2x2x2x3xf32>, %output: memref<2x6x7x2x3xf32>) {
  // CHECK:      linalg.depthwise_conv_2d_input_nhwc_filter_hwcf
  // CHECK-SAME:   {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<2x8x9x2xf32>, memref<2x2x2x3xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<2x6x7x2x3xf32>)
  linalg.depthwise_conv_2d_input_nhwc_filter_hwcf
     { dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%input, %filter : memref<2x8x9x2xf32>, memref<2x2x2x3xf32>)
    outs(%output : memref<2x6x7x2x3xf32>)
  return
}

// -----

func @depthwise_conv_2d_input_nhwc_filter_missing_stride(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // expected-error @+1 {{missing indexing map required attribute 'strides'}}
  linalg.depthwise_conv_2d_input_nhwc_filter_hwc {dilations = dense<1> : vector<2xi64>}
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

// -----

func @depthwise_conv_2d_input_nhwc_filter_missing_dilations(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // expected-error @+1 {{missing indexing map required attribute 'dilations'}}
  linalg.depthwise_conv_2d_input_nhwc_filter_hwc {strides = dense<1> : vector<2xi64>}
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

// -----

func @depthwise_conv_2d_input_nhwc_filter_wrong_stride_element_type(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // expected-error @+1 {{incorrect element type for indexing map required attribute 'strides'}}
  linalg.depthwise_conv_2d_input_nhwc_filter_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2.0> : vector<2xf32>}
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

// -----

func @depthwise_conv_2d_input_nhwc_filter_wrong_stride_size(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  // expected-error @+1 {{incorrect shape for indexing map required attribute 'strides'}}
  linalg.depthwise_conv_2d_input_nhwc_filter_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<3xi64> }
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

// -----

// CHECK-LABEL: func @conv_1d_input_nwc_filter_wcf
func @conv_1d_input_nwc_filter_wcf(%input: tensor<?x?x?xf32>, %filter: tensor<?x?x?xf32>, %init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK:      %{{.+}} = linalg.conv_1d_input_nwc_filter_wcf
  // CHECK-SAME:   dilations = dense<1> : tensor<1xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<1xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %0 = linalg.conv_1d_input_nwc_filter_wcf {dilations = dense<1> : tensor<1xi64>,
                                            strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs (%init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @conv_1d_input_nwc_filter_wcf
func @conv_1d_input_nwc_filter_wcf(%input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?xf32>) {
  // CHECK:      linalg.conv_1d_input_nwc_filter_wcf
  // CHECK-SAME:   dilations = dense<1> : tensor<1xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<1xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<?x?x?xf32>, memref<?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.conv_1d_input_nwc_filter_wcf {dilations = dense<1> : tensor<1xi64>,
                                       strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: memref<?x?x?xf32>, memref<?x?x?xf32>)
    outs (%output: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @conv_1d_input_ncw_filter_wcf
func @conv_1d_input_ncw_filter_wcf(%input: tensor<?x?x?xf32>, %filter: tensor<?x?x?xf32>, %init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK:      %{{.+}} = linalg.conv_1d_input_ncw_filter_wcf
  // CHECK-SAME:   dilations = dense<1> : tensor<1xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<1xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %0 = linalg.conv_1d_input_ncw_filter_wcf {dilations = dense<1> : tensor<1xi64>,
                                            strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs (%init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @conv_1d_input_ncw_filter_wcf
func @conv_1d_input_ncw_filter_wcf(%input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?xf32>) {
  // CHECK:      linalg.conv_1d_input_ncw_filter_wcf
  // CHECK-SAME:   dilations = dense<1> : tensor<1xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<1xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<?x?x?xf32>, memref<?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<?x?x?xf32>)
  linalg.conv_1d_input_ncw_filter_wcf {dilations = dense<1> : tensor<1xi64>,
                                       strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: memref<?x?x?xf32>, memref<?x?x?xf32>)
    outs (%output: memref<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @conv_2d_input_nhwc_filter_hwcf
func @conv_2d_input_nhwc_filter_hwcf(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK:      %{{.+}} = linalg.conv_2d_input_nhwc_filter_hwcf
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %0 = linalg.conv_2d_input_nhwc_filter_hwcf {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @conv_2d_input_nhwc_filter_hwcf
func @conv_2d_input_nhwc_filter_hwcf(%input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32>) {
  // CHECK:      linalg.conv_2d_input_nhwc_filter_hwcf
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<?x?x?x?xf32>)
  linalg.conv_2d_input_nhwc_filter_hwcf {dilations = dense<1> : tensor<2xi64>,
                                         strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
    outs (%output: memref<?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @conv_2d_input_nchw_filter_hwcf
func @conv_2d_input_nchw_filter_hwcf(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK:      %{{.+}} = linalg.conv_2d_input_nchw_filter_hwcf
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %0 = linalg.conv_2d_input_nchw_filter_hwcf {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @conv_2d_input_nchw_filter_hwcf
func @conv_2d_input_nchw_filter_hwcf(%input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32>) {
  // CHECK:      linalg.conv_2d_input_nchw_filter_hwcf
  // CHECK-SAME:   dilations = dense<1> : tensor<2xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<2xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<?x?x?x?xf32>)
  linalg.conv_2d_input_nchw_filter_hwcf {dilations = dense<1> : tensor<2xi64>,
                                         strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
    outs (%output: memref<?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @conv_3d_input_ndhwc_filter_dhwcf
func @conv_3d_input_ndhwc_filter_dhwcf(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  // CHECK:      %{{.+}} = linalg.conv_3d_input_ndhwc_filter_dhwcf
  // CHECK-SAME:   dilations = dense<1> : tensor<3xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<3xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  %0 = linalg.conv_3d_input_ndhwc_filter_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @conv_3d_input_ndhwc_filter_dhwcf
func @conv_3d_input_ndhwc_filter_dhwcf(%input: memref<?x?x?x?x?xf32>, %filter: memref<?x?x?x?x?xf32>, %output: memref<?x?x?x?x?xf32>) {
  // CHECK:      linalg.conv_3d_input_ndhwc_filter_dhwcf
  // CHECK-SAME:   dilations = dense<1> : tensor<3xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<3xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<?x?x?x?x?xf32>)
  linalg.conv_3d_input_ndhwc_filter_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                           strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
    outs (%output: memref<?x?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: func @conv_3d_input_ncdhw_filter_dhwcf
func @conv_3d_input_ncdhw_filter_dhwcf(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  // CHECK:      %{{.+}} = linalg.conv_3d_input_ncdhw_filter_dhwcf
  // CHECK-SAME:   dilations = dense<1> : tensor<3xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<3xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  %0 = linalg.conv_3d_input_ncdhw_filter_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @conv_3d_input_ncdhw_filter_dhwcf
func @conv_3d_input_ncdhw_filter_dhwcf(%input: memref<?x?x?x?x?xf32>, %filter: memref<?x?x?x?x?xf32>, %output: memref<?x?x?x?x?xf32>) {
  // CHECK:      linalg.conv_3d_input_ncdhw_filter_dhwcf
  // CHECK-SAME:   dilations = dense<1> : tensor<3xi64>
  // CHECK-SAME:   strides = dense<1> : tensor<3xi64>
  // CHECK-SAME:   ins(%{{.+}}, %{{.+}} : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
  // CHECK-SAME:   outs(%{{.+}} : memref<?x?x?x?x?xf32>)
  linalg.conv_3d_input_ncdhw_filter_dhwcf {dilations = dense<1> : tensor<3xi64>,
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
func @pooling_nhwc_sum_tensor(%input: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %fake = linalg.init_tensor [3, 3] : tensor<3x3xf32>
  %init = linalg.init_tensor [1, 2, 2, 1] : tensor<1x2x2x1xf32>
  %cst = constant 0.000000e+00 : f32
  %fill = linalg.fill(%init, %cst) : tensor<1x2x2x1xf32>, f32 -> tensor<1x2x2x1xf32>
  %res = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xf32>, tensor<3x3xf32>)
    outs(%fill: tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  return %res : tensor<1x2x2x1xf32>
}

// -----

// CHECK-LABEL: func @pooling_nhwc_sum
// CHECK:         linalg.pooling_nhwc_sum
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x1xf32>, memref<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x1xf32>)
func @pooling_nhwc_sum(%input: memref<1x4x4x1xf32>, %fake: memref<3x3xf32>, %output: memref<1x2x2x1xf32>) {
  linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x4x4x1xf32>, memref<3x3xf32>)
    outs(%output: memref<1x2x2x1xf32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nhwc_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_nhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x1xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
func @pooling_nhwc_max_tensor(%input: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %fake = linalg.init_tensor [3, 3] : tensor<3x3xf32>
  %init = linalg.init_tensor [1, 2, 2, 1] : tensor<1x2x2x1xf32>
  %cst = constant 0.000000e+00 : f32
  %fill = linalg.fill(%init, %cst) : tensor<1x2x2x1xf32>, f32 -> tensor<1x2x2x1xf32>
  %res = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xf32>, tensor<3x3xf32>)
    outs(%fill: tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  return %res : tensor<1x2x2x1xf32>
}

// -----

// CHECK-LABEL: func @pooling_nhwc_max
// CHECK:         linalg.pooling_nhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x1xf32>, memref<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x1xf32>)
func @pooling_nhwc_max(%input: memref<1x4x4x1xf32>, %fake: memref<3x3xf32>, %output: memref<1x2x2x1xf32>) {
  linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x4x4x1xf32>, memref<3x3xf32>)
    outs(%output: memref<1x2x2x1xf32>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nhwc_i8_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_nhwc_i8_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x1xi8>, tensor<3x3xi8>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x1xi8>) -> tensor<1x2x2x1xi8>
func @pooling_nhwc_i8_max_tensor(%input: tensor<1x4x4x1xi8>) -> tensor<1x2x2x1xi8> {
  %fake = linalg.init_tensor [3, 3] : tensor<3x3xi8>
  %init = linalg.init_tensor [1, 2, 2, 1] : tensor<1x2x2x1xi8>
  %cst = constant 0 : i8
  %fill = linalg.fill(%init, %cst) : tensor<1x2x2x1xi8>, i8 -> tensor<1x2x2x1xi8>
  %res = linalg.pooling_nhwc_i8_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xi8>, tensor<3x3xi8>)
    outs(%fill: tensor<1x2x2x1xi8>) -> tensor<1x2x2x1xi8>
  return %res : tensor<1x2x2x1xi8>
}

// -----

// CHECK-LABEL: func @pooling_nhwc_i8_max
// CHECK:         linalg.pooling_nhwc_i8_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x1xi8>, memref<3x3xi8>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x1xi8>)
func @pooling_nhwc_i8_max(%input: memref<1x4x4x1xi8>, %fake: memref<3x3xi8>, %output: memref<1x2x2x1xi8>) {
  linalg.pooling_nhwc_i8_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x4x4x1xi8>, memref<3x3xi8>)
    outs(%output: memref<1x2x2x1xi8>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nhwc_i16_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_nhwc_i16_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x1xi16>, tensor<3x3xi16>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x1xi16>) -> tensor<1x2x2x1xi16>
func @pooling_nhwc_i16_max_tensor(%input: tensor<1x4x4x1xi16>) -> tensor<1x2x2x1xi16> {
  %fake = linalg.init_tensor [3, 3] : tensor<3x3xi16>
  %init = linalg.init_tensor [1, 2, 2, 1] : tensor<1x2x2x1xi16>
  %cst = constant 0 : i16
  %fill = linalg.fill(%init, %cst) : tensor<1x2x2x1xi16>, i16 -> tensor<1x2x2x1xi16>
  %res = linalg.pooling_nhwc_i16_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xi16>, tensor<3x3xi16>)
    outs(%fill: tensor<1x2x2x1xi16>) -> tensor<1x2x2x1xi16>
  return %res : tensor<1x2x2x1xi16>
}

// -----

// CHECK-LABEL: func @pooling_nhwc_i16_max
// CHECK:         linalg.pooling_nhwc_i16_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x1xi16>, memref<3x3xi16>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x1xi16>)
func @pooling_nhwc_i16_max(%input: memref<1x4x4x1xi16>, %fake: memref<3x3xi16>, %output: memref<1x2x2x1xi16>) {
  linalg.pooling_nhwc_i16_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x4x4x1xi16>, memref<3x3xi16>)
    outs(%output: memref<1x2x2x1xi16>)
  return
}

// -----

// CHECK-LABEL: func @pooling_nhwc_i32_max_tensor
// CHECK:         %{{.+}} = linalg.pooling_nhwc_i32_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x1xi32>, tensor<3x3xi32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x1xi32>) -> tensor<1x2x2x1xi32>
func @pooling_nhwc_i32_max_tensor(%input: tensor<1x4x4x1xi32>) -> tensor<1x2x2x1xi32> {
  %fake = linalg.init_tensor [3, 3] : tensor<3x3xi32>
  %init = linalg.init_tensor [1, 2, 2, 1] : tensor<1x2x2x1xi32>
  %cst = constant 0 : i32
  %fill = linalg.fill(%init, %cst) : tensor<1x2x2x1xi32>, i32 -> tensor<1x2x2x1xi32>
  %res = linalg.pooling_nhwc_i32_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xi32>, tensor<3x3xi32>)
    outs(%fill: tensor<1x2x2x1xi32>) -> tensor<1x2x2x1xi32>
  return %res : tensor<1x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @pooling_nhwc_i32_max
// CHECK:         linalg.pooling_nhwc_i32_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x1xi32>, memref<3x3xi32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x1xi32>)
func @pooling_nhwc_i32_max(%input: memref<1x4x4x1xi32>, %fake: memref<3x3xi32>, %output: memref<1x2x2x1xi32>) {
  linalg.pooling_nhwc_i32_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x4x4x1xi32>, memref<3x3xi32>)
    outs(%output: memref<1x2x2x1xi32>)
  return
}


// -----

// CHECK-LABEL: func @pooling_nhwc_min_tensor
// CHECK:         %{{.+}} = linalg.pooling_nhwc_min
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<1x4x4x1xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
func @pooling_nhwc_min_tensor(%input: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %fake = linalg.init_tensor [3, 3] : tensor<3x3xf32>
  %init = linalg.init_tensor [1, 2, 2, 1] : tensor<1x2x2x1xf32>
  %cst = constant 0.000000e+00 : f32
  %fill = linalg.fill(%init, %cst) : tensor<1x2x2x1xf32>, f32 -> tensor<1x2x2x1xf32>
  %res = linalg.pooling_nhwc_min {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xf32>, tensor<3x3xf32>)
    outs(%fill: tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  return %res : tensor<1x2x2x1xf32>
}

// -----

// CHECK-LABEL: func @pooling_nhwc_min
// CHECK:         linalg.pooling_nhwc_min
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:      strides = dense<1> : tensor<2xi64>
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : memref<1x4x4x1xf32>, memref<3x3xf32>)
// CHECK-SAME:      outs(%{{.+}} : memref<1x2x2x1xf32>)
func @pooling_nhwc_min(%input: memref<1x4x4x1xf32>, %fake: memref<3x3xf32>, %output: memref<1x2x2x1xf32>) {
  linalg.pooling_nhwc_min {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x4x4x1xf32>, memref<3x3xf32>)
    outs(%output: memref<1x2x2x1xf32>)
  return
}
