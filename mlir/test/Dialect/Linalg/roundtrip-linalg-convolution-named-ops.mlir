// The following test examples of linalg convolution named ops lowered to linalg.generic and then
// lifted back up to named op.
// RUN: mlir-opt %s -linalg-generalize-named-ops | mlir-opt --linalg-specialize-generic-ops | FileCheck %s

func.func @conv_1d_nwc_wcf(%input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?xf32>) {
  linalg.conv_1d_nwc_wcf {dilations = dense<3> : tensor<1xi64>,
                                       strides = dense<2> : tensor<1xi64>}
     ins (%input, %filter: memref<?x?x?xf32>, memref<?x?x?xf32>)
    outs (%output: memref<?x?x?xf32>)
  return
}
//      CHECK: @conv_1d_nwc_wcf
//      CHECK:   linalg.conv_1d_nwc_wcf
// CHECK-SAME:      dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_1d_ncw_fcw(%input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?xf32>) {
  linalg.conv_1d_ncw_fcw {dilations = dense<3> : tensor<1xi64>,
                                       strides = dense<2> : tensor<1xi64>}
     ins (%input, %filter: memref<?x?x?xf32>, memref<?x?x?xf32>)
    outs (%output: memref<?x?x?xf32>)
  return
}
//      CHECK: @conv_1d_ncw_fcw
//      CHECK:   linalg.conv_1d_ncw_fcw
// CHECK-SAME:      dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_1d(%in : memref<?xf32>, %filter : memref<?xf32>, %out : memref<?xf32>) -> () {
  linalg.conv_1d ins(%in, %filter : memref<?xf32>, memref<?xf32>)
                outs(%out : memref<?xf32>)
  return
}
//      CHECK: @conv_1d
//      CHECK:   linalg.conv_1d
//  CHECK-NOT:   linalg.generic

// -----

func.func @depthwise_conv_1d_ncw_cw(%input: memref<?x?x?xf32>, %filter: memref<?x?xf32>, %output: memref<?x?x?xf32>) {
  linalg.depthwise_conv_1d_ncw_cw {dilations = dense<3> : tensor<1xi64>,
                                       strides = dense<2> : tensor<1xi64>}
     ins (%input, %filter: memref<?x?x?xf32>, memref<?x?xf32>)
    outs (%output: memref<?x?x?xf32>)
  return
}
//      CHECK: @depthwise_conv_1d_ncw_cw
//      CHECK:   linalg.depthwise_conv_1d_ncw_cw
// CHECK-SAME:      dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @depthwise_conv_1d_nwc_wc(%input: memref<?x?x?xf32>, %filter: memref<?x?xf32>, %output: memref<?x?x?xf32>) {
  linalg.depthwise_conv_1d_nwc_wc {dilations = dense<3> : tensor<1xi64>,
                                       strides = dense<2> : tensor<1xi64>}
     ins (%input, %filter: memref<?x?x?xf32>, memref<?x?xf32>)
    outs (%output: memref<?x?x?xf32>)
  return
}
//      CHECK: @depthwise_conv_1d_nwc_wc
//      CHECK:   linalg.depthwise_conv_1d_nwc_wc
// CHECK-SAME:      dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @depthwise_conv_1d_nwc_wcm(%input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?x?xf32>) {
  linalg.depthwise_conv_1d_nwc_wcm {dilations = dense<1> : tensor<1xi64>,
                                       strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: memref<?x?x?xf32>, memref<?x?x?xf32>)
    outs (%output: memref<?x?x?x?xf32>)
  return
}
//      CHECK: @depthwise_conv_1d_nwc_wcm
//      CHECK:   linalg.depthwise_conv_1d_nwc_wcm
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_2d(%in : memref<?x?xf32>, %filter : memref<?x?xf32>, %out : memref<?x?xf32>) -> () {
  linalg.conv_2d ins(%in, %filter : memref<?x?xf32>, memref<?x?xf32>)
                outs(%out: memref<?x?xf32>)
  return
}
//      CHECK: @conv_2d
//      CHECK:   linalg.conv_2d
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_2d_nchw_fchw(%arg0 : tensor<?x?x?x?xf32>,
    %arg1 : tensor<?x?x?x?xf32>,  %arg2 : tensor<?x?x?x?xf32>) ->
    (tensor<4x8x12x16xf32>, tensor<?x?x?x?xf32>) {
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<[2,4]> : tensor<2xi64>, strides = dense<[3,5]> : tensor<2xi64>}
  ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %1 = tensor.cast %0 : tensor<?x?x?x?xf32> to tensor<4x8x12x16xf32>
  return %1, %0 : tensor<4x8x12x16xf32>, tensor<?x?x?x?xf32>
}
//      CHECK: @conv_2d_nchw_fchw
//      CHECK:   linalg.conv_2d_nchw_fchw
// CHECK-SAME:      dilations = dense<[2, 4]> : tensor<2xi64>, strides = dense<[3, 5]> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_2d_nchw_fchw_q(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %inputzp: i32, %filterzp: i32, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.conv_2d_nchw_fchw_q {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter, %inputzp, %filterzp: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, i32, i32)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @conv_2d_nchw_fchw_q
//      CHECK:   linalg.conv_2d_nchw_fchw_q
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_2d_ngchw_fgchw(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg.conv_2d_ngchw_fgchw {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}
//      CHECK: @conv_2d_ngchw_fgchw
//      CHECK:   linalg.conv_2d_ngchw_fgchw
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_2d_ngchw_gfchw(%input: memref<?x?x?x?x?xi8>, %filter: memref<?x?x?x?x?xi8>, %output: memref<?x?x?x?x?xi32>) {
  linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : tensor<2xi64>,
                                       strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: memref<?x?x?x?x?xi8>, memref<?x?x?x?x?xi8>)
    outs (%output: memref<?x?x?x?x?xi32>)
  return
}
//      CHECK: @conv_2d_ngchw_gfchw
//      CHECK:   linalg.conv_2d_ngchw_gfchw
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_2d_ngchw_gfchw_q(%input: memref<?x?x?x?x?xi8>, %filter: memref<?x?x?x?x?xi8>, %inputzp: i32, %filterzp: i32, %output: memref<?x?x?x?x?xi32>) {
  linalg.conv_2d_ngchw_gfchw_q {dilations = dense<1> : tensor<2xi64>,
                                       strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter, %inputzp, %filterzp: memref<?x?x?x?x?xi8>, memref<?x?x?x?x?xi8>, i32, i32)
    outs (%output: memref<?x?x?x?x?xi32>)
  return
}
//      CHECK: @conv_2d_ngchw_gfchw_q
//      CHECK:   linalg.conv_2d_ngchw_gfchw_q
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_2d_nhwc_hwcf_q(%input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?xf32>, %inputzp: i32, %filterzp: i32, %output: memref<?x?x?x?xf32>) {
  linalg.conv_2d_nhwc_hwcf_q {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %inputzp, %filterzp : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32, i32) outs(%output : memref<?x?x?x?xf32>)
  return
}
//      CHECK: @conv_2d_nhwc_hwcf_q
//      CHECK:   linalg.conv_2d_nhwc_hwcf_q
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_2d_nhwgc_gfhwc_q(%input: memref<?x?x?x?x?xf32>, %filter: memref<?x?x?x?x?xf32>, %inputzp: i32, %filterzp: i32, %output: memref<?x?x?x?x?xf32>) {
  linalg.conv_2d_nhwgc_gfhwc_q {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %inputzp, %filterzp : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32, i32) outs(%output : memref<?x?x?x?x?xf32>)
  return
}
//      CHECK: @conv_2d_nhwgc_gfhwc_q
//      CHECK:   linalg.conv_2d_nhwgc_gfhwc_q
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @depthwise_conv_2d_nhwc_hwc_q(%input: tensor<?x?x?x?xi8>, %filter: tensor<?x?x?xi8>, %inputzp: i32, %filterzp: i32, %output: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>{
  %res = linalg.depthwise_conv_2d_nhwc_hwc_q {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter, %inputzp, %filterzp : tensor<?x?x?x?xi8>, tensor<?x?x?xi8>, i32, i32) outs(%output : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  return %res : tensor<?x?x?x?xi32>
}
//      CHECK: @depthwise_conv_2d_nhwc_hwc_q
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwc_q
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_2d_nhwc_fhwc(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @conv_2d_nhwc_fhwc
//      CHECK:   linalg.conv_2d_nhwc_fhwc
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_2d_nhwc_fhwc_q(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %inputzp: i32, %filterzp: i32, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc_q {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter, %inputzp, %filterzp: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, i32, i32)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @conv_2d_nhwc_fhwc_q
//      CHECK:   linalg.conv_2d_nhwc_fhwc_q
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_2d_nhwc_hwcf(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @conv_2d_nhwc_hwcf
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_2d_nhwgc_gfhwc(%input: memref<?x?x?x?x?xf32>, %filter: memref<?x?x?x?x?xf32>, %output: memref<?x?x?x?x?xf32>) {
  linalg.conv_2d_nhwgc_gfhwc {dilations = dense<1> : tensor<2xi64>,
                                         strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
    outs (%output: memref<?x?x?x?x?xf32>)
  return
}
//      CHECK: @conv_2d_nhwgc_gfhwc
//      CHECK:   linalg.conv_2d_nhwgc_gfhwc
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @depthwise_conv_2d_nchw_chw(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<[2,3]> : vector<2xi64>, strides = dense<[4,5]> : vector<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @depthwise_conv_2d_nchw_chw
//      CHECK:   linalg.depthwise_conv_2d_nchw_chw
// CHECK-SAME:      dilations = dense<[2, 3]> : tensor<2xi64>, strides = dense<[4, 5]> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @depthwise_conv_2d_nhwc_hwc(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>,
                                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @depthwise_conv_2d_nhwc_hwc
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @depthwise_conv_2d_nhwc_hwcm(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>,
                                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}
//      CHECK: @depthwise_conv_2d_nhwc_hwcm
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwcm
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @depthwise_conv_2d_nhwc_hwcm_q(%arg0: tensor<?x?x?x?xi8>, %arg1: tensor<?x?x?x?xi8>, %arg2: tensor<?x?x?x?x?xi32>, %arg3 : i32, %arg4 : i32) -> tensor<?x?x?x?x?xi32> {
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1, %arg3, %arg4 : tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>, i32, i32) outs(%arg2 : tensor<?x?x?x?x?xi32>) -> tensor<?x?x?x?x?xi32>
  return %0 : tensor<?x?x?x?x?xi32>
}
//      CHECK: @depthwise_conv_2d_nhwc_hwcm_q
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwcm_q
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_3d(%in : memref<?x?x?xf32>, %filter : memref<?x?x?xf32>, %out : memref<?x?x?xf32>) -> () {
  linalg.conv_3d ins(%in, %filter : memref<?x?x?xf32>, memref<?x?x?xf32>)
                outs(%out : memref<?x?x?xf32>)
  return
}
//      CHECK: @conv_3d
//      CHECK:   linalg.conv_3d
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_3d_ncdhw_fcdhw(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg.conv_3d_ncdhw_fcdhw {dilations = dense<1> : tensor<3xi64>,
                                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}
//      CHECK: @conv_3d_ncdhw_fcdhw
//      CHECK:   linalg.conv_3d_ncdhw_fcdhw
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_3d_ndhwc_dhwcf(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}
//      CHECK: @conv_3d_ndhwc_dhwcf
//      CHECK:   linalg.conv_3d_ndhwc_dhwcf
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @conv_3d_ndhwc_dhwcf_q(%input: tensor<?x?x?x?x?xi8>, %filter: tensor<?x?x?x?x?xi8>, %inputzp: i32, %filterzp: i32, %init: tensor<?x?x?x?x?xi32>) -> tensor<?x?x?x?x?xi32> {
  %0 = linalg.conv_3d_ndhwc_dhwcf_q {dilations = dense<1> : tensor<3xi64>,
                                                strides = dense<1> : tensor<3xi64>}
    ins(%input, %filter, %inputzp, %filterzp : tensor<?x?x?x?x?xi8>, tensor<?x?x?x?x?xi8>, i32, i32)
    outs (%init: tensor<?x?x?x?x?xi32>) -> tensor<?x?x?x?x?xi32>
  return %0 : tensor<?x?x?x?x?xi32>
}
//      CHECK: @conv_3d_ndhwc_dhwcf_q
//      CHECK:   linalg.conv_3d_ndhwc_dhwcf_q
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @depthwise_conv_3d_ncdhw_cdhw(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg.depthwise_conv_3d_ncdhw_cdhw {dilations = dense<1> : tensor<3xi64>,
                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}
//      CHECK: @depthwise_conv_3d_ncdhw_cdhw
//      CHECK:   linalg.depthwise_conv_3d_ncdhw_cdhw
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @depthwise_conv_3d_ndhwc_dhwc(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg.depthwise_conv_3d_ndhwc_dhwc {dilations = dense<1> : tensor<3xi64>,
                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}
//      CHECK: @depthwise_conv_3d_ndhwc_dhwc
//      CHECK:   linalg.depthwise_conv_3d_ndhwc_dhwc
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @depthwise_conv_3d_ndhwc_dhwcm(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32> {
  %0 = linalg.depthwise_conv_3d_ndhwc_dhwcm {dilations = dense<1> : tensor<3xi64>,
                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?x?xf32>
}
//      CHECK: @depthwise_conv_3d_ndhwc_dhwcm
//      CHECK:   linalg.depthwise_conv_3d_ndhwc_dhwcm
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_nchw_max(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @pooling_nchw_max
//      CHECK:   linalg.pooling_nchw_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_nchw_sum(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @pooling_nchw_sum
//      CHECK:   linalg.pooling_nchw_sum
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_ncw_max(%input: tensor<?x?x?xf32>, %output: tensor<?x?x?xf32>, %filter: tensor<?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.pooling_ncw_max {dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>}
    ins(%input, %filter: tensor<?x?x?xf32>, tensor<?xf32>)
    outs(%output: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
//      CHECK: @pooling_ncw_max
//      CHECK:   linalg.pooling_ncw_max
// CHECK-SAME:      dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_ncw_sum(%input: tensor<?x?x?xf32>, %output: tensor<?x?x?xf32>, %filter: tensor<?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.pooling_ncw_sum {dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>}
    ins(%input, %filter: tensor<?x?x?xf32>, tensor<?xf32>)
    outs(%output: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
//      CHECK: @pooling_ncw_sum
//      CHECK:   linalg.pooling_ncw_sum
// CHECK-SAME:      dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_ndhwc_max(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg.pooling_ndhwc_max {dilations = dense<1> : tensor<3xi64>,
                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}
//      CHECK: @pooling_ndhwc_max
//      CHECK:   linalg.pooling_ndhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_ndhwc_min(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg.pooling_ndhwc_min {dilations = dense<1> : tensor<3xi64>,
                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}
//      CHECK: @pooling_ndhwc_min
//      CHECK:   linalg.pooling_ndhwc_min
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_ndhwc_sum(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg.pooling_ndhwc_sum {dilations = dense<1> : tensor<3xi64>,
                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}
//      CHECK: @pooling_ndhwc_sum
//      CHECK:   linalg.pooling_ndhwc_sum
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_nhwc_max(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @pooling_nhwc_max
//      CHECK:   linalg.pooling_nhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_nhwc_min(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.pooling_nhwc_min {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @pooling_nhwc_min
//      CHECK:   linalg.pooling_nhwc_min
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_nhwc_sum(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @pooling_nhwc_sum
//      CHECK:   linalg.pooling_nhwc_sum
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_nhwc_max_unsigned(%input: tensor<?x?x?x?xi32>, %filter: tensor<?x?xi32>, %init: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> {
  %0 = linalg.pooling_nhwc_max_unsigned {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xi32>, tensor<?x?xi32>)
    outs (%init: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  return %0 : tensor<?x?x?x?xi32>
}
//      CHECK: @pooling_nhwc_max_unsigned
//      CHECK:   linalg.pooling_nhwc_max_unsigned
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_nhwc_min_unsigned(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.pooling_nhwc_min_unsigned {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @pooling_nhwc_min_unsigned
//      CHECK:   linalg.pooling_nhwc_min
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_nwc_max(%input: tensor<?x?x?xf32>, %output: tensor<?x?x?xf32>, %filter: tensor<?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter: tensor<?x?x?xf32>, tensor<?xf32>)
    outs(%output: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
//      CHECK: @pooling_nwc_max
//      CHECK:   linalg.pooling_nwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_nwc_min(%input: tensor<?x?x?xf32>, %output: tensor<?x?x?xf32>, %filter: tensor<?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.pooling_nwc_min {dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>}
    ins(%input, %filter: tensor<?x?x?xf32>, tensor<?xf32>)
    outs(%output: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
//      CHECK: @pooling_nwc_min
//      CHECK:   linalg.pooling_nwc_min
// CHECK-SAME:      dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>
//  CHECK-NOT:   linalg.generic

// -----

func.func @pooling_nwc_sum(%input: tensor<?x?x?xf32>, %output: tensor<?x?x?xf32>, %filter: tensor<?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.pooling_nwc_sum {dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>}
    ins(%input, %filter: tensor<?x?x?xf32>, tensor<?xf32>)
    outs(%output: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
//      CHECK: @pooling_nwc_sum
//      CHECK:   linalg.pooling_nwc_sum
// CHECK-SAME:      dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>
//  CHECK-NOT:   linalg.generic
