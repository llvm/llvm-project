// The following test examples of linalg convolution named ops lowered to linalg.generic and then
// lifted back up to named op.
// NOTE: Most tests in this file use dynamic shapes as the underlying transformations don't modify shapes. There's one exception that's added as a smoke test.

// RUN: mlir-opt %s -linalg-generalize-named-ops | mlir-opt --linalg-specialize-generic-ops | FileCheck %s --implicit-check-not=linalg.generic

// -----------------------------
// Convolution ops.
// -----------------------------
func.func @conv_1d(%in : tensor<?xf32>, %filter : tensor<?xf32>, %out : tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.conv_1d
         ins(%in, %filter : tensor<?xf32>, tensor<?xf32>)
         outs(%out : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
//      CHECK: @conv_1d
//      CHECK:   linalg.conv_1d

// -----

func.func @conv_1d_nwc_wcf(%input: tensor<?x?x?xf32>, %filter: tensor<?x?x?xf32>, %output: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.conv_1d_nwc_wcf
         {dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>}
         ins (%input, %filter: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
         outs (%output: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
//      CHECK: @conv_1d_nwc_wcf
//      CHECK:   linalg.conv_1d_nwc_wcf
// CHECK-SAME:      dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>

// -----

func.func @conv_1d_ncw_fcw(%input: tensor<?x?x?xf32>, %filter: tensor<?x?x?xf32>, %output: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.conv_1d_ncw_fcw
         {dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>}
         ins (%input, %filter: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
         outs (%output: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
//      CHECK: @conv_1d_ncw_fcw
//      CHECK:   linalg.conv_1d_ncw_fcw
// CHECK-SAME:      dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>

// -----

func.func @conv_2d(%in : tensor<?x?xf32>, %filter : tensor<?x?xf32>, %out : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.conv_2d
         ins(%in, %filter : tensor<?x?xf32>, tensor<?x?xf32>)
         outs(%out: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: @conv_2d
//      CHECK:   linalg.conv_2d

// -----

func.func @conv_3d(%in : tensor<?x?x?xf32>, %filter : tensor<?x?x?xf32>, %out : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.conv_3d
         ins(%in, %filter : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
         outs(%out : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
//      CHECK: @conv_3d
//      CHECK:   linalg.conv_3d

// -----

// -----------------------------
// Depthwise Convolution ops.
// -----------------------------
func.func @depthwise_conv_1d_ncw_cw(%input: tensor<?x?x?xf32>, %filter: tensor<?x?xf32>, %output: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.depthwise_conv_1d_ncw_cw
         {dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>}
         ins (%input, %filter: tensor<?x?x?xf32>, tensor<?x?xf32>)
         outs (%output: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
//      CHECK: @depthwise_conv_1d_ncw_cw
//      CHECK:   linalg.depthwise_conv_1d_ncw_cw
// CHECK-SAME:      dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>

// -----

func.func @depthwise_conv_1d_nwc_wc_static(%input: tensor<1x25x8xi8>, %filter: tensor<3x8xi8>, %output: tensor<1x10x8xi32>) -> tensor<1x10x8xi32> {
  %0 = linalg.depthwise_conv_1d_nwc_wc 
         {dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>}
         ins (%input, %filter: tensor<1x25x8xi8>, tensor<3x8xi8>)
         outs (%output: tensor<1x10x8xi32>) -> tensor<1x10x8xi32>
  return %0 : tensor<1x10x8xi32>
}
//      CHECK: @depthwise_conv_1d_nwc_wc_static
//      CHECK:   linalg.depthwise_conv_1d_nwc_wc
// CHECK-SAME:      dilations = dense<3> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>

// -----

func.func @depthwise_conv_1d_nwc_wcm(%input: tensor<?x?x?xf32>, %filter: tensor<?x?x?xf32>, %output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.depthwise_conv_1d_nwc_wcm
         {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
         ins (%input, %filter: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
         outs (%output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @depthwise_conv_1d_nwc_wcm
//      CHECK:   linalg.depthwise_conv_1d_nwc_wcm
// CHECK-SAME:      dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>

// -----

func.func @depthwise_conv_2d_nchw_chw(%input: tensor<?x?x?x?xf16>, %filter: tensor<?x?x?xf16>, %output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.depthwise_conv_2d_nchw_chw
         {dilations = dense<[2,3]> : vector<2xi64>, strides = dense<[4,5]> : vector<2xi64>}
         ins (%input, %filter: tensor<?x?x?x?xf16>, tensor<?x?x?xf16>)
         outs (%output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @depthwise_conv_2d_nchw_chw
//      CHECK:   linalg.depthwise_conv_2d_nchw_chw
// CHECK-SAME:      dilations = dense<[2, 3]> : tensor<2xi64>, strides = dense<[4, 5]> : tensor<2xi64>

// -----

func.func @depthwise_conv_3d_ndhwc_dhwcm(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %output: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32> {
  %0 = linalg.depthwise_conv_3d_ndhwc_dhwcm
         {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
         ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
         outs (%output: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?x?xf32>
}
//      CHECK: @depthwise_conv_3d_ndhwc_dhwcm
//      CHECK:   linalg.depthwise_conv_3d_ndhwc_dhwcm
// CHECK-SAME:      dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>

// -----

// -----------------------------
// Pooling ops.
// -----------------------------
func.func @pooling_nhwc_max(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?xf32>, %output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.pooling_nhwc_max
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?xf32>)
         outs (%output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @pooling_nhwc_max
//      CHECK:   linalg.pooling_nhwc_max
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>

// -----

func.func @pooling_nhwc_min(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?xf32>, %output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.pooling_nhwc_min
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?xf32>)
         outs (%output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @pooling_nhwc_min
//      CHECK:   linalg.pooling_nhwc_min
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>

// -----

func.func @pooling_nhwc_sum(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?xf32>, %output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.pooling_nhwc_sum
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?xf32>)
         outs (%output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @pooling_nhwc_sum
//      CHECK:   linalg.pooling_nhwc_sum
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>

// -----

func.func @pooling_nhwc_max_unsigned(%input: tensor<?x?x?x?xi8>, %filter: tensor<?x?xi8>, %output: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> {
  %0 = linalg.pooling_nhwc_max_unsigned
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins (%input, %filter: tensor<?x?x?x?xi8>, tensor<?x?xi8>)
         outs (%output: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  return %0 : tensor<?x?x?x?xi32>
}
//      CHECK: @pooling_nhwc_max_unsigned
//      CHECK:   linalg.pooling_nhwc_max_unsigned
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>

// -----

func.func @pooling_nhwc_min_unsigned_integer(%input: tensor<?x?x?x?xi32>, %filter: tensor<?x?xi32>, %output: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> {
  %0 = linalg.pooling_nhwc_min_unsigned
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins (%input, %filter: tensor<?x?x?x?xi32>, tensor<?x?xi32>)
         outs (%output: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  return %0 : tensor<?x?x?x?xi32>
}
//      CHECK: @pooling_nhwc_min_unsigned_integer
//      CHECK:   linalg.pooling_nhwc_min_unsigned
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>

// -----

func.func @pooling_nhwc_min_unsigned_float(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?xf32>, %output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.pooling_nhwc_min_unsigned
         {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?xf32>)
         outs (%output: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: @pooling_nhwc_min_unsigned_float
//      CHECK:   linalg.pooling_nhwc_min
// CHECK-SAME:      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
