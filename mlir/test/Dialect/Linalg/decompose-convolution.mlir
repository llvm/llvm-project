// RUN: mlir-opt -split-input-file -test-linalg-transform-patterns=test-decompose-convolution-patterns %s | FileCheck %s

// CHECK-LABEL: func @conv2d_nhwc_4x1x2x8_tensor
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<4x1x6x3xf32>, %[[FILTER:.+]]: tensor<1x2x3x8xf32>, %[[INIT:.+]]: tensor<4x1x2x8xf32>)
func @conv2d_nhwc_4x1x2x8_tensor(%input: tensor<4x1x6x3xf32>, %filter: tensor<1x2x3x8xf32>, %init: tensor<4x1x2x8xf32>) -> tensor<4x1x2x8xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<[2, 3]> : tensor<2xi64>, strides = dense<[3, 2]> : tensor<2xi64>}
    ins(%input, %filter : tensor<4x1x6x3xf32>, tensor<1x2x3x8xf32>)
    outs(%init : tensor<4x1x2x8xf32>) -> tensor<4x1x2x8xf32>
  return %0 : tensor<4x1x2x8xf32>
}

//               CHECK: %[[INPUT_1D:.+]] = linalg.tensor_collapse_shape %[[INPUT]]
// CHECK-SAME{LITERAL}:   [[0], [1, 2], [3]] : tensor<4x1x6x3xf32> into tensor<4x6x3xf32>
//               CHECK: %[[FILTER_1D:.+]] = linalg.tensor_collapse_shape %[[FILTER]]
// CHECK-SAME{LITERAL}:   [[0, 1], [2], [3]] : tensor<1x2x3x8xf32> into tensor<2x3x8xf32>
//               CHECK: %[[INIT_1D:.+]] = linalg.tensor_collapse_shape %[[INIT]]
// CHECK-SAME{LITERAL}:   [[0], [1, 2], [3]] : tensor<4x1x2x8xf32> into tensor<4x2x8xf32>
//               CHECK: %[[CONV_1D:.+]] = linalg.conv_1d_nwc_wcf
//          CHECK-SAME:     dilations = dense<3> : vector<1xi64>
//          CHECK-SAME:     strides = dense<2> : vector<1xi64>
//          CHECK-SAME:   ins(%[[INPUT_1D]], %[[FILTER_1D]] : tensor<4x6x3xf32>, tensor<2x3x8xf32>)
//          CHECK-SAME:   outs(%[[INIT_1D]] : tensor<4x2x8xf32>)
//               CHECK: %[[CONV_2D:.+]] = linalg.tensor_expand_shape %[[CONV_1D]]
// CHECK-SAME{LITERAL}:   [[0], [1, 2], [3]] : tensor<4x2x8xf32> into tensor<4x1x2x8xf32>
//               CHECK: return %[[CONV_2D]]

// -----

// CHECK-LABEL: func @conv2d_nhwc_qxqx1xq_tensor
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<?x?x1x?xf32>, %[[FILTER:.+]]: tensor<?x1x?x?xf32>, %[[INIT:.+]]: tensor<?x?x1x?xf32>)
func @conv2d_nhwc_qxqx1xq_tensor(%input: tensor<?x?x1x?xf32>, %filter: tensor<?x1x?x?xf32>, %init: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<[2, 3]> : tensor<2xi64>, strides = dense<[3, 2]> : tensor<2xi64>}
    ins(%input, %filter : tensor<?x?x1x?xf32>, tensor<?x1x?x?xf32>)
    outs(%init : tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32>
  return %0 : tensor<?x?x1x?xf32>
}

//               CHECK: %[[INPUT_1D:.+]] = linalg.tensor_collapse_shape %[[INPUT]]
// CHECK-SAME{LITERAL}:   [[0], [1, 2], [3]] : tensor<?x?x1x?xf32> into tensor<?x?x?xf32>
//               CHECK: %[[FILTER_1D:.+]] = linalg.tensor_collapse_shape %[[FILTER]]
// CHECK-SAME{LITERAL}:   [[0, 1], [2], [3]] : tensor<?x1x?x?xf32> into tensor<?x?x?xf32>
//               CHECK: %[[INIT_1D:.+]] = linalg.tensor_collapse_shape %[[INIT]]
// CHECK-SAME{LITERAL}:   [[0], [1, 2], [3]] : tensor<?x?x1x?xf32> into tensor<?x?x?xf32>
//               CHECK: %[[CONV_1D:.+]] = linalg.conv_1d_nwc_wcf
//          CHECK-SAME:     dilations = dense<2> : vector<1xi64>
//          CHECK-SAME:     strides = dense<3> : vector<1xi64>
//          CHECK-SAME:   ins(%[[INPUT_1D]], %[[FILTER_1D]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
//          CHECK-SAME:   outs(%[[INIT_1D]] : tensor<?x?x?xf32>)
//               CHECK: %[[CONV_2D:.+]] = linalg.tensor_expand_shape %[[CONV_1D]]
// CHECK-SAME{LITERAL}:   [[0], [1, 2], [3]] : tensor<?x?x?xf32> into tensor<?x?x1x?xf32>
//               CHECK: return %[[CONV_2D]]

// -----

// Do not convert convolution ops whose window dimensions are not ones.

// CHECK-LABEL: func @conv2d_nhwc_4x1x2x8_tensor
func @conv2d_nhwc_4x1x2x8_tensor(%input: tensor<4x3x5x3xf32>, %filter: tensor<2x2x3x8xf32>, %init: tensor<4x1x2x8xf32>) -> tensor<4x1x2x8xf32> {
  // CHECK: linalg.conv_2d_nhwc_hwcf
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<[2, 3]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %filter : tensor<4x3x5x3xf32>, tensor<2x2x3x8xf32>)
    outs(%init : tensor<4x1x2x8xf32>) -> tensor<4x1x2x8xf32>
  return %0 : tensor<4x1x2x8xf32>
}
