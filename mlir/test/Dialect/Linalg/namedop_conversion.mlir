// RUN: mlir-opt %s -linalg-named-op-conversion -split-input-file | FileCheck %s

// CHECK-LABEL: @depthwise_conv
func @depthwise_conv(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x1xf32>, %arg2: tensor<?x?x?x?x1xf32>) -> tensor<?x?x?x?x1xf32> {
  // CHECK-DAG: %[[KERNEL:.+]] = tensor.collapse_shape %arg1 {{\[\[}}0], [1], [2, 3]]
  // CHECK-DAG: %[[INIT:.+]] = tensor.collapse_shape %arg2 {{\[\[}}0], [1], [2], [3, 4]]
  // CHECK-DAG: %[[CONV:.+]] = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %[[KERNEL]] : tensor<?x?x?x?xf32>, tensor<?x?x?xf32>) outs(%[[INIT]] : tensor<?x?x?x?xf32>)
  // CHECK: %[[OUT:.+]] = tensor.expand_shape %[[CONV]] {{\[\[}}0], [1], [2], [3, 4]]
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x1xf32>) outs(%arg2 : tensor<?x?x?x?x1xf32>) -> tensor<?x?x?x?x1xf32>
  return %0 : tensor<?x?x?x?x1xf32>
}


// -----

// CHECK-LABEL: @depthwise_conv_q
func @depthwise_conv_q(%arg0: tensor<?x?x?x?xi8>, %arg1: tensor<?x?x?x1xi8>, %arg2: tensor<?x?x?x?x1xi32>, %arg3 : i32, %arg4 : i32) -> tensor<?x?x?x?x1xi32> {
  // CHECK-DAG: %[[KERNEL:.+]] = tensor.collapse_shape %arg1 {{\[\[}}0], [1], [2, 3]]
  // CHECK-DAG: %[[INIT:.+]] = tensor.collapse_shape %arg2 {{\[\[}}0], [1], [2], [3, 4]]
  // CHECK-DAG: %[[CONV:.+]] = linalg.depthwise_conv_2d_nhwc_hwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %[[KERNEL]], %arg3, %arg4 : tensor<?x?x?x?xi8>, tensor<?x?x?xi8>, i32, i32) outs(%[[INIT]] : tensor<?x?x?x?xi32>)
  // CHECK: %[[OUT:.+]] = tensor.expand_shape %[[CONV]] {{\[\[}}0], [1], [2], [3, 4]]
  %0 = linalg.depthwise_conv_2d_nhwc_hwcm_q {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1, %arg3, %arg4 : tensor<?x?x?x?xi8>, tensor<?x?x?x1xi8>, i32, i32) outs(%arg2 : tensor<?x?x?x?x1xi32>) -> tensor<?x?x?x?x1xi32>
  return %0 : tensor<?x?x?x?x1xi32>
}
