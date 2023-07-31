// RUN: mlir-opt --test-transform-dialect-interpreter --split-input-file %s | FileCheck %s

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: @conv_2d_nhwc_hwcf
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?x?x?xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xf32>
func.func @conv_2d_nhwc_hwcf(%input: tensor<?x1x?x?xf32>, %filter: tensor<1x?x?x?xf32>, %init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.conv_1d_nwc_wcf
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xf32>, tensor<1x?x?x?xf32>)
    outs (%init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xf32>
}

// CHECK-LABEL: @conv_2d_nchw_fchw
// CHECK-SAME: (%[[ARG0:[0-9a-z]+]]: tensor<?x?x1x?xf32>,
// CHECK-SAME: %[[ARG1:[0-9a-z]+]]: tensor<?x?x1x?xf32>,
// CHECK-SAME: %[[ARG2:[0-9a-z]+]]: tensor<?x?x1x?xf32>)
func.func @conv_2d_nchw_fchw(%input: tensor<?x?x1x?xf32>, %filter: tensor<?x?x1x?xf32>, %init: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.conv_1d_ncw_fcw
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x1x?xf32>, tensor<?x?x1x?xf32>)
    outs (%init: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x?x1x?xf32>
}

// CHECK-LABEL: @depthwise_conv_2d_nhwc_hwc
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x1x113x96xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x3x96xf32>
func.func @depthwise_conv_2d_nhwc_hwc(%input: tensor<1x1x113x96xf32>, %filter: tensor<1x3x96xf32>) -> tensor<1x1x56x96xf32> {
  // CHECK: %[[RES:.+]] = tensor.empty
  %init = tensor.empty() : tensor<1x1x56x96xf32>
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICERES:.+]] = tensor.extract_slice %[[RES]]
  // CHECK: %[[OPRES:.+]] = linalg.depthwise_conv_1d_nwc_wc
  // CHECK-SAME: ins(%[[SLICE0]], %[[SLICE1]]
  // CHECK-SAME: outs(%[[SLICERES]]
  // CHECK: %[[INSERTED:.+]] = tensor.insert_slice %[[OPRES]] into %[[RES]]
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
         ins(%input, %filter: tensor<1x1x113x96xf32>, tensor<1x3x96xf32>)
         outs(%init: tensor<1x1x56x96xf32>) -> tensor<1x1x56x96xf32>
  // CHECK: %[[INSERTED]]
  return %0: tensor<1x1x56x96xf32>
}

// CHECK-LABEL: @conv_2d
// CHECK-SAME: (%[[ARG0:[0-9a-z]+]]: tensor<1x?xf32>,
// CHECK-SAME: %[[ARG1:[0-9a-z]+]]: tensor<1x?xf32>,
// CHECK-SAME: %[[ARG2:[0-9a-z]+]]: tensor<1x?xf32>)
func.func @conv_2d(%input: tensor<1x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<1x?xf32>) -> tensor<1x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.conv_1d
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.conv_2d
     ins (%input, %filter: tensor<1x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<1x?xf32>) -> tensor<1x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<1x?xf32>
}

// CHECK-LABEL: @pooling_nhwc_sum
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xf32>
func.func @pooling_nhwc_sum(%input: tensor<?x1x?x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_nwc_sum
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xf32>
}

// CHECK-LABEL: @pooling_nchw_sum
// CHECK-SAME: (%[[ARG0:[0-9a-z]+]]: tensor<?x?x1x?xf32>,
// CHECK-SAME: %[[ARG1:[0-9a-z]+]]: tensor<1x?xf32>,
// CHECK-SAME: %[[ARG2:[0-9a-z]+]]: tensor<?x?x1x?xf32>)
func.func @pooling_nchw_sum(%input: tensor<?x?x1x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_ncw_sum
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x1x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x?x1x?xf32>
}

// CHECK-LABEL: @pooling_nhwc_max
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xf32>
func.func @pooling_nhwc_max(%input: tensor<?x1x?x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_nwc_max
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xf32>
}

// CHECK-LABEL: @pooling_nhwc_max_unsigned
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xf32>
func.func @pooling_nhwc_max_unsigned(%input: tensor<?x1x?x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_nwc_max_unsigned
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nhwc_max_unsigned {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xf32>
}

// CHECK-LABEL: @pooling_nhwc_min
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xf32>
func.func @pooling_nhwc_min(%input: tensor<?x1x?x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_nwc_min
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nhwc_min {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xf32>
}

// CHECK-LABEL: @pooling_nhwc_min_unsigned
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xf32>
func.func @pooling_nhwc_min_unsigned(%input: tensor<?x1x?x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_nwc_min_unsigned
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nhwc_min_unsigned {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xf32>
}

// CHECK-LABEL: @pooling_nchw_max
// CHECK-SAME: (%[[ARG0:[0-9a-z]+]]: tensor<?x?x1x?xf32>,
// CHECK-SAME: %[[ARG1:[0-9a-z]+]]: tensor<1x?xf32>,
// CHECK-SAME: %[[ARG2:[0-9a-z]+]]: tensor<?x?x1x?xf32>)
func.func @pooling_nchw_max(%input: tensor<?x?x1x?xf32>, %filter: tensor<1x?xf32>, %init: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_ncw_max
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x1x?xf32>, tensor<1x?xf32>)
    outs (%init: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x?x1x?xf32>
}

func.func @softmax(%arg0: tensor<2x16x32xf32>, %dst: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
  %1 = linalg.softmax dimension(2) ins(%arg0 : tensor<2x16x32xf32>) outs(%dst: tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %1 : tensor<2x16x32xf32>
}

// CHECK-LABEL:      func.func @softmax(
// CHECK-SAME:           %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>, %[[DST:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK-DAG:        %[[D1:.+]] = tensor.empty() : tensor<2x16xf32>
// CHECK-DAG:        %[[CST:.+]] = arith.constant 0xFF800000 : f32
// CHECK:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK:        %[[D3:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel", "reduction"]} ins(%[[ARG0]] : tensor<2x16x32xf32>) outs(%[[D2]] : tensor<2x16xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[D8:.+]] = arith.maxf %[[IN]], %[[OUT]] : f32
// CHECK:          linalg.yield %[[D8]] : f32
// CHECK:        } -> tensor<2x16xf32>
// CHECK:        %[[D4:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "parallel"]} ins(%[[ARG0]], %[[D3]] : tensor<2x16x32xf32>, tensor<2x16xf32>)
// CHECK-SAME:     outs(%[[DST]] : tensor<2x16x32xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[D8]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK:          %[[D9:.+]] = math.exp %[[D8]] : f32
// CHECK:          linalg.yield %[[D9]] : f32
// CHECK:        } -> tensor<2x16x32xf32>
// CHECK:        %[[CST_0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D5:.+]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK:        %[[D6:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel", "reduction"]} ins(%[[D4]] : tensor<2x16x32xf32>) outs(%[[D5]] : tensor<2x16xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[D8]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK:          linalg.yield %[[D8]] : f32
// CHECK:        } -> tensor<2x16xf32>
// CHECK:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "parallel"]} ins(%[[D4]], %[[D6]] : tensor<2x16x32xf32>, tensor<2x16xf32>)
// CHECK-SAME:     outs(%[[DST]] : tensor<2x16x32xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[D8]] = arith.divf %[[IN]], %[[IN_1]] : f32
// CHECK:          linalg.yield %[[D8]] : f32
// CHECK:        } -> tensor<2x16x32xf32>
// CHECK:        return %[[D7]] : tensor<2x16x32xf32>

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.structured.decompose %0 : (!transform.any_op) -> !transform.any_op

  %2 = transform.structured.match ops{["linalg.softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %3 = transform.structured.decompose_interface %2 : (!transform.any_op) -> !transform.any_op
}
