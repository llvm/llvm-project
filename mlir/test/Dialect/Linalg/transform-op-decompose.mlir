// RUN: mlir-opt --transform-interpreter --split-input-file %s | FileCheck %s

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:  #[[$MAP2:.+]] = affine_map<(d0, d1) -> ()>
// CHECK-DAG:  #[[$MAP3:.+]] = affine_map<(d0, d1) -> (d0, d1)>

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
// CHECK-DAG:        %[[EMP:.+]] = tensor.empty() : tensor<2x16xf32>
// CHECK-DAG:        %[[CST:.+]] = arith.constant -3.40282347E+38 : f32
// CHECK:        %[[FILL:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP3]]],
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:      ins(%[[CST]] : f32) outs(%[[EMP]] : tensor<2x16xf32>) {
// CHECK-NEXT:      ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NEXT:      linalg.yield %[[IN]] : f32
// CHECK-NEXT:    } -> tensor<2x16xf32>
// CHECK:        %[[MAX:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel", "reduction"]} ins(%[[ARG0]] : tensor<2x16x32xf32>) outs(%[[FILL]] : tensor<2x16xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[MAXF:.+]] = arith.maxnumf %[[IN]], %[[OUT]] : f32
// CHECK:          linalg.yield %[[MAXF]] : f32
// CHECK:        } -> tensor<2x16xf32>
// CHECK:        %[[BCST:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]],
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:      ins(%[[MAX]] : tensor<2x16xf32>) outs(%[[DST]] : tensor<2x16x32xf32>)
// CHECK-NEXT:      ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NEXT:        linalg.yield %[[IN]] : f32
// CHECK-NEXT:    } -> tensor<2x16x32xf32>
// CHECK:        %[[SUB:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "parallel"]} ins(%[[ARG0]], %[[BCST]] : tensor<2x16x32xf32>, tensor<2x16x32xf32>)
// CHECK-SAME:     outs(%[[DST]] : tensor<2x16x32xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[SUBF:.+]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK:          linalg.yield %[[SUBF]] : f32
// CHECK:        } -> tensor<2x16x32xf32>
// CHECK:         %[[EXP:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:      ins(%[[SUB]] : tensor<2x16x32xf32>)
// CHECK-SAME:      outs(%[[DST]] : tensor<2x16x32xf32>) {
// CHECK-NEXT:      ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NEXT:      %[[EXPF:.+]] = math.exp %[[IN]] : f32
// CHECK-NEXT:      linalg.yield %[[EXPF]] : f32
// CHECK-NEXT:    } -> tensor<2x16x32xf32>
// CHECK:        %[[CST_0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[BCST_1:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP3]]],
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:      ins(%[[CST_0]] : f32) outs(%[[EMP]] : tensor<2x16xf32>) {
// CHECK-NEXT:      ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NEXT:      linalg.yield %[[IN]] : f32
// CHECK-NEXT:    } -> tensor<2x16xf32>
// CHECK:        %[[SUM:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel", "reduction"]} ins(%[[EXP]] : tensor<2x16x32xf32>) outs(%[[BCST_1]] : tensor<2x16xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[ADDF:.+]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK:          linalg.yield %[[ADDF]] : f32
// CHECK:        } -> tensor<2x16xf32>
// CHECK:        %[[EMP:.+]] = tensor.empty() : tensor<2x16x32xf32>
// CHECK:        %[[CST:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP]]],
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:      ins(%[[SUM]] : tensor<2x16xf32>) outs(%[[EMP]] : tensor<2x16x32xf32>) {
// CHECK-NEXT:      ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NEXT:        linalg.yield %[[IN]] : f32
// CHECK-NEXT:    } -> tensor<2x16x32xf32>
// CHECK:        %[[DIV:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "parallel"]} ins(%[[EXP]], %[[CST]] : tensor<2x16x32xf32>, tensor<2x16x32xf32>)
// CHECK-SAME:     outs(%[[DST]] : tensor<2x16x32xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[DIVF:.+]] = arith.divf %[[IN]], %[[IN_1]] : f32
// CHECK:          linalg.yield %[[DIVF]] : f32
// CHECK:        } -> tensor<2x16x32xf32>
// CHECK:        return %[[DIV]] : tensor<2x16x32xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.decompose %0 : (!transform.any_op) -> !transform.any_op

    %2 = transform.structured.match ops{["linalg.softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.decompose_interface %2 : (!transform.any_op) -> !transform.any_op
    %4 = transform.structured.generalize %3: (!transform.any_op) -> !transform.any_op
    
    transform.yield
  }
}
