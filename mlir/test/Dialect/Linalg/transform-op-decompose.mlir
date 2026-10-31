// RUN: mlir-opt --transform-interpreter --split-input-file %s | FileCheck %s

// Expected indexing maps for batchless conv_1d_nwc_wcf.
// CHECK-DAG:  #[[$CONV_I:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 + d2, d3)>
// CHECK-DAG:  #[[$CONV_F:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
// CHECK-DAG:  #[[$CONV_O:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// Expected indexing maps for batchless depthwise_conv_1d_wc_wcf.
// CHECK-DAG:  #[[$DW_I:.+]] = affine_map<(d0, d1, d2) -> (d0 + d2, d1)>
// CHECK-DAG:  #[[$DW_F:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>

// Expected indexing maps for batchless pooling_cw_min.
// CHECK-DAG:  #[[$POOL_I:.+]] = affine_map<(d0, d1, d2) -> (d0, d1 + d2)>
// CHECK-DAG:  #[[$POOL_F:.+]] = affine_map<(d0, d1, d2) -> (d2)>

// CHECK-DAG:  #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:  #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// Expected indexing maps for 1D conv (cross-conv after downscale from generic).
// CHECK-DAG:  #[[$CROSS_1D_I:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG:  #[[$CROSS_1D_F:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:  #[[$CROSS_1D_O:.+]] = affine_map<(d0, d1) -> (d0)>

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

// Depthwise conv with height=1 (downscales height dimension)
// CHECK-LABEL: @depthwise_conv_2d_nhwc_hwc_height
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x1x113x96xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x3x96xf32>
func.func @depthwise_conv_2d_nhwc_hwc_height(%input: tensor<1x1x113x96xf32>, %filter: tensor<1x3x96xf32>) -> tensor<1x1x56x96xf32> {
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

// Depthwise conv with width=1 (downscales width dimension)
// CHECK-LABEL: @depthwise_conv_2d_nhwc_hwc_width
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x113x1x96xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<3x1x96xf32>
func.func @depthwise_conv_2d_nhwc_hwc_width(%input: tensor<1x113x1x96xf32>, %filter: tensor<3x1x96xf32>) -> tensor<1x56x1x96xf32> {
  // CHECK: %[[RES:.+]] = tensor.empty
  %init = tensor.empty() : tensor<1x56x1x96xf32>
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICERES:.+]] = tensor.extract_slice %[[RES]]
  // CHECK: %[[OPRES:.+]] = linalg.depthwise_conv_1d_nwc_wc
  // CHECK-SAME: ins(%[[SLICE0]], %[[SLICE1]]
  // CHECK-SAME: outs(%[[SLICERES]]
  // CHECK: %[[INSERTED:.+]] = tensor.insert_slice %[[OPRES]] into %[[RES]]
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
         ins(%input, %filter: tensor<1x113x1x96xf32>, tensor<3x1x96xf32>)
         outs(%init: tensor<1x56x1x96xf32>) -> tensor<1x56x1x96xf32>
  // CHECK: %[[INSERTED]]
  return %0: tensor<1x56x1x96xf32>
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
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xi32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?xi32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xi32>
func.func @pooling_nhwc_max_unsigned(%input: tensor<?x1x?x?xi32>, %filter: tensor<1x?xi32>, %init: tensor<?x1x?x?xi32>) -> tensor<?x1x?x?xi32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_nwc_max_unsigned
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nhwc_max_unsigned {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xi32>, tensor<1x?xi32>)
    outs (%init: tensor<?x1x?x?xi32>) -> tensor<?x1x?x?xi32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xi32>
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
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x1x?x?xi32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x?xi32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<?x1x?x?xi32>
func.func @pooling_nhwc_min_unsigned(%input: tensor<?x1x?x?xi32>, %filter: tensor<1x?xi32>, %init: tensor<?x1x?x?xi32>) -> tensor<?x1x?x?xi32> {
  // CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK: %[[SLICERES:.+]] = linalg.pooling_nwc_min_unsigned
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.pooling_nhwc_min_unsigned {dilations = dense<1> : tensor<2xi64>,
                                strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x1x?x?xi32>, tensor<1x?xi32>)
    outs (%init: tensor<?x1x?x?xi32>) -> tensor<?x1x?x?xi32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?x1x?x?xi32>
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

#map_conv_i = affine_map<(oh, ow, f, kh, kw, c) -> (oh + kh, ow + kw, c)>
#map_conv_f = affine_map<(oh, ow, f, kh, kw, c) -> (kh, kw, c, f)>
#map_conv_o = affine_map<(oh, ow, f, kh, kw, c) -> (oh, ow, f)>

// CHECK-LABEL: @batchless_conv_2d_hwc_hwcf
// CHECK-SAME:    %[[ARG0:.+]]: tensor<1x14x8xf32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<1x3x8x16xf32>
// CHECK-SAME:    %[[ARG2:.+]]: tensor<1x12x16xf32>
func.func @batchless_conv_2d_hwc_hwcf(%input: tensor<1x14x8xf32>, %filter: tensor<1x3x8x16xf32>, %output: tensor<1x12x16xf32>) -> tensor<1x12x16xf32> {
  // CHECK:       %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK:       %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK:       %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK:       %[[SLICERES:.+]] = linalg.generic
  // CHECK-SAME:    indexing_maps = [#[[$CONV_I]], #[[$CONV_F]], #[[$CONV_O]]]
  // CHECK-SAME:    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  // CHECK:       %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.generic {
    indexing_maps = [#map_conv_i, #map_conv_f, #map_conv_o],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%input, %filter : tensor<1x14x8xf32>, tensor<1x3x8x16xf32>)
    outs(%output : tensor<1x12x16xf32>) {
  ^bb0(%in: f32, %fil: f32, %out: f32):
    %mul = arith.mulf %in, %fil : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<1x12x16xf32>
  // CHECK:       return %[[RES]]
  return %0 : tensor<1x12x16xf32>
}

// Dropping oh and kh must replace their indices by zero and renumber every
// remaining index from (oh, ow, f, kh, kw, c) to (ow, f, kw, c).
// CHECK-LABEL: @batchless_conv_2d_hwc_hwcf_indices
func.func @batchless_conv_2d_hwc_hwcf_indices(%input: tensor<1x3x1xi32>, %filter: tensor<1x1x1x2xi32>, %init: tensor<1x3x2xi32>) -> tensor<1x3x2xi32> {
  // CHECK: %[[CONV:.+]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$CONV_I]], #[[$CONV_F]], #[[$CONV_O]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  // CHECK: ^bb0(%[[IN:.+]]: i32, %[[FILTER:.+]]: i32, %[[ACC:.+]]: i32):
  // CHECK-NEXT: %[[OH:.+]] = arith.constant 0 : index
  // CHECK-NEXT: %[[OW:.+]] = linalg.index 0 : index
  // CHECK-NEXT: %[[F:.+]] = linalg.index 1 : index
  // CHECK-NEXT: %[[KH:.+]] = arith.constant 0 : index
  // CHECK-NEXT: %[[KW:.+]] = linalg.index 2 : index
  // CHECK-NEXT: %[[C:.+]] = linalg.index 3 : index
  // CHECK-NEXT: %[[S0:.+]] = arith.addi %[[OH]], %[[OW]] : index
  // CHECK-NEXT: %[[S1:.+]] = arith.addi %[[S0]], %[[F]] : index
  // CHECK-NEXT: %[[S2:.+]] = arith.addi %[[S1]], %[[KH]] : index
  // CHECK-NEXT: %[[S3:.+]] = arith.addi %[[S2]], %[[KW]] : index
  // CHECK-NEXT: %[[S4:.+]] = arith.addi %[[S3]], %[[C]] : index
  // CHECK-NEXT: %[[POSITION:.+]] = arith.index_cast %[[S4]] : index to i32
  // CHECK-NEXT: %[[PRODUCT:.+]] = arith.muli %[[IN]], %[[FILTER]] : i32
  // CHECK-NEXT: %[[SUM:.+]] = arith.addi %[[ACC]], %[[PRODUCT]] : i32
  // CHECK-NEXT: %[[VALUE:.+]] = arith.addi %[[SUM]], %[[POSITION]] : i32
  // CHECK-NEXT: linalg.yield %[[VALUE]] : i32
  // CHECK: %[[RESULT:.+]] = tensor.insert_slice %[[CONV]]
  // CHECK: return %[[RESULT]] : tensor<1x3x2xi32>
  %result = linalg.generic {
    indexing_maps = [#map_conv_i, #map_conv_f, #map_conv_o],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%input, %filter : tensor<1x3x1xi32>, tensor<1x1x1x2xi32>)
    outs(%init : tensor<1x3x2xi32>) {
  ^bb0(%a: i32, %b: i32, %acc: i32):
    %oh = linalg.index 0 : index
    %ow = linalg.index 1 : index
    %f = linalg.index 2 : index
    %kh = linalg.index 3 : index
    %kw = linalg.index 4 : index
    %c = linalg.index 5 : index
    %s0 = arith.addi %oh, %ow : index
    %s1 = arith.addi %s0, %f : index
    %s2 = arith.addi %s1, %kh : index
    %s3 = arith.addi %s2, %kw : index
    %s4 = arith.addi %s3, %c : index
    %position = arith.index_cast %s4 : index to i32
    %product = arith.muli %a, %b : i32
    %sum = arith.addi %acc, %product : i32
    %value = arith.addi %sum, %position : i32
    linalg.yield %value : i32
  } -> tensor<1x3x2xi32>
  return %result : tensor<1x3x2xi32>
}

#map_dw_i = affine_map<(oh, ow, c, kh, kw) -> (oh + kh, ow + kw, c)>
#map_dw_f = affine_map<(oh, ow, c, kh, kw) -> (kh, kw, c)>
#map_dw_o = affine_map<(oh, ow, c, kh, kw) -> (oh, ow, c)>

// CHECK-LABEL: @batchless_depthwise_conv_2d_hwc_hwc
// CHECK-SAME:    %[[ARG0:.+]]: tensor<1x14x8xf32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<1x3x8xf32>
// CHECK-SAME:    %[[ARG2:.+]]: tensor<1x12x8xf32>
func.func @batchless_depthwise_conv_2d_hwc_hwc(%input: tensor<1x14x8xf32>, %filter: tensor<1x3x8xf32>, %output: tensor<1x12x8xf32>) -> tensor<1x12x8xf32> {
  // CHECK:       %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK:       %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK:       %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK:       %[[SLICERES:.+]] = linalg.generic
  // CHECK-SAME:    indexing_maps = [#[[$DW_I]], #[[$DW_F]], #[[$MAP1]]]
  // CHECK-SAME:    iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK:       %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.generic {
    indexing_maps = [#map_dw_i, #map_dw_f, #map_dw_o],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<1x14x8xf32>, tensor<1x3x8xf32>)
    outs(%output : tensor<1x12x8xf32>) {
  ^bb0(%in: f32, %fil: f32, %out: f32):
    %mul = arith.mulf %in, %fil : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<1x12x8xf32>
  // CHECK:       return %[[RES]]
  return %0 : tensor<1x12x8xf32>
}

#map_pool_i = affine_map<(c, oh, ow, kh, kw) -> (c, oh + kh, ow + kw)>
#map_pool_f = affine_map<(c, oh, ow, kh, kw) -> (kh, kw)>
#map_pool_o = affine_map<(c, oh, ow, kh, kw) -> (c, oh, ow)>

// CHECK-LABEL: @batchless_pooling_chw_min
// CHECK-SAME:    %[[ARG0:.+]]: tensor<8x1x14xf32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<1x3xf32>
// CHECK-SAME:    %[[ARG2:.+]]: tensor<8x1x12xf32>
func.func @batchless_pooling_chw_min(%input: tensor<8x1x14xf32>, %filter: tensor<1x3xf32>, %output: tensor<8x1x12xf32>) -> tensor<8x1x12xf32> {
  // CHECK:       %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK:       %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK:       %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK:       %[[SLICERES:.+]] = linalg.generic
  // CHECK-SAME:    indexing_maps = [#[[$POOL_I]], #[[$POOL_F]], #[[$MAP1]]]
  // CHECK-SAME:    iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK:       %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  %0 = linalg.generic {
    indexing_maps = [#map_pool_i, #map_pool_f, #map_pool_o],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<8x1x14xf32>, tensor<1x3xf32>)
    outs(%output : tensor<8x1x12xf32>) {
  ^bb0(%in: f32, %fil: f32, %out: f32):
    %min = arith.minimumf %out, %in : f32
    linalg.yield %min : f32
  } -> tensor<8x1x12xf32>
  // CHECK:       return %[[RES]]
  return %0 : tensor<8x1x12xf32>
}

#map_cross_i = affine_map<(d0, d1, d2, d3) -> (d0 + d3, d1 + d2)>
#map_cross_f = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map_cross_o = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// CHECK-LABEL: @cross_conv_nonstandard_loop_order
// CHECK-SAME:    %[[ARG0:.+]]: tensor<1x15xf32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<3x1xf32>
// CHECK-SAME:    %[[ARG2:.+]]: tensor<1x12xf32>
func.func @cross_conv_nonstandard_loop_order(%input: tensor<1x15xf32>, %filter: tensor<3x1xf32>, %output: tensor<1x12xf32>) -> tensor<1x12xf32> {
  // CHECK:       %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]]
  // CHECK:       %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
  // CHECK:       %[[SLICE2:.+]] = tensor.extract_slice %[[ARG2]]
  // CHECK:       %[[SLICERES:.+]] = linalg.generic
  // CHECK-SAME:    indexing_maps = [#[[$CROSS_1D_I]], #[[$CROSS_1D_F]], #[[$CROSS_1D_O]]]
  // CHECK-SAME:    iterator_types = ["parallel", "reduction"]
  // CHECK:       %[[RES:.+]] = tensor.insert_slice %[[SLICERES]] into %[[ARG2]]
  // CHECK:       return %[[RES]]
  %0 = linalg.generic {
    indexing_maps = [#map_cross_i, #map_cross_f, #map_cross_o],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<1x15xf32>, tensor<3x1xf32>)
    outs(%output : tensor<1x12xf32>) {
  ^bb0(%in: f32, %fil: f32, %out: f32):
    %mul = arith.mulf %in, %fil : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<1x12xf32>
  return %0 : tensor<1x12xf32>
}

func.func @softmax(%arg0: tensor<2x16x32xf32>, %dst: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
  %1 = linalg.softmax dimension(2) ins(%arg0 : tensor<2x16x32xf32>) outs(%dst: tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %1 : tensor<2x16x32xf32>
}

// CHECK-LABEL:      func.func @softmax(
// CHECK-SAME:           %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>, %[[DST:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK-DAG:        %[[D1:.+]] = tensor.empty() : tensor<2x16xf32>
// CHECK-DAG:        %[[CST:.+]] = arith.constant -3.40282347E+38 : f32
// CHECK:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK:        %[[D3:.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel", "reduction"]} ins(%[[ARG0]] : tensor<2x16x32xf32>) outs(%[[D2]] : tensor<2x16xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[D8:.+]] = arith.maxnumf %[[IN]], %[[OUT]] : f32
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.decompose %0 : (!transform.any_op) -> !transform.any_op

    %2 = transform.structured.match ops{["linalg.softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.decompose_interface %2 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#input_map = affine_map<(oh, ow, f, kh, kw, c) -> (oh + kh, ow + kw, c)>
#filter_map = affine_map<(oh, ow, f, kh, kw, c) -> (kh, kw, c, f)>
#output_map = affine_map<(oh, ow, f, kh, kw, c) -> (oh, ow, f)>
#inner_map = affine_map<(i, j) -> (i, j)>

// Here ow and kw are removed instead. The nested generic's own index 1 must
// remain unchanged, while its use of the captured outer f index is updated.
// CHECK-LABEL: @batchless_conv_2d_nested_indices
func.func @batchless_conv_2d_nested_indices(%input: tensor<3x1x1xi32>, %filter: tensor<1x1x1x2xi32>, %init: tensor<3x1x2xi32>, %nested_init: tensor<2x2xi32>) -> tensor<3x1x2xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[CONV:.+]] = linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  // CHECK: ^bb0(%[[IN:.+]]: i32, %[[FILTER:.+]]: i32, %[[ACC:.+]]: i32):
  // CHECK-NEXT: %[[OH:.+]] = linalg.index 0 : index
  // CHECK-NEXT: %[[OW:.+]] = arith.constant 0 : index
  // CHECK-NEXT: %[[F:.+]] = linalg.index 1 : index
  // CHECK-NEXT: %[[SPATIAL:.+]] = arith.addi %[[OH]], %[[OW]] : index
  // CHECK-NEXT: %[[POSITION:.+]] = arith.index_cast %[[SPATIAL]] : index to i32
  // CHECK-NEXT: %[[INNER:.+]] = linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK: ^bb0
  // CHECK-NEXT: %[[INNER_INDEX:.+]] = linalg.index 1 : index
  // CHECK-NEXT: %[[WITH_F:.+]] = arith.addi %[[INNER_INDEX]], %[[F]] : index
  // CHECK-NEXT: %[[INNER_VALUE:.+]] = arith.index_cast %[[WITH_F]] : index to i32
  // CHECK-NEXT: linalg.yield %[[INNER_VALUE]] : i32
  // CHECK: %[[EXTRACTED:.+]] = tensor.extract %[[INNER]]
  // CHECK-NEXT: %[[PRODUCT:.+]] = arith.muli %[[IN]], %[[FILTER]] : i32
  // CHECK-NEXT: %[[SUM:.+]] = arith.addi %[[ACC]], %[[PRODUCT]] : i32
  // CHECK-NEXT: %[[INDEXED:.+]] = arith.addi %[[SUM]], %[[POSITION]] : i32
  // CHECK-NEXT: %[[VALUE:.+]] = arith.subi %[[INDEXED]], %[[EXTRACTED]] : i32
  // CHECK-NEXT: linalg.yield %[[VALUE]] : i32
  // CHECK: %[[RESULT:.+]] = tensor.insert_slice %[[CONV]]
  // CHECK: return %[[RESULT]] : tensor<3x1x2xi32>
  %result = linalg.generic {
    decompose_target,
    indexing_maps = [#input_map, #filter_map, #output_map],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%input, %filter : tensor<3x1x1xi32>, tensor<1x1x1x2xi32>)
    outs(%init : tensor<3x1x2xi32>) {
  ^bb0(%a: i32, %b: i32, %acc: i32):
    %oh = linalg.index 0 : index
    %ow = linalg.index 1 : index
    %f = linalg.index 2 : index
    %spatial = arith.addi %oh, %ow : index
    %position = arith.index_cast %spatial : index to i32
    %inner = linalg.generic {
      indexing_maps = [#inner_map], iterator_types = ["parallel", "parallel"]
    } outs(%nested_init : tensor<2x2xi32>) {
    ^bb0(%unused: i32):
      %inner_index = linalg.index 1 : index
      %with_f = arith.addi %inner_index, %f : index
      %inner_value = arith.index_cast %with_f : index to i32
      linalg.yield %inner_value : i32
    } -> tensor<2x2xi32>
    %extracted = tensor.extract %inner[%c0, %c1] : tensor<2x2xi32>
    %product = arith.muli %a, %b : i32
    %sum = arith.addi %acc, %product : i32
    %indexed = arith.addi %sum, %position : i32
    %value = arith.subi %indexed, %extracted : i32
    linalg.yield %value : i32
  } -> tensor<3x1x2xi32>
  return %result : tensor<3x1x2xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match attributes{decompose_target} in %root : (!transform.any_op) -> !transform.any_op
    %decomposed = transform.structured.decompose %target : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
