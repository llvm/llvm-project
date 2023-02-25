// RUN: mlir-opt %s -test-transform-dialect-interpreter -split-input-file -verify-diagnostics | FileCheck %s

// Check that the im2col patterns are properly connected with the
// transform dialect.

// Non static shapes are not supported.
// Check that we emit an error.
// TODO: Hook up the rewriter errors in transform dialect.
func.func @conv_non_static(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    // expected-note@below {{when applied to this op}}
    %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<?x?x?x?xf32>, tensor<3x3x4x16xf32>)
      outs(%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %0 : tensor<?x?x?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-error@below {{failed to apply}}
  %1:2 = transform.structured.convert_conv2d_to_img2col %0 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
}

// -----

// Check that we get the proper handles for the img2col tensor producer
// and the final instruction.

// CHECK: IR printer: tensor_producer
// CHECK-NEXT: %[[COL_TENSOR:.+]] = linalg.generic
// CHECK-SAME: affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3, d2 + d4, d5)>,
// CHECK-SAME: affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>]
// CHECK: ^bb0(%[[IN_DATA:.+]]: f32, %[[OUT_DATA:.+]]: f32)
// CHECK: linalg.yield %[[IN_DATA]] : f32

// CHECK: IR printer: transformed
// CHECK: tensor.expand_shape %{{[^ ]*}} {{\[}}[0, 1, 2], [3]] : tensor<196x16xf32> into tensor<1x14x14x16xf32>

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3, d2 + d4, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//      CHECK: @conv_16433136
//      CHECK: %[[INPUT:.+]]: tensor<1x16x16x4xf32>
//      CHECK: %[[FILTER:.+]]: tensor<3x3x4x16xf32>
//      CHECK: %[[OUTPUT:.+]]: tensor<1x14x14x16xf32>
//      CHECK: %[[INIT_COL_TENSOR:.+]] = tensor.empty() : tensor<1x14x14x3x3x4xf32>
//      CHECK: %[[COL_TENSOR:.+]] = linalg.generic
//           CHECK-SAME: #[[MAP0]]
//           CHECK-SAME: #[[MAP1]]
//                CHECK: ^bb0(%[[IN_DATA:.+]]: f32, %[[OUT_DATA:.+]]: f32)
//                CHECK: linalg.yield %[[IN_DATA]] : f32
//      CHECK-DAG: %[[RESHAPED_INIT_COL_TENSOR:.+]] = tensor.collapse_shape %[[COL_TENSOR]]
//           CHECK-SAME: [0, 1, 2], [3, 4, 5]
//           CHECK-SAME: tensor<1x14x14x3x3x4xf32> into tensor<196x36xf32>
//      CHECK-DAG: %[[RESHAPED_FILTER:.+]] = tensor.collapse_shape %[[FILTER]]
//           CHECK-SAME: [0, 1, 2], [3]
//           CHECK-SAME: tensor<3x3x4x16xf32> into tensor<36x16xf32>
//      CHECK-DAG: %[[RESHAPED_OUTPUT:.+]] = tensor.collapse_shape %[[OUTPUT]]
//           CHECK-SAME: [0, 1, 2], [3]
//      CHECK: %[[MATMUL_RESULT:.+]] = linalg.matmul ins(%[[RESHAPED_INIT_COL_TENSOR]], %[[RESHAPED_FILTER]] : tensor<196x36xf32>, tensor<36x16xf32>) outs(%[[RESHAPED_OUTPUT]] : tensor<196x16xf32>)
//      CHECK: %[[RESULT:.+]] = tensor.expand_shape %[[MATMUL_RESULT]] {{\[}}[0, 1, 2], [3]] : tensor<196x16xf32> into tensor<1x14x14x16xf32>
//      CHECK: return %[[RESULT]]

func.func @conv_16433136(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
    %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
      outs(%arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
    return %0 : tensor<1x14x14x16xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %img2col_tensor_producer, %transformed = transform.structured.convert_conv2d_to_img2col %0 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  transform.print %img2col_tensor_producer {name = "tensor_producer"}: !pdl.operation
  transform.print %transformed {name = "transformed"}: !pdl.operation
}

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 + d4, d3 + d5)>
// CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-DAG: #[[MAP6:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK: @depthwise_conv_hwc_114x16x3
// CHECK-SAME: %[[INPUT:.+]]: tensor<1x114x114x16xf32>
// CHECK-SAME: %[[FILTER:.+]]: tensor<3x3x16xf32>
// CHECK-SAME: %[[OUTPUT:.+]]: tensor<1x112x112x16xf32>
//      CHECK: %[[INPUT_T_INIT:.+]] = tensor.empty() : tensor<1x16x114x114xf32>
//      CHECK: %[[INPUT_T:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[INPUT]] : tensor<1x114x114x16xf32>) outs(%[[INPUT_T_INIT]] : tensor<1x16x114x114xf32>) {
// CHECK-NEXT: ^bb0(%[[ARG3:.+]]: f32, %[[ARG4:.+]]: f32):
// CHECK-NEXT:     linalg.yield %[[ARG3]] : f32
// CHECK-NEXT:  } -> tensor<1x16x114x114xf32>
//      CHECK: %[[FILTER_T_INIT:.+]] = tensor.empty() : tensor<16x3x3xf32>
//      CHECK: %[[FILTER_T:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP2]], #[[MAP3]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[FILTER]] : tensor<3x3x16xf32>) outs(%[[FILTER_T_INIT]] : tensor<16x3x3xf32>) {
// CHECK-NEXT:      ^bb0(%{{.*}}: f32, %{{.*}}: f32):
//      CHECK:      linalg.yield
//      CHECK:    } -> tensor<16x3x3xf32>
//      CHECK: %[[INIT_OUTPUT_TENSOR:.+]] = tensor.empty() : tensor<1x16x112x112xf32>
//      CHECK: %[[OUTPUT_T:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[OUTPUT]] : tensor<1x112x112x16xf32>) outs(%[[INIT_OUTPUT_TENSOR]] : tensor<1x16x112x112xf32>) {
// CHECK-NEXT:  ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:     linalg.yield
// CHECK-NEXT:  } -> tensor<1x16x112x112xf32>
//      CHECK:  %[[INIT_COL_TENSOR:.+]] = tensor.empty() : tensor<1x16x112x112x3x3xf32>
//      CHECK: %[[COL_TENSOR:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP4]], #[[MAP5]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:   ins(%[[INPUT_T]] : tensor<1x16x114x114xf32>) outs(%[[INIT_COL_TENSOR]] : tensor<1x16x112x112x3x3xf32>) {
// CHECK-NEXT:      ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:         linalg.yield
// CHECK-NEXT:    } -> tensor<1x16x112x112x3x3xf32>
//      CHECK: %[[COL_TENSOR_R:.+]] = tensor.collapse_shape %[[COL_TENSOR]]
// CHECK-SAME:    tensor<1x16x112x112x3x3xf32> into tensor<16x12544x9xf32>
//      CHECK: %[[FILTER_T_R:.+]] = tensor.collapse_shape %[[FILTER_T]]
// CHECK-SAME:    tensor<16x3x3xf32> into tensor<16x9xf32>
//      CHECK: %[[OUTPUT_T_R:.+]] = tensor.collapse_shape %[[OUTPUT_T]]
// CHECK-SAME:    tensor<1x16x112x112xf32> into tensor<16x12544xf32>
//      CHECK: %[[BMV_RESULT:.+]] = linalg.batch_matvec ins(%[[COL_TENSOR_R]], %[[FILTER_T_R]] : tensor<16x12544x9xf32>, tensor<16x9xf32>) outs(%[[OUTPUT_T_R]] : tensor<16x12544xf32>) -> tensor<16x12544xf32>
//      CHECK: %[[RESULT_R:.+]] = tensor.expand_shape %[[BMV_RESULT]]
// CHECK-SAME:    tensor<16x12544xf32> into tensor<1x16x112x112xf32>
//      CHECK: %[[RESULT_INIT:.+]] = tensor.empty() : tensor<1x112x112x16xf32>
//      CHECK: %[[RESULT:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP6]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[RESULT_R]] : tensor<1x16x112x112xf32>) outs(%[[RESULT_INIT]] : tensor<1x112x112x16xf32>) {
// CHECK-NEXT:      ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:      linalg.yield
// CHECK-NEXT:    } -> tensor<1x112x112x16xf32>
//      CHECK: return %[[RESULT]] : tensor<1x112x112x16xf32>
func.func @depthwise_conv_hwc_114x16x3(%input: tensor<1x114x114x16xf32>, %filter: tensor<3x3x16xf32>, %output: tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32> {
    %0 = linalg.depthwise_conv_2d_nhwc_hwc {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } ins(%input, %filter : tensor<1x114x114x16xf32>, tensor<3x3x16xf32>) outs(%output : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
    return %0 : tensor<1x112x112x16xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.depthwise_conv_2d_nhwc_hwc"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1:2 = transform.structured.convert_conv2d_to_img2col %0 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
}

// -----

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3, d2 + d4, d5)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//  CHECK-DAG: #[[LHSMAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[RHSMAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
//  CHECK-DAG: #[[RESMAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

//      CHECK: func.func @batch_nhwc_conv
// CHECK-SAME: (%[[INPUT:.+]]: tensor<8x16x16x4xf32>, %[[FILTER:.+]]: tensor<3x3x4x16xf32>, %[[INIT:.+]]: tensor<8x14x14x16xf32>)
//      CHECK:   %[[IT:.+]] = tensor.empty() : tensor<8x14x14x3x3x4xf32>
//      CHECK:   %[[IMG2COL:.+]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:   ins(%[[INPUT]] : tensor<8x16x16x4xf32>)
// CHECK-SAME:   outs(%[[IT]] : tensor<8x14x14x3x3x4xf32>)
//      CHECK:   %[[CS_INPUT:.+]] = tensor.collapse_shape %[[IMG2COL]] {{\[}}[0], [1, 2], [3, 4, 5]] : tensor<8x14x14x3x3x4xf32> into tensor<8x196x36xf32>
//      CHECK:   %[[CS_FILTER:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0, 1, 2], [3]] : tensor<3x3x4x16xf32> into tensor<36x16xf32>
//      CHECK:   %[[CS_RESULT:.+]] = tensor.collapse_shape %[[INIT]] {{\[}}[0], [1, 2], [3]] : tensor<8x14x14x16xf32> into tensor<8x196x16xf32>
//      CHECK:   %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[LHSMAP]], #[[RHSMAP]], #[[RESMAP]]],
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[CS_INPUT]], %[[CS_FILTER]] : tensor<8x196x36xf32>, tensor<36x16xf32>)
// CHECK-SAME:   outs(%[[CS_RESULT]] : tensor<8x196x16xf32>)
//      CHECK:   ^bb0(%[[ARG0:.+]]: f32, %[[ARG1:.+]]: f32, %[[ARG2:.+]]: f32):
//      CHECK:     %[[MUL:.+]] = arith.mulf %[[ARG0]], %[[ARG1]] : f32
//      CHECK:     %[[ADD:.+]] = arith.addf %[[MUL]], %[[ARG2]] : f32
//      CHECK:     linalg.yield %[[ADD]] : f32
//      CHECK:   } -> tensor<8x196x16xf32>
//      CHECK:   %[[CS_FINAL:.+]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0], [1, 2], [3]] : tensor<8x196x16xf32> into tensor<8x14x14x16xf32>
//      CHECK:   return %[[CS_FINAL]]
func.func @batch_nhwc_conv(%arg0: tensor<8x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<8x14x14x16xf32>) -> tensor<8x14x14x16xf32> {
    %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<8x16x16x4xf32>, tensor<3x3x4x16xf32>)
      outs(%arg2: tensor<8x14x14x16xf32>) -> tensor<8x14x14x16xf32>
    return %0 : tensor<8x14x14x16xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1:2 = transform.structured.convert_conv2d_to_img2col %0 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
}

// -----

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4 + d2, d5 + d3)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//  CHECK-DAG: #[[LHSMAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
//  CHECK-DAG: #[[RHSMAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[RESMAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

//      CHECK: func.func @batch_nchw_conv
// CHECK-SAME: (%[[INPUT:.+]]: tensor<8x4x16x16xf32>, %[[FILTER:.+]]: tensor<16x4x3x3xf32>, %[[INIT:.+]]: tensor<8x16x14x14xf32>)
//      CHECK:   %[[IT:.+]] = tensor.empty() : tensor<8x4x3x3x14x14xf32>
//      CHECK:   %[[IMG2COL:.+]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:   ins(%[[INPUT]] : tensor<8x4x16x16xf32>)
// CHECK-SAME:   outs(%[[IT]] : tensor<8x4x3x3x14x14xf32>)
//      CHECK:   %[[CS_FILTER:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0], [1, 2, 3]] : tensor<16x4x3x3xf32> into tensor<16x36xf32>
//      CHECK:   %[[CS_INPUT:.+]] = tensor.collapse_shape %[[IMG2COL]] {{\[}}[0], [1, 2, 3], [4, 5]] : tensor<8x4x3x3x14x14xf32> into tensor<8x36x196xf32>
//      CHECK:   %[[CS_RESULT:.+]] = tensor.collapse_shape %[[INIT]] {{\[}}[0], [1], [2, 3]] : tensor<8x16x14x14xf32> into tensor<8x16x196xf32>
//      CHECK:   %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[LHSMAP]], #[[RHSMAP]], #[[RESMAP]]],
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[CS_FILTER]], %[[CS_INPUT]] : tensor<16x36xf32>, tensor<8x36x196xf32>)
// CHECK-SAME:   outs(%[[CS_RESULT]] : tensor<8x16x196xf32>)
//      CHECK:   ^bb0(%[[ARG0:.+]]: f32, %[[ARG1:.+]]: f32, %[[ARG2:.+]]: f32):
//      CHECK:     %[[MUL:.+]] = arith.mulf %[[ARG0]], %[[ARG1]] : f32
//      CHECK:     %[[ADD:.+]] = arith.addf %[[MUL]], %[[ARG2]] : f32
//      CHECK:     linalg.yield %[[ADD]] : f32
//      CHECK:   } -> tensor<8x16x196xf32>
//      CHECK:   %[[CS_FINAL:.+]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0], [1], [2, 3]] : tensor<8x16x196xf32> into tensor<8x16x14x14xf32>
//      CHECK:   return %[[CS_FINAL]]
func.func @batch_nchw_conv(%arg0: tensor<8x4x16x16xf32>, %arg1: tensor<16x4x3x3xf32>, %arg2: tensor<8x16x14x14xf32>) -> tensor<8x16x14x14xf32> {
    %0 = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<8x4x16x16xf32>, tensor<16x4x3x3xf32>)
      outs(%arg2: tensor<8x16x14x14xf32>) -> tensor<8x16x14x14xf32>
    return %0 : tensor<8x16x14x14xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1:2 = transform.structured.convert_conv2d_to_img2col %0 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
}
