// RUN: mlir-opt %s -transform-interpreter -canonicalize --split-input-file | FileCheck %s

func.func @conv2d(%arg0: tensor<2x10x10x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x8x8x2xf32> {
  %0 = tensor.empty() : tensor<2x8x8x2xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1xf32>) outs(%0 : tensor<2x8x8x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x8x8x2xf32>
  %2 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x10x10x5xf32>, tensor<2x3x3x5xf32>) outs(%1 : tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %2 : tensor<2x8x8x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_fhwc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.winograd_conv2d %0 { m = 4, r = 3 } : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @conv2d
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x10x10x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<1xf32>) -> tensor<2x8x8x2xf32> {
// CHECK:        %[[S0:.*]] = tensor.empty() : tensor<2x8x8x2xf32>
// CHECK-NEXT:   %[[S1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG2]] : tensor<1xf32>) outs(%[[S0]] : tensor<2x8x8x2xf32>) {
// CHECK-NEXT:   ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:     linalg.yield %[[IN]] : f32
// CHECK-NEXT:   } -> tensor<2x8x8x2xf32>
// CHECK-NEXT:   %[[S2:.*]] = tensor.empty() : tensor<2x2x6x6x5x2xf32>
// CHECK-NEXT:   %[[S3:.*]] = linalg.winograd_filter_transform m(4) r(3) ins(%[[ARG1]] : tensor<2x3x3x5xf32>) outs(%[[S2]] : tensor<2x2x6x6x5x2xf32>) -> tensor<2x2x6x6x5x2xf32>
// CHECK-NEXT:   %[[S4:.*]] = tensor.empty() : tensor<2x2x6x6x2x5xf32>
// CHECK-NEXT:   %[[S5:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[ARG0]] : tensor<2x10x10x5xf32>) outs(%[[S4]] : tensor<2x2x6x6x2x5xf32>) -> tensor<2x2x6x6x2x5xf32>
// CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<2x2x6x6x5x2xf32> into tensor<144x5x2xf32>
// CHECK-NEXT:   %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<2x2x6x6x2x5xf32> into tensor<144x2x5xf32>
// CHECK-NEXT:   %[[S6:.*]] = tensor.empty() : tensor<144x2x2xf32>
// CHECK-NEXT:   %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_0]], %[[COLLAPSED]] : tensor<144x2x5xf32>, tensor<144x5x2xf32>) outs(%[[S6]] : tensor<144x2x2xf32>) -> tensor<144x2x2xf32>
// CHECK-NEXT:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1, 2, 3], [4], [5]] output_shape [2, 2, 6, 6, 2, 2] : tensor<144x2x2xf32> into tensor<2x2x6x6x2x2xf32>
// CHECK-NEXT:   %[[S8:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXPANDED]] : tensor<2x2x6x6x2x2xf32>) outs(%[[S1]] : tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
// CHECK-NEXT:   return %[[S8]] : tensor<2x8x8x2xf32>
// CHECK-NEXT: }

// -----

func.func @conv2d_unaligned(%arg0: tensor<2x11x11x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x9x9x2xf32> {
  %0 = tensor.empty() : tensor<2x9x9x2xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1xf32>) outs(%0 : tensor<2x9x9x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x9x9x2xf32>
  %2 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x11x11x5xf32>, tensor<2x3x3x5xf32>) outs(%1 : tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32>
  return %2 : tensor<2x9x9x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_fhwc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.winograd_conv2d %0 { m = 4, r = 3 } : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @conv2d_unaligned
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x11x11x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<1xf32>) -> tensor<2x9x9x2xf32> {
// CHECK:        %[[S0:.*]] = tensor.empty() : tensor<2x9x9x2xf32>
// CHECK-NEXT:   %[[S1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG2]] : tensor<1xf32>) outs(%[[S0]] : tensor<2x9x9x2xf32>) {
// CHECK-NEXT:   ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:     linalg.yield %[[IN]] : f32
// CHECK-NEXT:   } -> tensor<2x9x9x2xf32>
// CHECK-NEXT:   %[[S2:.*]] = tensor.empty() : tensor<3x3x6x6x5x2xf32>
// CHECK-NEXT:   %[[S3:.*]] = linalg.winograd_filter_transform m(4) r(3) ins(%[[ARG1]] : tensor<2x3x3x5xf32>) outs(%[[S2]] : tensor<3x3x6x6x5x2xf32>) -> tensor<3x3x6x6x5x2xf32>
// CHECK-NEXT:   %[[INPUT_BUF:.*]] = tensor.empty() : tensor<2x14x14x5xf32>
// CHECK-NEXT:   %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[ARG0]] into %[[INPUT_BUF]][0, 0, 0, 0] [2, 11, 11, 5] [1, 1, 1, 1] : tensor<2x11x11x5xf32> into tensor<2x14x14x5xf32>
// CHECK-NEXT:   %[[S4:.*]] = tensor.empty() : tensor<3x3x6x6x2x5xf32>
// CHECK-NEXT:   %[[S5:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[INSERTED_SLICE]] : tensor<2x14x14x5xf32>) outs(%[[S4]] : tensor<3x3x6x6x2x5xf32>) -> tensor<3x3x6x6x2x5xf32>
// CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<3x3x6x6x5x2xf32> into tensor<324x5x2xf32>
// CHECK-NEXT:   %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<3x3x6x6x2x5xf32> into tensor<324x2x5xf32>
// CHECK-NEXT:   %[[S6:.*]] = tensor.empty() : tensor<324x2x2xf32>
// CHECK-NEXT:   %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_0]], %[[COLLAPSED]] : tensor<324x2x5xf32>, tensor<324x5x2xf32>) outs(%[[S6]] : tensor<324x2x2xf32>) -> tensor<324x2x2xf32>
// CHECK-NEXT:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1, 2, 3], [4], [5]] output_shape [3, 3, 6, 6, 2, 2] : tensor<324x2x2xf32> into tensor<3x3x6x6x2x2xf32>
// CHECK-NEXT:   %[[OUTPUT_BUF:.*]] = tensor.empty() : tensor<2x12x12x2xf32>
// CHECK-NEXT:   %[[INSERTED_SLICE_2:.*]] = tensor.insert_slice %[[S1]] into %[[OUTPUT_BUF]][0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1] : tensor<2x9x9x2xf32> into tensor<2x12x12x2xf32>
// CHECK-NEXT:   %[[S8:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXPANDED]] : tensor<3x3x6x6x2x2xf32>) outs(%[[INSERTED_SLICE_2]] : tensor<2x12x12x2xf32>) -> tensor<2x12x12x2xf32>
// CHECK-NEXT:   %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[S8]][0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1] : tensor<2x12x12x2xf32> to tensor<2x9x9x2xf32>
// CHECK-NEXT:   return %[[EXTRACTED_SLICE]] : tensor<2x9x9x2xf32>
// CHECK-NEXT: }
