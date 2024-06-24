// RUN: mlir-opt %s -split-input-file -test-linalg-transform-patterns=test-winograd-conv2d | FileCheck %s

func.func @conv2d_4x4_3x3(%arg0: tensor<2x6x6x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>, %out: tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x6x6x5xf32>, tensor<2x3x3x5xf32>) outs(%out : tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32>
  return %0 : tensor<2x4x4x2xf32>
}

// CHECK-LABEL: func.func @conv2d_4x4_3x3
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x6x6x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<1xf32>, %[[ARG3:.*]]: tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32> {
// CHECK-NEXT:  %[[S2:.*]] = tensor.empty() : tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:  %[[S3:.*]] = linalg.winograd_filter_transform m(4) r(3) ins(%[[ARG1]] : tensor<2x3x3x5xf32>) outs(%[[S2]] : tensor<1x1x6x6x5x2xf32>) -> tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:  %[[S4:.*]] = tensor.empty() : tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:  %[[S5:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[ARG0]] : tensor<2x6x6x5xf32>) outs(%[[S4]] : tensor<1x1x6x6x2x5xf32>) -> tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:  %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x6x5x2xf32> into tensor<36x5x2xf32>
// CHECK-NEXT:  %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x6x2x5xf32> into tensor<36x2x5xf32>
// CHECK-NEXT:  %[[S6:.*]] = tensor.empty() : tensor<36x2x2xf32>
// CHECK-NEXT:  %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_0]], %[[COLLAPSED]] : tensor<36x2x5xf32>, tensor<36x5x2xf32>) outs(%[[S6]] : tensor<36x2x2xf32>) -> tensor<36x2x2xf32>
// CHECK-NEXT:  %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1, 2, 3], [4], [5]] output_shape [1, 1, 6, 6, 2, 2] : tensor<36x2x2xf32> into tensor<1x1x6x6x2x2xf32>
// CHECK-NEXT:  %[[S8:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXPANDED]] : tensor<1x1x6x6x2x2xf32>) outs(%[[ARG3]] : tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32>
// CHECK-NEXT:  return %[[S8]] : tensor<2x4x4x2xf32>
// CHECK-NEXT: }

// -----

func.func @conv2d_2x2_5x5(%arg0: tensor<2x6x6x5xf32>, %arg1: tensor<2x5x5x5xf32>, %arg2: tensor<1xf32>, %out: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x6x6x5xf32>, tensor<2x5x5x5xf32>) outs(%out : tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  return %0 : tensor<2x2x2x2xf32>
}

// CHECK-LABEL: func.func @conv2d_2x2_5x5
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x6x6x5xf32>, %[[ARG1:.*]]: tensor<2x5x5x5xf32>, %[[ARG2:.*]]: tensor<1xf32>, %[[ARG3:.*]]: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
// CHECK-NEXT:   %[[S2:.*]] = tensor.empty() : tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:   %[[S3:.*]] = linalg.winograd_filter_transform m(2) r(5) ins(%[[ARG1]] : tensor<2x5x5x5xf32>) outs(%[[S2]] : tensor<1x1x6x6x5x2xf32>) -> tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:   %[[S4:.*]] = tensor.empty() : tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:   %[[S5:.*]] = linalg.winograd_input_transform m(2) r(5) ins(%[[ARG0]] : tensor<2x6x6x5xf32>) outs(%[[S4]] : tensor<1x1x6x6x2x5xf32>) -> tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x6x5x2xf32> into tensor<36x5x2xf32>
// CHECK-NEXT:   %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x6x2x5xf32> into tensor<36x2x5xf32>
// CHECK-NEXT:   %[[S6:.*]] = tensor.empty() : tensor<36x2x2xf32>
// CHECK-NEXT:   %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_0]], %[[COLLAPSED]] : tensor<36x2x5xf32>, tensor<36x5x2xf32>) outs(%[[S6]] : tensor<36x2x2xf32>) -> tensor<36x2x2xf32>
// CHECK-NEXT:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1, 2, 3], [4], [5]] output_shape [1, 1, 6, 6, 2, 2] : tensor<36x2x2xf32> into tensor<1x1x6x6x2x2xf32>
// CHECK-NEXT:   %[[S8:.*]] = linalg.winograd_output_transform m(2) r(5) ins(%[[EXPANDED]] : tensor<1x1x6x6x2x2xf32>) outs(%[[ARG3]] : tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
// CHECK-NEXT:   return %[[S8]] : tensor<2x2x2x2xf32>
// CHECK-NEXT: }

// -----

func.func @conv2d_1x4_1x3(%arg0: tensor<2x1x6x5xf32>, %arg1: tensor<2x1x3x5xf32>, %arg2: tensor<1xf32>, %out: tensor<2x1x4x2xf32>) -> tensor<2x1x4x2xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x1x6x5xf32>, tensor<2x1x3x5xf32>) outs(%out : tensor<2x1x4x2xf32>) -> tensor<2x1x4x2xf32>
  return %0 : tensor<2x1x4x2xf32>
}

// CHECK-LABEL: func.func @conv2d_1x4_1x3
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x1x6x5xf32>, %[[ARG1:.*]]: tensor<2x1x3x5xf32>, %[[ARG2:.*]]: tensor<1xf32>, %[[ARG3:.*]]: tensor<2x1x4x2xf32>) -> tensor<2x1x4x2xf32> {
// CHECK-NEXT:   %[[S2:.*]] = tensor.empty() : tensor<1x1x1x6x5x2xf32>
// CHECK-NEXT:   %[[S3:.*]] = linalg.winograd_filter_transform m(4) r(3) ins(%[[ARG1]] : tensor<2x1x3x5xf32>) outs(%[[S2]] : tensor<1x1x1x6x5x2xf32>) -> tensor<1x1x1x6x5x2xf32>
// CHECK-NEXT:   %[[S4:.*]] = tensor.empty() : tensor<1x1x1x6x2x5xf32>
// CHECK-NEXT:   %[[S5:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[ARG0]] : tensor<2x1x6x5xf32>) outs(%[[S4]] : tensor<1x1x1x6x2x5xf32>) -> tensor<1x1x1x6x2x5xf32>
// CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<1x1x1x6x5x2xf32> into tensor<6x5x2xf32>
// CHECK-NEXT:   %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<1x1x1x6x2x5xf32> into tensor<6x2x5xf32>
// CHECK-NEXT:   %[[S6:.*]] = tensor.empty() : tensor<6x2x2xf32>
// CHECK-NEXT:   %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_0]], %[[COLLAPSED]] : tensor<6x2x5xf32>, tensor<6x5x2xf32>) outs(%[[S6]] : tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
// CHECK-NEXT:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1, 2, 3], [4], [5]] output_shape [1, 1, 1, 6, 2, 2] : tensor<6x2x2xf32> into tensor<1x1x1x6x2x2xf32>
// CHECK-NEXT:   %[[S8:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXPANDED]] : tensor<1x1x1x6x2x2xf32>) outs(%[[ARG3]] : tensor<2x1x4x2xf32>) -> tensor<2x1x4x2xf32>
// CHECK-NEXT:   return %[[S8]] : tensor<2x1x4x2xf32>
// CHECK-NEXT: }

// -----

func.func @conv2d_4x1_3x1(%arg0: tensor<2x6x1x5xf32>, %arg1: tensor<2x3x1x5xf32>, %arg2: tensor<1xf32>, %out: tensor<2x4x1x2xf32>) -> tensor<2x4x1x2xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x6x1x5xf32>, tensor<2x3x1x5xf32>) outs(%out : tensor<2x4x1x2xf32>) -> tensor<2x4x1x2xf32>
  return %0 : tensor<2x4x1x2xf32>
}

// CHECK-LABEL: func.func @conv2d_4x1_3x1
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x6x1x5xf32>, %[[ARG1:.*]]: tensor<2x3x1x5xf32>, %[[ARG2:.*]]: tensor<1xf32>, %[[ARG3:.*]]: tensor<2x4x1x2xf32>) -> tensor<2x4x1x2xf32> {
// CHECK-NEXT:   %[[S2:.*]] = tensor.empty() : tensor<1x1x6x1x5x2xf32>
// CHECK-NEXT:   %[[S3:.*]] = linalg.winograd_filter_transform m(4) r(3) ins(%[[ARG1]] : tensor<2x3x1x5xf32>) outs(%[[S2]] : tensor<1x1x6x1x5x2xf32>) -> tensor<1x1x6x1x5x2xf32>
// CHECK-NEXT:   %[[S4:.*]] = tensor.empty() : tensor<1x1x6x1x2x5xf32>
// CHECK-NEXT:   %[[S5:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[ARG0]] : tensor<2x6x1x5xf32>) outs(%[[S4]] : tensor<1x1x6x1x2x5xf32>) -> tensor<1x1x6x1x2x5xf32>
// CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x1x5x2xf32> into tensor<6x5x2xf32>
// CHECK-NEXT:   %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x1x2x5xf32> into tensor<6x2x5xf32>
// CHECK-NEXT:   %[[S6:.*]] = tensor.empty() : tensor<6x2x2xf32>
// CHECK-NEXT:   %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_0]], %[[COLLAPSED]] : tensor<6x2x5xf32>, tensor<6x5x2xf32>) outs(%[[S6]] : tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
// CHECK-NEXT:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1, 2, 3], [4], [5]] output_shape [1, 1, 6, 1, 2, 2] : tensor<6x2x2xf32> into tensor<1x1x6x1x2x2xf32>
// CHECK-NEXT:   %[[S8:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXPANDED]] : tensor<1x1x6x1x2x2xf32>) outs(%[[ARG3]] : tensor<2x4x1x2xf32>) -> tensor<2x4x1x2xf32>
// CHECK-NEXT:   return %[[S8]] : tensor<2x4x1x2xf32>
// CHECK-NEXT: }

// -----

func.func @conv2d_aligned(%arg0: tensor<2x10x10x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>, %out: tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x10x10x5xf32>, tensor<2x3x3x5xf32>) outs(%out : tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}

// CHECK-LABEL: func.func @conv2d_aligned
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x10x10x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<1xf32>, %[[ARG3:.*]]: tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32> {
// CHECK-NEXT:  %[[S2:.*]] = tensor.empty() : tensor<2x2x6x6x5x2xf32>
// CHECK-NEXT:  %[[S3:.*]] = linalg.winograd_filter_transform m(4) r(3) ins(%[[ARG1]] : tensor<2x3x3x5xf32>) outs(%[[S2]] : tensor<2x2x6x6x5x2xf32>) -> tensor<2x2x6x6x5x2xf32>
// CHECK-NEXT:  %[[S4:.*]] = tensor.empty() : tensor<2x2x6x6x2x5xf32>
// CHECK-NEXT:  %[[S5:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[ARG0]] : tensor<2x10x10x5xf32>) outs(%[[S4]] : tensor<2x2x6x6x2x5xf32>) -> tensor<2x2x6x6x2x5xf32>
// CHECK-NEXT:  %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<2x2x6x6x5x2xf32> into tensor<144x5x2xf32>
// CHECK-NEXT:  %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<2x2x6x6x2x5xf32> into tensor<144x2x5xf32>
// CHECK-NEXT:  %[[S6:.*]] = tensor.empty() : tensor<144x2x2xf32>
// CHECK-NEXT:  %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_0]], %[[COLLAPSED]] : tensor<144x2x5xf32>, tensor<144x5x2xf32>) outs(%[[S6]] : tensor<144x2x2xf32>) -> tensor<144x2x2xf32>
// CHECK-NEXT:  %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1, 2, 3], [4], [5]] output_shape [2, 2, 6, 6, 2, 2] : tensor<144x2x2xf32> into tensor<2x2x6x6x2x2xf32>
// CHECK-NEXT:  %[[S8:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXPANDED]] : tensor<2x2x6x6x2x2xf32>) outs(%[[ARG3]] : tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
// CHECK-NEXT:  return %[[S8]] : tensor<2x8x8x2xf32>
// CHECK-NEXT: }

// -----

func.func @conv2d_unaligned(%arg0: tensor<2x11x11x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>, %out: tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x11x11x5xf32>, tensor<2x3x3x5xf32>) outs(%out : tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32>
  return %0 : tensor<2x9x9x2xf32>
}

// CHECK-LABEL: func.func @conv2d_unaligned
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x11x11x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<1xf32>, %[[ARG3:.*]]: tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32> {
// CHECK-NEXT:  %[[S2:.*]] = tensor.empty() : tensor<3x3x6x6x5x2xf32>
// CHECK-NEXT:  %[[S3:.*]] = linalg.winograd_filter_transform m(4) r(3) ins(%[[ARG1]] : tensor<2x3x3x5xf32>) outs(%[[S2]] : tensor<3x3x6x6x5x2xf32>) -> tensor<3x3x6x6x5x2xf32>
// CHECK-NEXT:  %[[INPUT_BUF:.*]] = tensor.empty() : tensor<2x14x14x5xf32>
// CHECK-NEXT:  %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[ARG0]] into %[[INPUT_BUF]][0, 0, 0, 0] [2, 11, 11, 5] [1, 1, 1, 1] : tensor<2x11x11x5xf32> into tensor<2x14x14x5xf32>
// CHECK-NEXT:  %[[S4:.*]] = tensor.empty() : tensor<3x3x6x6x2x5xf32>
// CHECK-NEXT:  %[[S5:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[INSERTED_SLICE]] : tensor<2x14x14x5xf32>) outs(%[[S4]] : tensor<3x3x6x6x2x5xf32>) -> tensor<3x3x6x6x2x5xf32>
// CHECK-NEXT:  %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<3x3x6x6x5x2xf32> into tensor<324x5x2xf32>
// CHECK-NEXT:  %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<3x3x6x6x2x5xf32> into tensor<324x2x5xf32>
// CHECK-NEXT:  %[[S6:.*]] = tensor.empty() : tensor<324x2x2xf32>
// CHECK-NEXT:  %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_0]], %[[COLLAPSED]] : tensor<324x2x5xf32>, tensor<324x5x2xf32>) outs(%[[S6]] : tensor<324x2x2xf32>) -> tensor<324x2x2xf32>
// CHECK-NEXT:  %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1, 2, 3], [4], [5]] output_shape [3, 3, 6, 6, 2, 2] : tensor<324x2x2xf32> into tensor<3x3x6x6x2x2xf32>
// CHECK-NEXT:  %[[OUTPUT_BUF:.*]] = tensor.empty() : tensor<2x12x12x2xf32>
// CHECK-NEXT:  %[[INSERTED_SLICE_2:.*]] = tensor.insert_slice %[[ARG3]] into %[[OUTPUT_BUF]][0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1] : tensor<2x9x9x2xf32> into tensor<2x12x12x2xf32>
// CHECK-NEXT:  %[[S8:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXPANDED]] : tensor<3x3x6x6x2x2xf32>) outs(%[[INSERTED_SLICE_2]] : tensor<2x12x12x2xf32>) -> tensor<2x12x12x2xf32>
// CHECK-NEXT:  %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[S8]][0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1] : tensor<2x12x12x2xf32> to tensor<2x9x9x2xf32>
// CHECK-NEXT:  return %[[EXTRACTED_SLICE]] : tensor<2x9x9x2xf32>
// CHECK-NEXT: }

// -----

func.func @conv2d_unsupported_1(%arg0: tensor<2x6x5x5xf32>, %arg1: tensor<2x3x2x5xf32>, %arg2: tensor<1xf32>, %out: tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x6x5x5xf32>, tensor<2x3x2x5xf32>) outs(%out : tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32>
  return %0 : tensor<2x4x4x2xf32>
}

// CHECK-LABEL: conv2d_unsupported_1
// CHECK: linalg.conv_2d_nhwc_fhwc

// -----

func.func @conv2d_unsupported_2(%arg0: tensor<2x7x7x5xf32>, %arg1: tensor<2x4x4x5xf32>, %arg2: tensor<1xf32>, %out: tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x7x7x5xf32>, tensor<2x4x4x5xf32>) outs(%out : tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32>
  return %0 : tensor<2x4x4x2xf32>
}

// CHECK-LABEL: conv2d_unsupported_2
// CHECK: linalg.conv_2d_nhwc_fhwc

// -----

func.func @conv2d_unsupported_3(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<2x3x3x5xf32>) outs(%arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: conv2d_unsupported_3
// CHECK: linalg.conv_2d_nhwc_fhwc
