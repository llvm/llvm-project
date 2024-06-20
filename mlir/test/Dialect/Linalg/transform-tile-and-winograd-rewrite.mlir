// RUN: mlir-opt %s -transform-interpreter -canonicalize --split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv2d(%arg0: tensor<2x10x10x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x8x8x2xf32> {
  %0 = tensor.empty() : tensor<2x8x8x2xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1xf32>) outs(%0 : tensor<2x8x8x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x8x8x2xf32>
  %2 = tensor.empty() : tensor<2x2x6x6x5x2xf32>
  %3 = linalg.winograd_filter_transform m(4) r(3) ins(%arg1 : tensor<2x3x3x5xf32>) outs(%2 : tensor<2x2x6x6x5x2xf32>) -> tensor<2x2x6x6x5x2xf32>
  %4 = tensor.empty() : tensor<2x2x6x6x2x5xf32>
  %5 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x10x10x5xf32>) outs(%4 : tensor<2x2x6x6x2x5xf32>) -> tensor<2x2x6x6x2x5xf32>
  %collapsed = tensor.collapse_shape %3 [[0, 1, 2, 3], [4], [5]] : tensor<2x2x6x6x5x2xf32> into tensor<144x5x2xf32>
  %collapsed_0 = tensor.collapse_shape %5 [[0, 1, 2, 3], [4], [5]] : tensor<2x2x6x6x2x5xf32> into tensor<144x2x5xf32>
  %6 = tensor.empty() : tensor<144x2x2xf32>
  %7 = linalg.batch_matmul ins(%collapsed_0, %collapsed : tensor<144x2x5xf32>, tensor<144x5x2xf32>) outs(%6 : tensor<144x2x2xf32>) -> tensor<144x2x2xf32>
  %expanded = tensor.expand_shape %7 [[0, 1, 2, 3], [4], [5]] output_shape [2, 2, 6, 6, 2, 2] : tensor<144x2x2xf32> into tensor<2x2x6x6x2x2xf32>
  %8 = linalg.winograd_output_transform m(4) r(3) ins(%expanded : tensor<2x2x6x6x2x2xf32>) outs(%1 : tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %8 : tensor<2x8x8x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:2 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %3, %loop3:2 = transform.structured.tile_using_for %2 tile_sizes [1, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %4 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %5, %loop5:2 = transform.structured.tile_using_for %4 tile_sizes [1, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %6 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %1 : (!transform.any_op) -> !transform.any_op
    %7 = transform.structured.decompose_winograd_op %6 : (!transform.any_op) -> (!transform.any_op)
    %8 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %3 : (!transform.any_op) -> !transform.any_op
    %9 = transform.structured.decompose_winograd_op %8 : (!transform.any_op) -> (!transform.any_op)
    %10 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %5 : (!transform.any_op) -> !transform.any_op
    %11 = transform.structured.decompose_winograd_op %10 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @conv2d
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x10x10x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<1xf32>) -> tensor<2x8x8x2xf32> {
// CHECK-DAG:    %[[CST:.*]] = arith.constant 1.024000e+03 : f32
// CHECK-DAG:    %[[CST_0:.*]] = arith.constant dense<{{\[}}[1.250000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00], [2.500000e-01, -2.500000e-01, 2.500000e-01, -2.500000e-01], [2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01], [1.250000e-01, -2.500000e-01, 5.000000e-01, -1.000000e+00], [1.250000e-01, 2.500000e-01, 5.000000e-01, 1.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 5.000000e-01]]> : tensor<6x4xf32>
// CHECK-DAG:    %[[CST_1:.*]] = arith.constant dense<{{\[}}[1.250000e-01, 2.500000e-01, 2.500000e-01, 1.250000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -2.500000e-01, 2.500000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 2.500000e-01, 5.000000e-01, 5.000000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -1.000000e+00, 1.000000e+00, 5.000000e-01]]> : tensor<4x6xf32>
// CHECK-DAG:    %[[CST_2:.*]] = arith.constant dense<{{\[}}[2.500000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 2.500000e-01, -2.500000e-01, 2.500000e-01, -2.500000e-01, 2.500000e-01], [-3.125000e-01, -2.500000e-01, -2.500000e-01, -1.250000e-01, -1.250000e-01, 0.000000e+00], [0.000000e+00, -6.250000e-02, 6.250000e-02, -2.500000e-01, 2.500000e-01, -3.125000e-01], [6.250000e-02, 6.250000e-02, 6.250000e-02, 1.250000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 6.250000e-02]]> : tensor<6x6xf32>
// CHECK-DAG:    %[[CST_3:.*]] = arith.constant dense<{{\[}}[2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -2.500000e-01, -6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, -2.500000e-01, -2.500000e-01, 6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -1.250000e-01, -2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, -1.250000e-01, 2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02]]> : tensor<6x6xf32>
// CHECK-DAG:    %[[CST_4:.*]] = arith.constant dense<{{\[}}[1.000000e+00, -0.333333343, -0.333333343, 0.0833333358, 0.0833333358, 0.000000e+00], [0.000000e+00, 0.333333343, -0.333333343, -0.166666672, 0.166666672, 0.000000e+00], [0.000000e+00, -0.333333343, -0.333333343, 0.333333343, 0.333333343, 1.000000e+00]]> : tensor<3x6xf32>
// CHECK-DAG:    %[[CST_5:.*]] = arith.constant dense<{{\[}}[1.000000e+00, 0.000000e+00, 0.000000e+00], [-0.333333343, 0.333333343, -0.333333343], [-0.333333343, -0.333333343, -0.333333343], [0.0833333358, -0.166666672, 0.333333343], [0.0833333358, 0.166666672, 0.333333343], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<6x3xf32>
// CHECK-DAG:    %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[S0:.*]] = tensor.empty() : tensor<2x8x8x2xf32>
// CHECK-NEXT:   %[[S1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG2]] : tensor<1xf32>) outs(%[[S0]] : tensor<2x8x8x2xf32>) {
// CHECK-NEXT:   ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:     linalg.yield %[[IN]] : f32
// CHECK-NEXT:   } -> tensor<2x8x8x2xf32>
// CHECK-NEXT:   %[[S2:.*]] = tensor.empty() : tensor<2x2x6x6x5x2xf32>
// CHECK-NEXT:   %[[S3:.*]] = tensor.empty() : tensor<2x2x6x6x5x2xf32>
// CHECK-NEXT:   %[[S4:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S3]]) -> (tensor<2x2x6x6x5x2xf32>) {
// CHECK-NEXT:     %[[S12:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<2x2x6x6x5x2xf32>) {
// CHECK-NEXT:       %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[S2]][%[[ARG3]], %[[ARG5]], 0, 0, 0, 0] [1, 1, 6, 6, 5, 2] [1, 1, 1, 1, 1, 1] : tensor<2x2x6x6x5x2xf32> to tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:       %[[S15:.*]] = scf.for %[[ARG7:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.*]] = %[[EXTRACTED_SLICE]]) -> (tensor<1x1x6x6x5x2xf32>) {
// CHECK-NEXT:         %[[S18:.*]] = scf.for %[[ARG9:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG10:.*]] = %[[ARG8]]) -> (tensor<1x1x6x6x5x2xf32>) {
// CHECK-NEXT:           %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG7]], 0, 0, %[[ARG9]]] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<2x3x3x5xf32> to tensor<1x3x3x1xf32>
// CHECK-NEXT:           %[[EXTRACTED_SLICE_8:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE_7]][0, 0, 0, 0] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<1x3x3x1xf32> to tensor<3x3xf32>
// CHECK-NEXT:           %[[S19:.*]] = tensor.empty() : tensor<6x3xf32>
// CHECK-NEXT:           %[[S20:.*]] = linalg.matmul ins(%[[CST_5]], %[[EXTRACTED_SLICE_8]] : tensor<6x3xf32>, tensor<3x3xf32>) outs(%[[S19]] : tensor<6x3xf32>) -> tensor<6x3xf32>
// CHECK-NEXT:           %[[S21:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:           %[[S22:.*]] = linalg.matmul ins(%[[S20]], %[[CST_4]] : tensor<6x3xf32>, tensor<3x6xf32>) outs(%[[S21]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:           %[[S23:.*]] = tensor.empty() : tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE_9:.*]] = tensor.insert_slice %[[S22]] into %[[S23]][0, 0, 0, 0, 0, 0] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6xf32> into tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE_10:.*]] = tensor.insert_slice %[[INSERTED_SLICE_9]] into %[[ARG10]][0, 0, 0, 0, %[[ARG9]], %[[ARG7]]] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x1x1xf32> into tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:           scf.yield %[[INSERTED_SLICE_10]] : tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S18]] : tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S15]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0, 0, 0] [1, 1, 6, 6, 5, 2] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x5x2xf32> into tensor<2x2x6x6x5x2xf32>
// CHECK-NEXT:       scf.yield %[[INSERTED_SLICE]] : tensor<2x2x6x6x5x2xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S12]] : tensor<2x2x6x6x5x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[S5:.*]] = tensor.empty() : tensor<2x2x6x6x2x5xf32>
// CHECK-NEXT:   %[[S6:.*]] = tensor.empty() : tensor<2x2x6x6x2x5xf32>
// CHECK-NEXT:   %[[S7:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S6]]) -> (tensor<2x2x6x6x2x5xf32>) {
// CHECK-NEXT:     %[[S12:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<2x2x6x6x2x5xf32>) {
// CHECK-NEXT:   %[[S13:.*]] = affine.apply #[[$MAP2]](%[[ARG3]])
// CHECK-NEXT:   %[[S14:.*]] = affine.apply #[[$MAP2]](%[[ARG5]])
// CHECK-NEXT:       %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, %[[S13]], %[[S14]], 0] [2, 6, 6, 5] [1, 1, 1, 1] : tensor<2x10x10x5xf32> to tensor<2x6x6x5xf32>
// CHECK-NEXT:       %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[S5]][%[[ARG3]], %[[ARG5]], 0, 0, 0, 0] [1, 1, 6, 6, 2, 5] [1, 1, 1, 1, 1, 1] : tensor<2x2x6x6x2x5xf32> to tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:       %[[S15:.*]] = scf.for %[[ARG7:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.*]] = %[[EXTRACTED_SLICE_7]]) -> (tensor<1x1x6x6x2x5xf32>) {
// CHECK-NEXT:         %[[S18:.*]] = scf.for %[[ARG9:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG10:.*]] = %[[ARG8]]) -> (tensor<1x1x6x6x2x5xf32>) {
// CHECK-NEXT:           %[[EXTRACTED_SLICE_8:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG7]], 0, 0, %[[ARG9]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<2x6x6x5xf32> to tensor<1x6x6x1xf32>
// CHECK-NEXT:           %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE_8]][0, 0, 0, 0] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<1x6x6x1xf32> to tensor<6x6xf32>
// CHECK-NEXT:           %[[S19:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:           %[[S20:.*]] = linalg.matmul ins(%[[CST_3]], %[[EXTRACTED_SLICE]]_9 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S19]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:           %[[S21:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:           %[[S22:.*]] = linalg.matmul ins(%[[S20]], %[[CST_2]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S21]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:           %[[S23:.*]] = tensor.empty() : tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE_10:.*]] = tensor.insert_slice %[[S22]] into %[[S23]][0, 0, 0, 0, 0, 0] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6xf32> into tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE_11:.*]] = tensor.insert_slice %[[INSERTED_SLICE_10]] into %[[ARG10]][0, 0, 0, 0, %[[ARG7]], %[[ARG9]]] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x1x1xf32> into tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:           scf.yield %[[INSERTED_SLICE_11]] : tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S18]] : tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S15]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0, 0, 0] [1, 1, 6, 6, 2, 5] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x2x5xf32> into tensor<2x2x6x6x2x5xf32>
// CHECK-NEXT:       scf.yield %[[INSERTED_SLICE]] : tensor<2x2x6x6x2x5xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S12]] : tensor<2x2x6x6x2x5xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape %4 {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<2x2x6x6x5x2xf32> into tensor<144x5x2xf32>
// CHECK-NEXT:   %[[COLLAPSED_6:.*]] = tensor.collapse_shape %[[S7]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<2x2x6x6x2x5xf32> into tensor<144x2x5xf32>
// CHECK-NEXT:   %[[S8:.*]] = tensor.empty() : tensor<144x2x2xf32>
// CHECK-NEXT:   %[[S9:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_6]], %[[COLLAPSED]] : tensor<144x2x5xf32>, tensor<144x5x2xf32>) outs(%[[S8]] : tensor<144x2x2xf32>) -> tensor<144x2x2xf32>
// CHECK-NEXT:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S9]] {{\[}}[0, 1, 2, 3], [4], [5]] output_shape [2, 2, 6, 6, 2, 2] : tensor<144x2x2xf32> into tensor<2x2x6x6x2x2xf32>
// CHECK-NEXT:   %[[S10:.*]] = tensor.empty() : tensor<2x8x8x2xf32>
// CHECK-NEXT:   %[[S11:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S10]]) -> (tensor<2x8x8x2xf32>) {
// CHECK-NEXT:     %[[S12:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<2x8x8x2xf32>) {
// CHECK-NEXT:       %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[EXPANDED]][%[[ARG3]], %[[ARG5]], 0, 0, 0, 0] [1, 1, 6, 6, 2, 2] [1, 1, 1, 1, 1, 1] : tensor<2x2x6x6x2x2xf32> to tensor<1x1x6x6x2x2xf32>
// CHECK-NEXT:       %[[S13:.*]] = affine.apply #[[$MAP2]](%[[ARG3]])
// CHECK-NEXT:       %[[S14:.*]] = affine.apply #[[$MAP2]](%[[ARG5]])
// CHECK-NEXT:       %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[S1]][0, %[[S13]], %[[S14]], 0] [2, 4, 4, 2] [1, 1, 1, 1] : tensor<2x8x8x2xf32> to tensor<2x4x4x2xf32>
// CHECK-NEXT:       %[[S15:.*]] = scf.for %[[ARG7:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.*]] = %[[EXTRACTED_SLICE_7]]) -> (tensor<2x4x4x2xf32>) {
// CHECK-NEXT:         %[[S16:.*]] = scf.for %[[ARG9:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG10:.*]] = %[[ARG8]]) -> (tensor<2x4x4x2xf32>) {
// CHECK-NEXT:           %[[EXTRACTED_SLICE_8:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, 0, 0, %[[ARG7]], %[[ARG9]]] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x2x2xf32> to tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:           %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE_8]][0, 0, 0, 0, 0, 0] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x1x1xf32> to tensor<6x6xf32>
// CHECK-NEXT:           %[[S17:.*]] = tensor.empty() : tensor<4x6xf32>
// CHECK-NEXT:           %[[S18:.*]] = linalg.matmul ins(%[[CST_1]], %[[EXTRACTED_SLICE_9]] : tensor<4x6xf32>, tensor<6x6xf32>) outs(%[[S17]] : tensor<4x6xf32>) -> tensor<4x6xf32>
// CHECK-NEXT:           %[[S19:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:           %[[S20:.*]] = linalg.matmul ins(%[[S18]], %[[CST_0]] : tensor<4x6xf32>, tensor<6x4xf32>) outs(%[[S19]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:           %[[S21:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:           %[[S22:.*]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP4]], #[[$MAP4]]], iterator_types = ["parallel", "parallel"]} ins(%[[CST]], %[[S20]] : f32, tensor<4x4xf32>) outs(%[[S21]] : tensor<4x4xf32>) {
// CHECK-NEXT:           ^bb0(%[[IN:.*]]: f32, %[[IN_12:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:             %[[S24:.*]] = arith.mulf %[[IN]], %[[IN_12]] : f32
// CHECK-NEXT:             linalg.yield %[[S24]] : f32
// CHECK-NEXT:           } -> tensor<4x4xf32>
// CHECK-NEXT:           %[[S23:.*]] = tensor.empty() : tensor<1x4x4x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE_10:.*]] = tensor.insert_slice %[[S22]] into %[[S23]][0, 0, 0, 0] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<4x4xf32> into tensor<1x4x4x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE_11:.*]] = tensor.insert_slice %[[INSERTED_SLICE_10]] into %[[ARG10]][%[[ARG7]], 0, 0, %[[ARG9]]] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<1x4x4x1xf32> into tensor<2x4x4x2xf32>
// CHECK-NEXT:           scf.yield %[[INSERTED_SLICE_11]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S16]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %[[S16:.*]] = affine.apply #[[$MAP2]](%[[ARG3]])
// CHECK-NEXT:       %[[S17:.*]] = affine.apply #[[$MAP2]](%[[ARG5]])
// CHECK-NEXT:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S15]] into %[[ARG6]][0, %[[S16]], %[[S17]], 0] [2, 4, 4, 2] [1, 1, 1, 1] : tensor<2x4x4x2xf32> into tensor<2x8x8x2xf32>
// CHECK-NEXT:       scf.yield %[[INSERTED_SLICE]] : tensor<2x8x8x2xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S12]] : tensor<2x8x8x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[S11]] : tensor<2x8x8x2xf32>
// CHECK-NEXT: }

// -----

#map = affine_map<(d0, d1, d2, d3) -> (0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv2d_unaligned(%arg0: tensor<2x11x11x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x9x9x2xf32> {
  %0 = tensor.empty() : tensor<2x9x9x2xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1xf32>) outs(%0 : tensor<2x9x9x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x9x9x2xf32>
  %2 = tensor.empty() : tensor<3x3x6x6x5x2xf32>
  %3 = linalg.winograd_filter_transform m(4) r(3) ins(%arg1 : tensor<2x3x3x5xf32>) outs(%2 : tensor<3x3x6x6x5x2xf32>) -> tensor<3x3x6x6x5x2xf32>
  %4 = tensor.empty() : tensor<2x14x14x5xf32>
  %inserted_slice = tensor.insert_slice %arg0 into %4[0, 0, 0, 0] [2, 11, 11, 5] [1, 1, 1, 1] : tensor<2x11x11x5xf32> into tensor<2x14x14x5xf32>
  %5 = tensor.empty() : tensor<3x3x6x6x2x5xf32>
  %6 = linalg.winograd_input_transform m(4) r(3) ins(%inserted_slice : tensor<2x14x14x5xf32>) outs(%5 : tensor<3x3x6x6x2x5xf32>) -> tensor<3x3x6x6x2x5xf32>
  %collapsed = tensor.collapse_shape %3 [[0, 1, 2, 3], [4], [5]] : tensor<3x3x6x6x5x2xf32> into tensor<324x5x2xf32>
  %collapsed_0 = tensor.collapse_shape %6 [[0, 1, 2, 3], [4], [5]] : tensor<3x3x6x6x2x5xf32> into tensor<324x2x5xf32>
  %7 = tensor.empty() : tensor<324x2x2xf32>
  %8 = linalg.batch_matmul ins(%collapsed_0, %collapsed : tensor<324x2x5xf32>, tensor<324x5x2xf32>) outs(%7 : tensor<324x2x2xf32>) -> tensor<324x2x2xf32>
  %expanded = tensor.expand_shape %8 [[0, 1, 2, 3], [4], [5]] output_shape [3, 3, 6, 6, 2, 2] : tensor<324x2x2xf32> into tensor<3x3x6x6x2x2xf32>
  %9 = tensor.empty() : tensor<2x12x12x2xf32>
  %inserted_slice_1 = tensor.insert_slice %1 into %9[0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1] : tensor<2x9x9x2xf32> into tensor<2x12x12x2xf32>
  %10 = linalg.winograd_output_transform m(4) r(3) ins(%expanded : tensor<3x3x6x6x2x2xf32>) outs(%inserted_slice_1 : tensor<2x12x12x2xf32>) -> tensor<2x12x12x2xf32>
  %extracted_slice = tensor.extract_slice %10[0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1] : tensor<2x12x12x2xf32> to tensor<2x9x9x2xf32>
  return %extracted_slice : tensor<2x9x9x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:2 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %3, %loop3:2 = transform.structured.tile_using_for %2 tile_sizes [1, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %4 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %5, %loop5:2 = transform.structured.tile_using_for %4 tile_sizes [1, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %6 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %1 : (!transform.any_op) -> !transform.any_op
    %7 = transform.structured.decompose_winograd_op %6 : (!transform.any_op) -> (!transform.any_op)
    %8 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %3 : (!transform.any_op) -> !transform.any_op
    %9 = transform.structured.decompose_winograd_op %8 : (!transform.any_op) -> (!transform.any_op)
    %10 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %5 : (!transform.any_op) -> !transform.any_op
    %11 = transform.structured.decompose_winograd_op %10 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @conv2d_unaligned
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x11x11x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<1xf32>) -> tensor<2x9x9x2xf32> {
// CHECK-DAG:    %[[CST:.*]] = arith.constant 1.024000e+03 : f32
// CHECK-DAG:    %[[CST_0:.*]] = arith.constant dense<{{\[}}[1.250000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00], [2.500000e-01, -2.500000e-01, 2.500000e-01, -2.500000e-01], [2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01], [1.250000e-01, -2.500000e-01, 5.000000e-01, -1.000000e+00], [1.250000e-01, 2.500000e-01, 5.000000e-01, 1.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 5.000000e-01]]> : tensor<6x4xf32>
// CHECK-DAG:    %[[CST_1:.*]] = arith.constant dense<{{\[}}[1.250000e-01, 2.500000e-01, 2.500000e-01, 1.250000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -2.500000e-01, 2.500000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 2.500000e-01, 5.000000e-01, 5.000000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -1.000000e+00, 1.000000e+00, 5.000000e-01]]> : tensor<4x6xf32>
// CHECK-DAG:    %[[CST_2:.*]] = arith.constant dense<{{\[}}[2.500000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 2.500000e-01, -2.500000e-01, 2.500000e-01, -2.500000e-01, 2.500000e-01], [-3.125000e-01, -2.500000e-01, -2.500000e-01, -1.250000e-01, -1.250000e-01, 0.000000e+00], [0.000000e+00, -6.250000e-02, 6.250000e-02, -2.500000e-01, 2.500000e-01, -3.125000e-01], [6.250000e-02, 6.250000e-02, 6.250000e-02, 1.250000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 6.250000e-02]]> : tensor<6x6xf32>
// CHECK-DAG:    %[[CST_3:.*]] = arith.constant dense<{{\[}}[2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -2.500000e-01, -6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, -2.500000e-01, -2.500000e-01, 6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -1.250000e-01, -2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, -1.250000e-01, 2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02]]> : tensor<6x6xf32>
// CHECK-DAG:    %[[CST_4:.*]] = arith.constant dense<{{\[}}[1.000000e+00, -0.333333343, -0.333333343, 0.0833333358, 0.0833333358, 0.000000e+00], [0.000000e+00, 0.333333343, -0.333333343, -0.166666672, 0.166666672, 0.000000e+00], [0.000000e+00, -0.333333343, -0.333333343, 0.333333343, 0.333333343, 1.000000e+00]]> : tensor<3x6xf32>
// CHECK-DAG:    %[[CST_5:.*]] = arith.constant dense<{{\[}}[1.000000e+00, 0.000000e+00, 0.000000e+00], [-0.333333343, 0.333333343, -0.333333343], [-0.333333343, -0.333333343, -0.333333343], [0.0833333358, -0.166666672, 0.333333343], [0.0833333358, 0.166666672, 0.333333343], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<6x3xf32>
// CHECK-DAG:    %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[S0:.*]] = tensor.empty() : tensor<2x9x9x2xf32>
// CHECK-NEXT:   %[[S1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG2]] : tensor<1xf32>) outs(%[[S0]] : tensor<2x9x9x2xf32>) {
// CHECK-NEXT:   ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:     linalg.yield %[[IN]] : f32
// CHECK-NEXT:   } -> tensor<2x9x9x2xf32>
// CHECK-NEXT:   %[[S2:.*]] = tensor.empty() : tensor<3x3x6x6x5x2xf32>
// CHECK-NEXT:   %[[S3:.*]] = tensor.empty() : tensor<3x3x6x6x5x2xf32>
// CHECK-NEXT:   %[[S4:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S3]]) -> (tensor<3x3x6x6x5x2xf32>) {
// CHECK-NEXT:     %[[S12:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<3x3x6x6x5x2xf32>) {
// CHECK-NEXT:       %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[S2]][%[[ARG3]], %[[ARG5]], 0, 0, 0, 0] [1, 1, 6, 6, 5, 2] [1, 1, 1, 1, 1, 1] : tensor<3x3x6x6x5x2xf32> to tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:       %[[S15:.*]] = scf.for %[[ARG7:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.*]] = %[[EXTRACTED_SLICE]]) -> (tensor<1x1x6x6x5x2xf32>) {
// CHECK-NEXT:         %[[S18:.*]] = scf.for %[[ARG9:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG10:.*]] = %[[ARG8]]) -> (tensor<1x1x6x6x5x2xf32>) {
// CHECK-NEXT:           %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG7]], 0, 0, %[[ARG9]]] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<2x3x3x5xf32> to tensor<1x3x3x1xf32>
// CHECK-NEXT:           %[[EXTRACTED_SLICE_8:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE_7]][0, 0, 0, 0] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<1x3x3x1xf32> to tensor<3x3xf32>
// CHECK-NEXT:           %[[S19:.*]] = tensor.empty() : tensor<6x3xf32>
// CHECK-NEXT:           %[[S20:.*]] = linalg.matmul ins(%[[CST_5]], %[[EXTRACTED_SLICE_8]] : tensor<6x3xf32>, tensor<3x3xf32>) outs(%[[S19]] : tensor<6x3xf32>) -> tensor<6x3xf32>
// CHECK-NEXT:           %[[S21:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:           %[[S22:.*]] = linalg.matmul ins(%[[S20]], %[[CST_4]] : tensor<6x3xf32>, tensor<3x6xf32>) outs(%[[S21]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:           %[[S23:.*]] = tensor.empty() : tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE_9:.*]] = tensor.insert_slice %[[S22]] into %[[S23]][0, 0, 0, 0, 0, 0] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6xf32> into tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE_10:.*]] = tensor.insert_slice %[[INSERTED_SLICE_9]] into %[[ARG10]][0, 0, 0, 0, %[[ARG9]], %[[ARG7]]] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x1x1xf32> into tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:           scf.yield %[[INSERTED_SLICE_10]] : tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S18]] : tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S15]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0, 0, 0] [1, 1, 6, 6, 5, 2] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x5x2xf32> into tensor<3x3x6x6x5x2xf32>
// CHECK-NEXT:       scf.yield %[[INSERTED_SLICE]] : tensor<3x3x6x6x5x2xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S12]] : tensor<3x3x6x6x5x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[INPUT_BUF:.*]] = tensor.empty() : tensor<2x14x14x5xf32>
// CHECK-NEXT:   %[[INSERTED_INPUT_BUF:.*]] = tensor.insert_slice %[[ARG0]] into %[[INPUT_BUF]][0, 0, 0, 0] [2, 11, 11, 5] [1, 1, 1, 1] : tensor<2x11x11x5xf32> into tensor<2x14x14x5xf32>
// CHECK-NEXT:   %[[S5:.*]] = tensor.empty() : tensor<3x3x6x6x2x5xf32>
// CHECK-NEXT:   %[[S6:.*]] = tensor.empty() : tensor<3x3x6x6x2x5xf32>
// CHECK-NEXT:   %[[S7:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S6]]) -> (tensor<3x3x6x6x2x5xf32>) {
// CHECK-NEXT:     %[[S12:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<3x3x6x6x2x5xf32>) {
// CHECK-NEXT:   %[[S13:.*]] = affine.apply #[[$MAP2]](%[[ARG3]])
// CHECK-NEXT:   %[[S14:.*]] = affine.apply #[[$MAP2]](%[[ARG5]])
// CHECK-NEXT:       %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[INSERTED_INPUT_BUF]][0, %[[S13]], %[[S14]], 0] [2, 6, 6, 5] [1, 1, 1, 1] : tensor<2x14x14x5xf32> to tensor<2x6x6x5xf32>
// CHECK-NEXT:       %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[S5]][%[[ARG3]], %[[ARG5]], 0, 0, 0, 0] [1, 1, 6, 6, 2, 5] [1, 1, 1, 1, 1, 1] : tensor<3x3x6x6x2x5xf32> to tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:       %[[S15:.*]] = scf.for %[[ARG7:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.*]] = %[[EXTRACTED_SLICE_7]]) -> (tensor<1x1x6x6x2x5xf32>) {
// CHECK-NEXT:         %[[S18:.*]] = scf.for %[[ARG9:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG10:.*]] = %[[ARG8]]) -> (tensor<1x1x6x6x2x5xf32>) {
// CHECK-NEXT:           %[[EXTRACTED_SLICE_8:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG7]], 0, 0, %[[ARG9]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<2x6x6x5xf32> to tensor<1x6x6x1xf32>
// CHECK-NEXT:           %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE_8]][0, 0, 0, 0] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<1x6x6x1xf32> to tensor<6x6xf32>
// CHECK-NEXT:           %[[S19:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:           %[[S20:.*]] = linalg.matmul ins(%[[CST_3]], %[[EXTRACTED_SLICE_9]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S19]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:           %[[S21:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:           %[[S22:.*]] = linalg.matmul ins(%[[S20]], %[[CST_2]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S21]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:           %[[S23:.*]] = tensor.empty() : tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE_10:.*]] = tensor.insert_slice %[[S22]] into %[[S23]][0, 0, 0, 0, 0, 0] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6xf32> into tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE_11:.*]] = tensor.insert_slice %[[INSERTED_SLICE_10]] into %[[ARG10]][0, 0, 0, 0, %[[ARG7]], %[[ARG9]]] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x1x1xf32> into tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:           scf.yield %[[INSERTED_SLICE_11]] : tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S18]] : tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S15]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0, 0, 0] [1, 1, 6, 6, 2, 5] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x2x5xf32> into tensor<3x3x6x6x2x5xf32>
// CHECK-NEXT:       scf.yield %[[INSERTED_SLICE]] : tensor<3x3x6x6x2x5xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S12]] : tensor<3x3x6x6x2x5xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape %4 {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<3x3x6x6x5x2xf32> into tensor<324x5x2xf32>
// CHECK-NEXT:   %[[COLLAPSED_6:.*]] = tensor.collapse_shape %[[S7]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<3x3x6x6x2x5xf32> into tensor<324x2x5xf32>
// CHECK-NEXT:   %[[S8:.*]] = tensor.empty() : tensor<324x2x2xf32>
// CHECK-NEXT:   %[[S9:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_6]], %[[COLLAPSED]] : tensor<324x2x5xf32>, tensor<324x5x2xf32>) outs(%[[S8]] : tensor<324x2x2xf32>) -> tensor<324x2x2xf32>
// CHECK-NEXT:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S9]] {{\[}}[0, 1, 2, 3], [4], [5]] output_shape [3, 3, 6, 6, 2, 2] : tensor<324x2x2xf32> into tensor<3x3x6x6x2x2xf32>
// CHECK-NEXT:   %[[OUTPUT_BUF:.*]] = tensor.empty() : tensor<2x12x12x2xf32>
// CHECK-NEXT:  %[[INSERTED_OUTPUT_BUF:.*]] = tensor.insert_slice %[[S1]] into %[[OUTPUT_BUF]][0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1] : tensor<2x9x9x2xf32> into tensor<2x12x12x2xf32>
// CHECK-NEXT:   %[[S10:.*]] = tensor.empty() : tensor<2x12x12x2xf32>
// CHECK-NEXT:   %[[S11:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S10]]) -> (tensor<2x12x12x2xf32>) {
// CHECK-NEXT:     %[[S12:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<2x12x12x2xf32>) {
// CHECK-NEXT:       %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[EXPANDED]][%[[ARG3]], %[[ARG5]], 0, 0, 0, 0] [1, 1, 6, 6, 2, 2] [1, 1, 1, 1, 1, 1] : tensor<3x3x6x6x2x2xf32> to tensor<1x1x6x6x2x2xf32>
// CHECK-NEXT:       %[[S13:.*]] = affine.apply #[[$MAP2]](%[[ARG3]])
// CHECK-NEXT:       %[[S14:.*]] = affine.apply #[[$MAP2]](%[[ARG5]])
// CHECK-NEXT:       %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[INSERTED_OUTPUT_BUF]][0, %[[S13]], %[[S14]], 0] [2, 4, 4, 2] [1, 1, 1, 1] : tensor<2x12x12x2xf32> to tensor<2x4x4x2xf32>
// CHECK-NEXT:       %[[S15:.*]] = scf.for %[[ARG7:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.*]] = %[[EXTRACTED_SLICE_7]]) -> (tensor<2x4x4x2xf32>) {
// CHECK-NEXT:         %[[S16:.*]] = scf.for %[[ARG9:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG10:.*]] = %[[ARG8]]) -> (tensor<2x4x4x2xf32>) {
// CHECK-NEXT:           %[[EXTRACTED_SLICE_8:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, 0, 0, %[[ARG7]], %[[ARG9]]] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x2x2xf32> to tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:           %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE_8]][0, 0, 0, 0, 0, 0] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x1x1xf32> to tensor<6x6xf32>
// CHECK-NEXT:           %[[S17:.*]] = tensor.empty() : tensor<4x6xf32>
// CHECK-NEXT:           %[[S18:.*]] = linalg.matmul ins(%[[CST_1]], %[[EXTRACTED_SLICE_9]] : tensor<4x6xf32>, tensor<6x6xf32>) outs(%[[S17]] : tensor<4x6xf32>) -> tensor<4x6xf32>
// CHECK-NEXT:           %[[S19:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:           %[[S20:.*]] = linalg.matmul ins(%[[S18]], %[[CST_0]] : tensor<4x6xf32>, tensor<6x4xf32>) outs(%[[S19]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:           %[[S21:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:           %[[S22:.*]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP4]], #[[$MAP4]]], iterator_types = ["parallel", "parallel"]} ins(%[[CST]], %[[S20]] : f32, tensor<4x4xf32>) outs(%[[S21]] : tensor<4x4xf32>) {
// CHECK-NEXT:           ^bb0(%[[IN:.*]]: f32, %[[IN_12:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:             %[[S24:.*]] = arith.mulf %[[IN]], %[[IN_12]] : f32
// CHECK-NEXT:             linalg.yield %[[S24]] : f32
// CHECK-NEXT:           } -> tensor<4x4xf32>
// CHECK-NEXT:           %[[S23:.*]] = tensor.empty() : tensor<1x4x4x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE_10:.*]] = tensor.insert_slice %[[S22]] into %[[S23]][0, 0, 0, 0] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<4x4xf32> into tensor<1x4x4x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE_11:.*]] = tensor.insert_slice %[[INSERTED_SLICE_10]] into %[[ARG10]][%[[ARG7]], 0, 0, %[[ARG9]]] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<1x4x4x1xf32> into tensor<2x4x4x2xf32>
// CHECK-NEXT:           scf.yield %[[INSERTED_SLICE_11]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S16]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %[[S16:.*]] = affine.apply #[[$MAP2]](%[[ARG3]])
// CHECK-NEXT:       %[[S17:.*]] = affine.apply #[[$MAP2]](%[[ARG5]])
// CHECK-NEXT:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S15]] into %[[ARG6]][0, %[[S16]], %[[S17]], 0] [2, 4, 4, 2] [1, 1, 1, 1] : tensor<2x4x4x2xf32> into tensor<2x12x12x2xf32>
// CHECK-NEXT:       scf.yield %[[INSERTED_SLICE]] : tensor<2x12x12x2xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S12]] : tensor<2x12x12x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[RET:.*]] = tensor.extract_slice %[[S11]][0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1] : tensor<2x12x12x2xf32> to tensor<2x9x9x2xf32
// CHECK-NEXT:   return %[[RET]] : tensor<2x9x9x2xf32>
// CHECK-NEXT: }

// -----

#map = affine_map<(d0, d1, d2, d3) -> (0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv2d_mx1_rx1(%arg0: tensor<2x6x1x5xf32>, %arg1: tensor<2x3x1x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x4x1x2xf32> {
  %0 = tensor.empty() : tensor<2x4x1x2xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1xf32>) outs(%0 : tensor<2x4x1x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x4x1x2xf32>
  %2 = tensor.empty() : tensor<1x1x6x1x5x2xf32>
  %3 = linalg.winograd_filter_transform m(4) r(3) ins(%arg1 : tensor<2x3x1x5xf32>) outs(%2 : tensor<1x1x6x1x5x2xf32>) -> tensor<1x1x6x1x5x2xf32>
  %4 = tensor.empty() : tensor<1x1x6x1x2x5xf32>
  %5 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x6x1x5xf32>) outs(%4 : tensor<1x1x6x1x2x5xf32>) -> tensor<1x1x6x1x2x5xf32>
  %collapsed = tensor.collapse_shape %3 [[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x1x5x2xf32> into tensor<6x5x2xf32>
  %collapsed_0 = tensor.collapse_shape %5 [[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x1x2x5xf32> into tensor<6x2x5xf32>
  %6 = tensor.empty() : tensor<6x2x2xf32>
  %7 = linalg.batch_matmul ins(%collapsed_0, %collapsed : tensor<6x2x5xf32>, tensor<6x5x2xf32>) outs(%6 : tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
  %expanded = tensor.expand_shape %7 [[0, 1, 2, 3], [4], [5]] output_shape [1, 1, 1, 6, 2, 2] : tensor<6x2x2xf32> into tensor<1x1x6x1x2x2xf32>
  %8 = linalg.winograd_output_transform m(4) r(3) ins(%expanded : tensor<1x1x6x1x2x2xf32>) outs(%1 : tensor<2x4x1x2xf32>) -> tensor<2x4x1x2xf32>
  return %8 : tensor<2x4x1x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:2 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %3, %loop3:2 = transform.structured.tile_using_for %2 tile_sizes [1, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %4 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %5, %loop5:2 = transform.structured.tile_using_for %4 tile_sizes [1, 1, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %6 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %1 : (!transform.any_op) -> !transform.any_op
    %7 = transform.structured.decompose_winograd_op %6 : (!transform.any_op) -> (!transform.any_op)
    %8 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %3 : (!transform.any_op) -> !transform.any_op
    %9 = transform.structured.decompose_winograd_op %8 : (!transform.any_op) -> (!transform.any_op)
    %10 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %5 : (!transform.any_op) -> !transform.any_op
    %11 = transform.structured.decompose_winograd_op %10 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @conv2d_mx1_rx1
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x6x1x5xf32>, %[[ARG1:.*]]: tensor<2x3x1x5xf32>, %[[ARG2:.*]]: tensor<1xf32>) -> tensor<2x4x1x2xf32> {
// CHECK-DAG:     %[[CST:.*]] = arith.constant 3.200000e+01 : f32
// CHECK-DAG:     %[[CST_0:.*]] = arith.constant dense<{{\[}}[1.250000e-01, 2.500000e-01, 2.500000e-01, 1.250000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -2.500000e-01, 2.500000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 2.500000e-01, 5.000000e-01, 5.000000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -1.000000e+00, 1.000000e+00, 5.000000e-01]]> : tensor<4x6xf32>
// CHECK-DAG:     %[[CST_1:.*]] = arith.constant dense<{{\[}}[2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -2.500000e-01, -6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, -2.500000e-01, -2.500000e-01, 6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -1.250000e-01, -2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, -1.250000e-01, 2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02]]> : tensor<6x6xf32>
// CHECK-DAG:     %[[CST_2:.*]] = arith.constant dense<{{\[}}[1.000000e+00, 0.000000e+00, 0.000000e+00], [-0.333333343, 0.333333343, -0.333333343], [-0.333333343, -0.333333343, -0.333333343], [0.0833333358, -0.166666672, 0.333333343], [0.0833333358, 0.166666672, 0.333333343], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<6x3xf32>
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[S0:.*]] = tensor.empty() : tensor<2x4x1x2xf32>
// CHECK-NEXT:    %[[S1:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG2]] : tensor<1xf32>) outs(%[[S0]] : tensor<2x4x1x2xf32>) {
// CHECK-NEXT:    ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:      linalg.yield %[[IN]] : f32
// CHECK-NEXT:    } -> tensor<2x4x1x2xf32>
// CHECK-NEXT:    %[[S2:.*]] = tensor.empty() : tensor<1x1x6x1x5x2xf32>
// CHECK-NEXT:    %[[S3:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S2]]) -> (tensor<1x1x6x1x5x2xf32>) {
// CHECK-NEXT:      %[[S9:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<1x1x6x1x5x2xf32>) {
// CHECK-NEXT:        %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], 0, 0, %[[ARG5]]] [1, 3, 1, 1] [1, 1, 1, 1] : tensor<2x3x1x5xf32> to tensor<1x3x1x1xf32>
// CHECK-NEXT:        %[[EXTRACTED_SLICE_4:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, 0, 0] [1, 3, 1, 1] [1, 1, 1, 1] : tensor<1x3x1x1xf32> to tensor<3x1xf32>
// CHECK-NEXT:        %[[S10:.*]] = tensor.empty() : tensor<6x1xf32>
// CHECK-NEXT:        %[[S11:.*]] = linalg.matmul ins(%[[CST_2]], %[[EXTRACTED_SLICE_4]] : tensor<6x3xf32>, tensor<3x1xf32>) outs(%[[S10]] : tensor<6x1xf32>) -> tensor<6x1xf32>
// CHECK-NEXT:        %[[S12:.*]] = tensor.empty() : tensor<1x1x6x1x1x1xf32>
// CHECK-NEXT:        %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S11]] into %[[S12]][0, 0, 0, 0, 0, 0] [1, 1, 6, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x1xf32> into tensor<1x1x6x1x1x1xf32>
// CHECK-NEXT:        %[[INSERTED_SLICE_5:.*]] = tensor.insert_slice %[[INSERTED_SLICE]] into %[[ARG6]][0, 0, 0, 0, %[[ARG5]], %[[ARG3]]] [1, 1, 6, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x1x1x1xf32> into tensor<1x1x6x1x5x2xf32>
// CHECK-NEXT:        scf.yield %[[INSERTED_SLICE_5]] : tensor<1x1x6x1x5x2xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[S9]] : tensor<1x1x6x1x5x2xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[S4:.*]] = tensor.empty() : tensor<1x1x6x1x2x5xf32>
// CHECK-NEXT:    %[[S5:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S4]]) -> (tensor<1x1x6x1x2x5xf32>) {
// CHECK-NEXT:      %[[S9:.*]] = scf.for %[[ARG5]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<1x1x6x1x2x5xf32>) {
// CHECK-NEXT:        %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], 0, 0, %[[ARG5]]] [1, 6, 1, 1] [1, 1, 1, 1] : tensor<2x6x1x5xf32> to tensor<1x6x1x1xf32>
// CHECK-NEXT:        %[[EXTRACTED_SLICE_4:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, 0, 0] [1, 6, 1, 1] [1, 1, 1, 1] : tensor<1x6x1x1xf32> to tensor<6x1xf32>
// CHECK-NEXT:        %[[S10:.*]] = tensor.empty() : tensor<6x1xf32>
// CHECK-NEXT:        %[[S11:.*]] = linalg.matmul ins(%[[CST_1]], %[[EXTRACTED_SLICE_4]] : tensor<6x6xf32>, tensor<6x1xf32>) outs(%[[S10]] : tensor<6x1xf32>) -> tensor<6x1xf32>
// CHECK-NEXT:        %[[S12:.*]] = tensor.empty() : tensor<1x1x6x1x1x1xf32>
// CHECK-NEXT:        %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S11]] into %[[S12]][0, 0, 0, 0, 0, 0] [1, 1, 6, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x1xf32> into tensor<1x1x6x1x1x1xf32>
// CHECK-NEXT:        %[[INSERTED_SLICE_5:.*]] = tensor.insert_slice %[[INSERTED_SLICE]] into %[[ARG6]][0, 0, 0, 0, %[[ARG3]], %[[ARG5]]] [1, 1, 6, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x1x1x1xf32> into tensor<1x1x6x1x2x5xf32>
// CHECK-NEXT:        scf.yield %[[INSERTED_SLICE_5]] : tensor<1x1x6x1x2x5xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[S9]] : tensor<1x1x6x1x2x5xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x1x5x2xf32> into tensor<6x5x2xf32>
// CHECK-NEXT:    %[[COLLAPSED_3:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x1x2x5xf32> into tensor<6x2x5xf32>
// CHECK-NEXT:    %[[S6:.*]] = tensor.empty() : tensor<6x2x2xf32>
// CHECK-NEXT:    %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_3]], %[[COLLAPSED]] : tensor<6x2x5xf32>, tensor<6x5x2xf32>) outs(%[[S6]] : tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
// CHECK-NEXT:    %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1, 2, 3], [4], [5]] output_shape [1, 1, 1, 6, 2, 2] : tensor<6x2x2xf32> into tensor<1x1x6x1x2x2xf32>
// CHECK-NEXT:    %[[S8:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S1]]) -> (tensor<2x4x1x2xf32>) {
// CHECK-NEXT:      %[[S9:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<2x4x1x2xf32>) {
// CHECK-NEXT:        %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[EXPANDED]][0, 0, 0, 0, %[[ARG3]], %[[ARG5]]] [1, 1, 6, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x1x2x2xf32> to tensor<1x1x6x1x1x1xf32>
// CHECK-NEXT:        %[[EXTRACTED_SLICE_4:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, 0, 0, 0, 0] [1, 1, 6, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x1x1x1xf32> to tensor<6x1xf32>
// CHECK-NEXT:        %[[S10:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK-NEXT:        %[[S11:.*]] = linalg.matmul ins(%[[CST_0]], %[[EXTRACTED_SLICE_4]] : tensor<4x6xf32>, tensor<6x1xf32>) outs(%[[S10]] : tensor<4x1xf32>) -> tensor<4x1xf32>
// CHECK-NEXT:        %[[S12:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK-NEXT:        %[[S13:.*]] = linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%[[CST]], %[[S11]] : f32, tensor<4x1xf32>) outs(%[[S12]] : tensor<4x1xf32>) {
// CHECK-NEXT:        ^bb0(%[[IN:.*]]: f32, %[[IN_6:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:          %[[S15:.*]] = arith.mulf %[[IN]], %[[IN_6]] : f32
// CHECK-NEXT:          linalg.yield %[[S15]] : f32
// CHECK-NEXT:        } -> tensor<4x1xf32>
// CHECK-NEXT:        %[[S14:.*]] = tensor.empty() : tensor<1x4x1x1xf32>
// CHECK-NEXT:        %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S13]] into %[[S14]][0, 0, 0, 0] [1, 4, 1, 1] [1, 1, 1, 1] : tensor<4x1xf32> into tensor<1x4x1x1xf32>
// CHECK-NEXT:        %[[INSERTED_SLICE_5:.*]] = tensor.insert_slice %[[INSERTED_SLICE]] into %[[ARG6]][%[[ARG3]], 0, 0, %[[ARG5]]] [1, 4, 1, 1] [1, 1, 1, 1] : tensor<1x4x1x1xf32> into tensor<2x4x1x2xf32>
// CHECK-NEXT:        scf.yield %[[INSERTED_SLICE_5]] : tensor<2x4x1x2xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[S9]] : tensor<2x4x1x2xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[S8]] : tensor<2x4x1x2xf32>
// CHECK-NEXT:  }
