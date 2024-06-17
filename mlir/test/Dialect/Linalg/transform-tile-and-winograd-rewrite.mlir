// RUN: mlir-opt %s -transform-interpreter -canonicalize --split-input-file | FileCheck %s

func.func @conv2d(%arg0: tensor<2x10x10x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32> {
  %0 = tensor.empty() : tensor<6x6x5x2xf32>
  %1 = linalg.winograd_filter_transform m(4) r(3) ins(%arg1 : tensor<2x3x3x5xf32>) outs(%0 : tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32>
  %2 = tensor.empty() : tensor<6x6x2x2x2x5xf32>
  %3 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x10x10x5xf32>) outs(%2 : tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<6x6x5x2xf32> into tensor<36x5x2xf32>
  %collapsed_0 = tensor.collapse_shape %3 [[0, 1], [2, 3, 4], [5]] : tensor<6x6x2x2x2x5xf32> into tensor<36x8x5xf32>
  %4 = tensor.empty() : tensor<36x8x2xf32>
  %5 = linalg.batch_matmul ins(%collapsed_0, %collapsed : tensor<36x8x5xf32>, tensor<36x5x2xf32>) outs(%4 : tensor<36x8x2xf32>) -> tensor<36x8x2xf32>
  %expanded = tensor.expand_shape %5 [[0, 1], [2, 3, 4], [5]] output_shape [6, 6, 2, 2, 2, 2] : tensor<36x8x2xf32> into tensor<6x6x2x2x2x2xf32>
  %6 = linalg.winograd_output_transform m(4) r(3) ins(%expanded : tensor<6x6x2x2x2x2xf32>) outs(%arg2 : tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %6 : tensor<2x8x8x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %3, %loop3:2 = transform.structured.tile_using_for %2 tile_sizes [0, 0, 1, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %4 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %5, %loop5:2 = transform.structured.tile_using_for %4 tile_sizes [0, 0, 1, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %7 = transform.structured.decompose_winograd_op %0 : (!transform.any_op) -> (!transform.any_op)
    %8 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %3 : (!transform.any_op) -> !transform.any_op
    %9 = transform.structured.decompose_winograd_op %8 : (!transform.any_op) -> (!transform.any_op)
    %10 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %5 : (!transform.any_op) -> !transform.any_op
    %11 = transform.structured.decompose_winograd_op %10 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK-LABEL: func.func @conv2d
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x10x10x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32> {
// CHECK-DAG:    %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[S0:.*]] = tensor.empty() : tensor<6x6x5x2xf32>
// CHECK:        scf.for {{.*}} = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG3:.*]] = %[[S0]]) -> (tensor<6x6x5x2xf32>) {
// CHECK-NEXT:     scf.for {{.*}} = %[[C0]] to %[[C5]] step %[[C1]] iter_args({{.*}} = %[[ARG3]]) -> (tensor<6x6x5x2xf32>) {
// CHECK:            linalg.matmul
// CHECK:            linalg.matmul
// CHECK:        %[[S5:.*]] = tensor.empty() : tensor<6x6x2x2x2x5xf32>
// CHECK:        %[[S6:.*]] = tensor.empty() : tensor<6x6x2x2x2x5xf32>
// CHECK:        scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S6]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:     scf.for %[[ARG5:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK:            %[[S13:.*]] = affine.apply #[[$MAP0]](%[[ARG3]])
// CHECK:            %[[S14:.*]] = affine.apply #[[$MAP0]](%[[ARG5]])
// CHECK:            tensor.extract_slice %[[ARG0]][0, %[[S13]], %[[S14]], 0] [2, 6, 6, 5] [1, 1, 1, 1] : tensor<2x10x10x5xf32> to tensor<2x6x6x5xf32>
// CHECK:            %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[S5]][0, 0, %[[ARG3]], %[[ARG5]], 0, 0] [6, 6, 1, 1, 2, 5] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x2x5xf32> to tensor<6x6x1x1x2x5xf32>
// CHECK:            %[[S15:.*]] = scf.for {{.*}} = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.*]] = %[[EXTRACTED_SLICE_7]]) -> (tensor<6x6x1x1x2x5xf32>) {
// CHECK-NEXt:         scf.for {{.*}} = %[[C0]] to %[[C5]] step %[[C1]] iter_args({{.*}} = %[[ARG8]]) -> (tensor<6x6x1x1x2x5xf32>) {
// CHECK:                linalg.matmul
// CHECK:                linalg.matmul
// CHECK:          %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S15]] into %[[ARG6]][0, 0, %[[ARG3]], %[[ARG5]], 0, 0] [6, 6, 1, 1, 2, 5] [1, 1, 1, 1, 1, 1] : tensor<6x6x1x1x2x5xf32> into tensor<6x6x2x2x2x5xf32>
// CHECK:       scf.yield %[[INSERTED_SLICE]] : tensor<6x6x2x2x2x5xf32>
// CHECK:       %[[S9:.*]] = linalg.batch_matmul
// CHECK:       %[[EXPANDED:.*]] = tensor.expand_shape %[[S9]] {{\[}}[0, 1], [2, 3, 4], [5]] output_shape [6, 6, 2, 2, 2, 2] : tensor<36x8x2xf32> into tensor<6x6x2x2x2x2xf32>
// CHECK:   %[[S10:.*]] = tensor.empty() : tensor<2x8x8x2xf32>
// CHECK:   scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S10]]) -> (tensor<2x8x8x2xf32>) {
// CHECK-NEXT:     scf.for %[[ARG5:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<2x8x8x2xf32>) {
// CHECK:       tensor.extract_slice %[[EXPANDED]][0, 0, %[[ARG3]], %[[ARG5]], 0, 0] [6, 6, 1, 1, 2, 2] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x2x2xf32> to tensor<6x6x1x1x2x2xf32>
// CHECK:       %[[S13:.*]] = affine.apply #[[$MAP0]](%[[ARG3]])
// CHECK:       %[[S14:.*]] = affine.apply #[[$MAP0]](%[[ARG5]])
// CHECK:       %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[ARG2]][0, %[[S13]], %[[S14]], 0] [2, 4, 4, 2] [1, 1, 1, 1] : tensor<2x8x8x2xf32> to tensor<2x4x4x2xf32>
// CHECK:       %[[S15:.*]] = scf.for %[[ARG7:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.*]] = %[[EXTRACTED_SLICE_7]]) -> (tensor<2x4x4x2xf32>) {
// CHECK-NEXT:         scf.for %[[ARG9:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG10:.*]] = %[[ARG8]]) -> (tensor<2x4x4x2xf32>) {
// CHECK:           linalg.matmul
// CHECK:           linalg.matmul
// CHECK:       %[[S16:.*]] = affine.apply #[[$MAP0]](%[[ARG3]])
// CHECK:       %[[S17:.*]] = affine.apply #[[$MAP0]](%[[ARG5]])
// CHECK:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S15]] into %[[ARG6]][0, %[[S16]], %[[S17]], 0] [2, 4, 4, 2] [1, 1, 1, 1] : tensor<2x4x4x2xf32> into tensor<2x8x8x2xf32>
// CHECK:       scf.yield %[[INSERTED_SLICE]] : tensor<2x8x8x2xf32>

// -----

func.func @conv2d_unaligned(%arg0: tensor<2x11x11x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>, %arg3: tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<6x6x5x2xf32>
  %1 = linalg.winograd_filter_transform m(4) r(3) ins(%arg1 : tensor<2x3x3x5xf32>) outs(%0 : tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32>
  %padded = tensor.pad %arg0 low[0, 0, 0, 0] high[0, 3, 3, 0] {
  ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
    tensor.yield %cst : f32
  } : tensor<2x11x11x5xf32> to tensor<2x14x14x5xf32>
  %2 = tensor.empty() : tensor<6x6x3x3x2x5xf32>
  %3 = linalg.winograd_input_transform m(4) r(3) ins(%padded : tensor<2x14x14x5xf32>) outs(%2 : tensor<6x6x3x3x2x5xf32>) -> tensor<6x6x3x3x2x5xf32>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<6x6x5x2xf32> into tensor<36x5x2xf32>
  %collapsed_0 = tensor.collapse_shape %3 [[0, 1], [2, 3, 4], [5]] : tensor<6x6x3x3x2x5xf32> into tensor<36x18x5xf32>
  %4 = tensor.empty() : tensor<36x18x2xf32>
  %5 = linalg.batch_matmul ins(%collapsed_0, %collapsed : tensor<36x18x5xf32>, tensor<36x5x2xf32>) outs(%4 : tensor<36x18x2xf32>) -> tensor<36x18x2xf32>
  %expanded = tensor.expand_shape %5 [[0, 1], [2, 3, 4], [5]] output_shape [6, 6, 3, 3, 2, 2] : tensor<36x18x2xf32> into tensor<6x6x3x3x2x2xf32>
  %padded_1 = tensor.pad %arg3 low[0, 0, 0, 0] high[0, 3, 3, 0] {
  ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
    tensor.yield %cst : f32
  } : tensor<2x9x9x2xf32> to tensor<2x12x12x2xf32>
  %6 = linalg.winograd_output_transform m(4) r(3) ins(%expanded : tensor<6x6x3x3x2x2xf32>) outs(%padded_1 : tensor<2x12x12x2xf32>) -> tensor<2x12x12x2xf32>
  %extracted_slice = tensor.extract_slice %6[0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1] : tensor<2x12x12x2xf32> to tensor<2x9x9x2xf32>
  return %extracted_slice : tensor<2x9x9x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %3, %loop3:2 = transform.structured.tile_using_for %2 tile_sizes [0, 0, 1, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %4 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %5, %loop5:2 = transform.structured.tile_using_for %4 tile_sizes [0, 0, 1, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %7 = transform.structured.decompose_winograd_op %0 : (!transform.any_op) -> (!transform.any_op)
    %8 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %3 : (!transform.any_op) -> !transform.any_op
    %9 = transform.structured.decompose_winograd_op %8 : (!transform.any_op) -> (!transform.any_op)
    %10 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %5 : (!transform.any_op) -> !transform.any_op
    %11 = transform.structured.decompose_winograd_op %10 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK-LABEL: func.func @conv2d_unaligned
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x11x11x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<1xf32>, %[[ARG3:.*]]: tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32> {
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[CST_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:   %[[S0:.*]] = tensor.empty() : tensor<6x6x5x2xf32>
// CHECK:   scf.for %[[ARG4:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG5:.*]] = %[[S0]]) -> (tensor<6x6x5x2xf32>) {
// CHECK:     scf.for %[[ARG6:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<6x6x5x2xf32>) {
// CHECK:       %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG4]], 0, 0, %[[ARG6]]] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<2x3x3x5xf32> to tensor<1x3x3x1xf32>
// CHECK:       tensor.extract_slice %[[EXTRACTED_SLICE_9]][0, 0, 0, 0] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<1x3x3x1xf32> to tensor<3x3xf32>
// CHECK:       linalg.matmul
// CHECK:       %[[S13:.*]] = linalg.matmul
// CHECK:       %[[S14:.*]] = tensor.empty() : tensor<6x6x1x1xf32>
// CHECK:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S13]] into %[[S14]][0, 0, 0, 0] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6xf32> into tensor<6x6x1x1xf32>
// CHECK:       %[[INSERTED_SLICE_11:.*]] = tensor.insert_slice %[[INSERTED_SLICE]] into %[[ARG7]][0, 0, %[[ARG6]], %[[ARG4]]] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6x1x1xf32> into tensor<6x6x5x2xf32>
// CHECK:       scf.yield %[[INSERTED_SLICE_11]] : tensor<6x6x5x2xf32>
// CHECK:   %[[PADDED:.*]] = tensor.pad %[[ARG0]] low[0, 0, 0, 0] high[0, 3, 3, 0] {
// CHECK:   ^bb0({{.*}}):
// CHECK:     tensor.yield %[[CST_6]] : f32
// CHECK:   } : tensor<2x11x11x5xf32> to tensor<2x14x14x5xf32>
// CHECK:   %[[S2:.*]] = tensor.empty() : tensor<6x6x3x3x2x5xf32>
// CHECK:   %[[S3:.*]] = tensor.empty() : tensor<6x6x3x3x2x5xf32>
// CHECK:   scf.for %[[ARG4:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG5:.*]] = %[[S3]]) -> (tensor<6x6x3x3x2x5xf32>) {
// CHECK:     scf.for %[[ARG6:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<6x6x3x3x2x5xf32>) {
// CHECK:       %[[S10:.*]] = affine.apply #[[$MAP0]](%[[ARG4]])
// CHECK:       %[[S11:.*]] = affine.apply #[[$MAP0]](%[[ARG6]])
// CHECK:       tensor.extract_slice %[[PADDED]][0, %[[S10]], %[[S11]], 0] [2, 6, 6, 5] [1, 1, 1, 1] : tensor<2x14x14x5xf32> to tensor<2x6x6x5xf32>
// CHECK:       %[[EXTRACTED_SLICE_10:.*]] = tensor.extract_slice %[[S2]][0, 0, %[[ARG4]], %[[ARG6]], 0, 0] [6, 6, 1, 1, 2, 5] [1, 1, 1, 1, 1, 1] : tensor<6x6x3x3x2x5xf32> to tensor<6x6x1x1x2x5xf32>
// CHECK:       %[[S12:.*]] = scf.for %[[ARG8:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG9:.*]] = %[[EXTRACTED_SLICE_10]]) -> (tensor<6x6x1x1x2x5xf32>) {
// CHECK:         scf.for %[[ARG10:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG11:.*]] = %[[ARG9]]) -> (tensor<6x6x1x1x2x5xf32>) {
// CHECK:           linalg.matmul
// CHECK:           %[[S17:.*]] = linalg.matmul
// CHECK:           %[[S18:.*]] = tensor.empty() : tensor<6x6x1x1x1x1xf32>
// CHECK:           %[[INSERTED_SLICE_13:.*]] = tensor.insert_slice %[[S17]] into %[[S18]][0, 0, 0, 0, 0, 0] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6xf32> into tensor<6x6x1x1x1x1xf32>
// CHECK:           %[[INSERTED_SLICE_14:.*]] = tensor.insert_slice %[[INSERTED_SLICE_13]] into %[[ARG11]][0, 0, 0, 0, %[[ARG8]], %[[ARG10]]] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6x1x1x1x1xf32> into tensor<6x6x1x1x2x5xf32>
// CHECK:           scf.yield %[[INSERTED_SLICE_14]] : tensor<6x6x1x1x2x5xf32>
// CHECK:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S12]] into %[[ARG7]][0, 0, %[[ARG4]], %[[ARG6]], 0, 0] [6, 6, 1, 1, 2, 5] [1, 1, 1, 1, 1, 1] : tensor<6x6x1x1x2x5xf32> into tensor<6x6x3x3x2x5xf32>
// CHECK:       scf.yield %[[INSERTED_SLICE]] : tensor<6x6x3x3x2x5xf32>
// CHECK:   %[[S6:.*]] = linalg.batch_matmul
// CHECK:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S6]] {{\[}}[0, 1], [2, 3, 4], [5]] output_shape [6, 6, 3, 3, 2, 2] : tensor<36x18x2xf32> into tensor<6x6x3x3x2x2xf32>
// CHECK:   %[[PADDED_8:.*]] = tensor.pad %[[ARG3]] low[0, 0, 0, 0] high[0, 3, 3, 0] {
// CHECK:   ^bb0({{.*}}):
// CHECK:     tensor.yield %[[CST_6]] : f32
// CHECK:   } : tensor<2x9x9x2xf32> to tensor<2x12x12x2xf32>
// CHECK:   %[[S7:.*]] = tensor.empty() : tensor<2x12x12x2xf32>
// CHECK:   %[[S8:.*]] = scf.for %[[ARG4:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG5:.*]] = %[[S7]]) -> (tensor<2x12x12x2xf32>) {
// CHECK:     scf.for %[[ARG6:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<2x12x12x2xf32>) {
// CHECK:       tensor.extract_slice %[[EXPANDED]][0, 0, %[[ARG4]], %[[ARG6]], 0, 0] [6, 6, 1, 1, 2, 2] [1, 1, 1, 1, 1, 1] : tensor<6x6x3x3x2x2xf32> to tensor<6x6x1x1x2x2xf32>
// CHECK:       %[[S10:.*]] = affine.apply #[[$MAP0]](%[[ARG4]])
// CHECK:       %[[S11:.*]] = affine.apply #[[$MAP0]](%[[ARG6]])
// CHECK:       %[[EXTRACTED_SLICE_10:.*]] = tensor.extract_slice %[[PADDED_8]][0, %[[S10]], %[[S11]], 0] [2, 4, 4, 2] [1, 1, 1, 1] : tensor<2x12x12x2xf32> to tensor<2x4x4x2xf32>
// CHECK:       %[[S12:.*]] = scf.for %[[ARG8:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG9:.*]] = %[[EXTRACTED_SLICE_10]]) -> (tensor<2x4x4x2xf32>) {
// CHECK:         scf.for %[[ARG10:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG11:.*]] = %[[ARG9]]) -> (tensor<2x4x4x2xf32>) {
// CHECK:           linalg.matmul
// CHECK:           linalg.matmul
// CHECK:           %[[S22:.*]] = linalg.mul
// CHECK:           %[[S23:.*]] = tensor.empty() : tensor<1x4x4x1xf32>
// CHECK:           %[[INSERTED_SLICE_13:.*]] = tensor.insert_slice %[[S22]] into %[[S23]][0, 0, 0, 0] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<4x4xf32> into tensor<1x4x4x1xf32>
// CHECK:           %[[INSERTED_SLICE_14:.*]] = tensor.insert_slice %[[INSERTED_SLICE_13]] into %[[ARG11]][%[[ARG8]], 0, 0, %[[ARG10]]] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<1x4x4x1xf32> into tensor<2x4x4x2xf32>
// CHECK:           scf.yield %[[INSERTED_SLICE_14]] : tensor<2x4x4x2xf32>
// CHECK:       %[[S13:.*]] = affine.apply #[[$MAP0]](%[[ARG4]])
// CHECK:       %[[S14:.*]] = affine.apply #[[$MAP0]](%[[ARG6]])
// CHECK:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S12]] into %[[ARG7]][0, %[[S13]], %[[S14]], 0] [2, 4, 4, 2] [1, 1, 1, 1] : tensor<2x4x4x2xf32> into tensor<2x12x12x2xf32>
// CHECK:       scf.yield %[[INSERTED_SLICE]] : tensor<2x12x12x2xf32>
// CHECK:   %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[S8]][0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1] : tensor<2x12x12x2xf32> to tensor<2x9x9x2xf32>
// CHECK:   return %[[EXTRACTED_SLICE]] : tensor<2x9x9x2xf32>
// CHECK: }

// -----

func.func @conv2d_mx1_rx1(%arg0: tensor<2x6x1x5xf32>, %arg1: tensor<2x3x1x5xf32>, %arg2: tensor<1xf32>, %arg3: tensor<2x4x1x2xf32>) -> tensor<2x4x1x2xf32> {
  %0 = tensor.empty() : tensor<6x1x5x2xf32>
  %1 = linalg.winograd_filter_transform m(4) r(3) ins(%arg1 : tensor<2x3x1x5xf32>) outs(%0 : tensor<6x1x5x2xf32>) -> tensor<6x1x5x2xf32>
  %2 = tensor.empty() : tensor<6x1x1x1x2x5xf32>
  %3 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x6x1x5xf32>) outs(%2 : tensor<6x1x1x1x2x5xf32>) -> tensor<6x1x1x1x2x5xf32>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<6x1x5x2xf32> into tensor<6x5x2xf32>
  %collapsed_0 = tensor.collapse_shape %3 [[0, 1], [2, 3, 4], [5]] : tensor<6x1x1x1x2x5xf32> into tensor<6x2x5xf32>
  %4 = tensor.empty() : tensor<6x2x2xf32>
  %5 = linalg.batch_matmul ins(%collapsed_0, %collapsed : tensor<6x2x5xf32>, tensor<6x5x2xf32>) outs(%4 : tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
  %expanded = tensor.expand_shape %5 [[0, 1], [2, 3, 4], [5]] output_shape [6, 1, 1, 1, 2, 2] : tensor<6x2x2xf32> into tensor<6x1x1x1x2x2xf32>
  %6 = linalg.winograd_output_transform m(4) r(3) ins(%expanded : tensor<6x1x1x1x2x2xf32>) outs(%arg3 : tensor<2x4x1x2xf32>) -> tensor<2x4x1x2xf32>
  return %6 : tensor<2x4x1x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %3, %loop3:2 = transform.structured.tile_using_for %2 tile_sizes [0, 0, 1, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %4 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %5, %loop5:2 = transform.structured.tile_using_for %4 tile_sizes [0, 0, 1, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %7 = transform.structured.decompose_winograd_op %0 : (!transform.any_op) -> (!transform.any_op)
    %8 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %3 : (!transform.any_op) -> !transform.any_op
    %9 = transform.structured.decompose_winograd_op %8 : (!transform.any_op) -> (!transform.any_op)
    %10 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %5 : (!transform.any_op) -> !transform.any_op
    %11 = transform.structured.decompose_winograd_op %10 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @conv2d_mx1_rx1
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x6x1x5xf32>, %[[ARG1:.*]]: tensor<2x3x1x5xf32>, %[[ARG2:.*]]: tensor<1xf32>, %[[ARG3:.*]]: tensor<2x4x1x2xf32>) -> tensor<2x4x1x2xf32> {
// CHECK:  %[[C1:.*]] = arith.constant 1 : index
// CHECK:  %[[C5:.*]] = arith.constant 5 : index
// CHECK:  %[[C2:.*]] = arith.constant 2 : index
// CHECK:  %[[C0:.*]] = arith.constant 0 : index
// CHECK:  %[[S0:.*]] = tensor.empty() : tensor<6x1x5x2xf32>
// CHECK:  scf.for %[[ARG4:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG5:.*]] = %[[S0]]) -> (tensor<6x1x5x2xf32>) {
// CHECK:    scf.for %[[ARG6:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<6x1x5x2xf32>) {
// CHECK:      %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG4]], 0, 0, %[[ARG6]]] [1, 3, 1, 1] [1, 1, 1, 1] : tensor<2x3x1x5xf32> to tensor<1x3x1x1xf32>
// CHECK:      tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, 0, 0] [1, 3, 1, 1] [1, 1, 1, 1] : tensor<1x3x1x1xf32> to tensor<3x1xf32>
// CHECK:      %[[S9:.*]] = linalg.matmul
// CHECK:      %[[S10:.*]] = tensor.empty() : tensor<6x1x1x1xf32>
// CHECK:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S9]] into %[[S10]][0, 0, 0, 0] [6, 1, 1, 1] [1, 1, 1, 1] : tensor<6x1xf32> into tensor<6x1x1x1xf32>
// CHECK:      %[[INSERTED_SLICE_5:.*]] = tensor.insert_slice %[[INSERTED_SLICE]] into %[[ARG7]][0, 0, %[[ARG6]], %[[ARG4]]] [6, 1, 1, 1] [1, 1, 1, 1] : tensor<6x1x1x1xf32> into tensor<6x1x5x2xf32>
// CHECK:      scf.yield %[[INSERTED_SLICE_5]] : tensor<6x1x5x2xf32>
// CHECK:  %[[S2:.*]] = tensor.empty() : tensor<6x1x1x1x2x5xf32>
// CHECK:  scf.for %[[ARG4:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG5:.*]] = %[[S2]]) -> (tensor<6x1x1x1x2x5xf32>) {
// CHECK:    scf.for %[[ARG6:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<6x1x1x1x2x5xf32>) {
// CHECK:      %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG4]], 0, 0, %[[ARG6]]] [1, 6, 1, 1] [1, 1, 1, 1] : tensor<2x6x1x5xf32> to tensor<1x6x1x1xf32>
// CHECK:      tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, 0, 0] [1, 6, 1, 1] [1, 1, 1, 1] : tensor<1x6x1x1xf32> to tensor<6x1xf32>
// CHECK:      %[[S9:.*]] = linalg.matmul
// CHECK:      %[[S10:.*]] = tensor.empty() : tensor<6x1x1x1x1x1xf32>
// CHECK:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S9]] into %[[S10]][0, 0, 0, 0, 0, 0] [6, 1, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x1xf32> into tensor<6x1x1x1x1x1xf32>
// CHECK:      %[[INSERTED_SLICE_5:.*]] = tensor.insert_slice %[[INSERTED_SLICE]] into %[[ARG7]][0, 0, 0, 0, %[[ARG4]], %[[ARG6]]] [6, 1, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x1x1x1x1x1xf32> into tensor<6x1x1x1x2x5xf32>
// CHECK:      scf.yield %[[INSERTED_SLICE_5]] : tensor<6x1x1x1x2x5xf32>
// CHECK:  %[[S5:.*]] = linalg.batch_matmul
// CHECK:  %[[EXPANDED:.*]] = tensor.expand_shape %[[S5]] {{\[}}[0, 1], [2, 3, 4], [5]] output_shape [6, 1, 1, 1, 2, 2] : tensor<6x2x2xf32> into tensor<6x1x1x1x2x2xf32>
// CHECK:  scf.for %[[ARG4:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG5:.*]] = %[[ARG3]]) -> (tensor<2x4x1x2xf32>) {
// CHECK:    scf.for %[[ARG6:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<2x4x1x2xf32>) {
// CHECK:      %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[EXPANDED]][0, 0, 0, 0, %[[ARG4]], %[[ARG6]]] [6, 1, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x1x1x1x2x2xf32> to tensor<6x1x1x1x1x1xf32>
// CHECK:      tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, 0, 0, 0, 0] [6, 1, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x1x1x1x1x1xf32> to tensor<6x1xf32>
// CHECK:      linalg.matmul
// CHECK:      %[[S12:.*]] = linalg.mul
// CHECK:      %[[S13:.*]] = tensor.empty() : tensor<1x4x1x1xf32>
// CHECK:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S12]] into %[[S13]][0, 0, 0, 0] [1, 4, 1, 1] [1, 1, 1, 1] : tensor<4x1xf32> into tensor<1x4x1x1xf32>
// CHECK:      %[[INSERTED_SLICE_5:.*]] = tensor.insert_slice %[[INSERTED_SLICE]] into %[[ARG7]][%[[ARG4]], 0, 0, %[[ARG6]]] [1, 4, 1, 1] [1, 1, 1, 1] : tensor<1x4x1x1xf32> into tensor<2x4x1x2xf32>
// CHECK:      scf.yield %[[INSERTED_SLICE_5]] : tensor<2x4x1x2xf32>
