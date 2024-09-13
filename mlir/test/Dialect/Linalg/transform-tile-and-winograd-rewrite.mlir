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
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @conv2d
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x10x10x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32> {
// CHECK:  %[[CST:.*]] = arith.constant 1.024000e+03 : f32
// CHECK:  %[[CST_0:.*]] = arith.constant dense<{{.*}}> : tensor<6x4xf32>
// CHECK:  %[[CST_1:.*]] = arith.constant dense<{{.*}}> : tensor<4x6xf32>
// CHECK:  %[[CST_2:.*]] = arith.constant dense<{{.*}}> : tensor<6x6xf32>
// CHECK:  %[[CST_3:.*]] = arith.constant dense<{{.*}}> : tensor<6x6xf32>
// CHECK:  %[[CST_4:.*]] = arith.constant dense<{{.*}}> : tensor<3x6xf32>
// CHECK:  %[[CST_5:.*]] = arith.constant dense<{{.*}}> : tensor<6x3xf32>
// CHECK:  %[[CST_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:  %[[C1:.*]] = arith.constant 1 : index
// CHECK:  %[[C5:.*]] = arith.constant 5 : index
// CHECK:  %[[C2:.*]] = arith.constant 2 : index
// CHECK:  %[[C0:.*]] = arith.constant 0 : index
// CHECK:  %[[S0:.*]] = tensor.empty()
// CHECK:  %[[S1:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S0]])
// CHECK:    %[[S9:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]])
// CHECK:      %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], 0, 0, %[[ARG5]]] [1, 3, 3, 1] [1, 1, 1, 1]
// CHECK:      %[[S10:.*]] = tensor.empty() : tensor<6x3xf32>
// CHECK:      %[[S11:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S10]] : tensor<6x3xf32>) -> tensor<6x3xf32>
// CHECK:      %[[S12:.*]] = linalg.matmul ins(%[[CST_5]], %[[EXTRACTED_SLICE]] : tensor<6x3xf32>, tensor<3x3xf32>) outs(%[[S11]] : tensor<6x3xf32>) -> tensor<6x3xf32>
// CHECK:      %[[S13:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK:      %[[S14:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S13]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:      %[[S15:.*]] = linalg.matmul ins(%[[S12]], %[[CST_4]] : tensor<6x3xf32>, tensor<3x6xf32>) outs(%[[S14]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S15]] into %[[ARG6]][0, 0, %[[ARG5]], %[[ARG3]]] [6, 6, 1, 1] [1, 1, 1, 1]
// CHECK:      scf.yield %[[INSERTED_SLICE]]
// CHECK:    scf.yield %[[S9]]
// CHECK:  %[[S2:.*]] = tensor.empty() : tensor<6x6x2x2x2x5xf32>
// CHECK:  %[[S3:.*]] = tensor.empty() : tensor<6x6x2x2x2x5xf32>
// CHECK:  %[[S4:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S3]])
// CHECK:    %[[S9:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]])
// CHECK:      %[[S10:.*]] = affine.apply #[[$MAP0]](%[[ARG3]])
// CHECK:      %[[S11:.*]] = affine.apply #[[$MAP0]](%[[ARG5]])
// CHECK:      %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, %[[S10]], %[[S11]], 0] [2, 6, 6, 5] [1, 1, 1, 1]
// CHECK:      %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[S2]][0, 0, %[[ARG3]], %[[ARG5]], 0, 0] [6, 6, 1, 1, 2, 5] [1, 1, 1, 1, 1, 1]
// CHECK:      %[[S12:.*]] = scf.for %[[ARG7:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.*]] = %[[EXTRACTED_SLICE_7]])
// CHECK:        %[[S13:.*]] = scf.for %[[ARG9:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG10:.*]] = %[[ARG8]])
// CHECK:          %[[EXTRACTED_SLICE_8:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG7]], 0, 0, %[[ARG9]]] [1, 6, 6, 1] [1, 1, 1, 1]
// CHECK:          %[[S14:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK:          %[[S15:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S14]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:          %[[S16:.*]] = linalg.matmul ins(%[[CST_3]], %[[EXTRACTED_SLICE_8]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S15]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:          %[[S17:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK:          %[[S18:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S17]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:          %[[S19:.*]] = linalg.matmul ins(%[[S16]], %[[CST_2]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S18]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:          %[[INSERTED_SLICE_9:.*]] = tensor.insert_slice %[[S19]] into %[[ARG10]][0, 0, 0, 0, %[[ARG7]], %[[ARG9]]] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1]
// CHECK:          scf.yield %[[INSERTED_SLICE_9]]
// CHECK:        scf.yield %[[S13]]
// CHECK:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S12]] into %[[ARG6]][0, 0, %[[ARG3]], %[[ARG5]], 0, 0] [6, 6, 1, 1, 2, 5] [1, 1, 1, 1, 1, 1]
// CHECK:      scf.yield %[[INSERTED_SLICE]]
// CHECK:    scf.yield %[[S9]]
// CHECK:  %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S1]] {{\[}}[0, 1], [2], [3]]
// CHECK:  %[[COLLAPSED_6:.*]] = tensor.collapse_shape %[[S4]] {{\[}}[0, 1], [2, 3, 4], [5]]
// CHECK:  %[[S6:.*]] = linalg.batch_matmul
// CHECK:  %[[EXPANDED:.*]] = tensor.expand_shape %[[S6]] {{\[}}[0, 1], [2, 3, 4], [5]] output_shape [6, 6, 2, 2, 2, 2]
// CHECK:  %[[S7:.*]] = tensor.empty() : tensor<2x8x8x2xf32>
// CHECK:  %[[S8:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S7]])
// CHECK:    %[[S9:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]])
// CHECK:      %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[EXPANDED]][0, 0, %[[ARG3]], %[[ARG5]], 0, 0] [6, 6, 1, 1, 2, 2] [1, 1, 1, 1, 1, 1]
// CHECK:      %[[S10:.*]] = affine.apply #[[$MAP0]](%[[ARG3]])
// CHECK:      %[[S11:.*]] = affine.apply #[[$MAP0]](%[[ARG5]])
// CHECK:      %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[ARG2]][0, %[[S10]], %[[S11]], 0] [2, 4, 4, 2] [1, 1, 1, 1]
// CHECK:      %[[S12:.*]] = scf.for %[[ARG7:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.*]] = %[[EXTRACTED_SLICE_7]])
// CHECK:        %[[S15:.*]] = scf.for %[[ARG9:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG10:.*]] = %[[ARG8]])
// CHECK:          %[[EXTRACTED_SLICE_8:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, 0, 0, %[[ARG7]], %[[ARG9]]] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1]
// CHECK:          %[[S16:.*]] = tensor.empty() : tensor<4x6xf32>
// CHECK:          %[[S17:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S16]] : tensor<4x6xf32>) -> tensor<4x6xf32>
// CHECK:          %[[S18:.*]] = linalg.matmul ins(%[[CST_1]], %[[EXTRACTED_SLICE_8]] : tensor<4x6xf32>, tensor<6x6xf32>) outs(%[[S17]] : tensor<4x6xf32>) -> tensor<4x6xf32>
// CHECK:          %[[S19:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK:          %[[S20:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S19]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:          %[[S21:.*]] = linalg.matmul ins(%[[S18]], %[[CST_0]] : tensor<4x6xf32>, tensor<6x4xf32>) outs(%[[S20]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:          %[[S22:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK:          %[[S23:.*]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%[[CST]] : f32) outs(%[[S22]] : tensor<4x4xf32>) {
// CHECK:          ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:            linalg.yield %[[IN]] : f32
// CHECK:          } -> tensor<4x4xf32>
// CHECK:          %[[S24:.*]] = linalg.mul ins(%[[S23]], %[[S21]] : tensor<4x4xf32>, tensor<4x4xf32>) outs(%[[S22]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:          %[[INSERTED_SLICE_9:.*]] = tensor.insert_slice %[[S24]] into %[[ARG10]][%[[ARG7]], 0, 0, %[[ARG9]]] [1, 4, 4, 1] [1, 1, 1, 1]
// CHECK:          scf.yield %[[INSERTED_SLICE_9]]
// CHECK:        scf.yield %[[S15]]
// CHECK:      %[[S13:.*]] = affine.apply #[[$MAP0]](%[[ARG3]])
// CHECK:      %[[S14:.*]] = affine.apply #[[$MAP0]](%[[ARG5]])
// CHECK:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S12]] into %[[ARG6]][0, %[[S13]], %[[S14]], 0] [2, 4, 4, 2] [1, 1, 1, 1]
// CHECK:      scf.yield %[[INSERTED_SLICE]]
// CHECK:    scf.yield %[[S9]]

// -----

func.func @conv2d_unaligned(%arg0: tensor<2x11x11x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32> {
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
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<36x18x2xf32>) -> tensor<36x18x2xf32>
  %6 = linalg.batch_matmul ins(%collapsed_0, %collapsed : tensor<36x18x5xf32>, tensor<36x5x2xf32>) outs(%5 : tensor<36x18x2xf32>) -> tensor<36x18x2xf32>
  %expanded = tensor.expand_shape %6 [[0, 1], [2, 3, 4], [5]] output_shape [6, 6, 3, 3, 2, 2] : tensor<36x18x2xf32> into tensor<6x6x3x3x2x2xf32>
  %padded_1 = tensor.pad %arg2 low[0, 0, 0, 0] high[0, 3, 3, 0] {
  ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
    tensor.yield %cst : f32
  } : tensor<2x9x9x2xf32> to tensor<2x12x12x2xf32>
  %7 = linalg.winograd_output_transform m(4) r(3) ins(%expanded : tensor<6x6x3x3x2x2xf32>) outs(%padded_1 : tensor<2x12x12x2xf32>) -> tensor<2x12x12x2xf32>
  %extracted_slice = tensor.extract_slice %7[0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1] : tensor<2x12x12x2xf32> to tensor<2x9x9x2xf32>
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
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @conv2d_unaligned
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x11x11x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32> {
// CHECK:  %[[CST:.*]] = arith.constant 1.024000e+03 : f32
// CHECK:  %[[CST_0:.*]] = arith.constant dense<{{.*}}> : tensor<6x4xf32>
// CHECK:  %[[CST_1:.*]] = arith.constant dense<{{.*}}> : tensor<4x6xf32>
// CHECK:  %[[CST_2:.*]] = arith.constant dense<{{.*}}> : tensor<6x6xf32>
// CHECK:  %[[CST_3:.*]] = arith.constant dense<{{.*}}> : tensor<6x6xf32>
// CHECK:  %[[C3:.*]] = arith.constant 3 : index
// CHECK:  %[[CST_4:.*]] = arith.constant dense<{{.*}}> : tensor<3x6xf32>
// CHECK:  %[[CST_5:.*]] = arith.constant dense<{{.*}}> : tensor<6x3xf32>
// CHECK:  %[[C1:.*]] = arith.constant 1 : index
// CHECK:  %[[C5:.*]] = arith.constant 5 : index
// CHECK:  %[[C2:.*]] = arith.constant 2 : index
// CHECK:  %[[C0:.*]] = arith.constant 0 : index
// CHECK:  %[[CST_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:  %[[S0:.*]] = tensor.empty()
// CHECK:  %[[S1:.*]] = scf.for %[[ARG4:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG5:.*]] = %[[S0]])
// CHECK:    %[[S9:.*]] = scf.for %[[ARG6:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG7:.*]] = %[[ARG5]])
// CHECK:      %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG4]], 0, 0, %[[ARG6]]] [1, 3, 3, 1] [1, 1, 1, 1]
// CHECK:      %[[S11:.*]] = tensor.empty() : tensor<6x3xf32>
// CHECK:      %[[S12:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S11]] : tensor<6x3xf32>) -> tensor<6x3xf32>
// CHECK:      %[[S13:.*]] = linalg.matmul ins(%[[CST_5]], %[[EXTRACTED_SLICE_9]] : tensor<6x3xf32>, tensor<3x3xf32>) outs(%[[S12]] : tensor<6x3xf32>) -> tensor<6x3xf32>
// CHECK:      %[[S14:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK:      %[[S15:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S14]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:      %[[S16:.*]] = linalg.matmul ins(%[[S13]], %[[CST_4]] : tensor<6x3xf32>, tensor<3x6xf32>) outs(%[[S15]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S16]] into %[[ARG7]][0, 0, %[[ARG6]], %[[ARG4]]] [6, 6, 1, 1] [1, 1, 1, 1]
// CHECK:      scf.yield %[[INSERTED_SLICE]] : tensor<6x6x5x2xf32>
// CHECK:    scf.yield %[[S9]] : tensor<6x6x5x2xf32>
// CHECK:  %[[PADDED:.*]] = tensor.pad %[[ARG0]] low[0, 0, 0, 0] high[0, 3, 3, 0]
// CHECK:  %[[S2:.*]] = tensor.empty() : tensor<6x6x3x3x2x5xf32>
// CHECK:  %[[S3:.*]] = tensor.empty() : tensor<6x6x3x3x2x5xf32>
// CHECK:  %[[S4:.*]] = scf.for %[[ARG4:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG5:.*]] = %[[S3]])
// CHECK:    %[[S9:.*]] = scf.for %[[ARG6:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG7:.*]] = %[[ARG5]])
// CHECK:      %[[S10:.*]] = affine.apply #[[$MAP0]](%[[ARG4]])
// CHECK:      %[[S11:.*]] = affine.apply #[[$MAP0]](%[[ARG6]])
// CHECK:      %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[PADDED]][0, %[[S10]], %[[S11]], 0] [2, 6, 6, 5] [1, 1, 1, 1]
// CHECK:      %[[EXTRACTED_SLICE_10:.*]] = tensor.extract_slice %[[S2]][0, 0, %[[ARG4]], %[[ARG6]], 0, 0] [6, 6, 1, 1, 2, 5] [1, 1, 1, 1, 1, 1]
// CHECK:      %[[S12:.*]] = scf.for %[[ARG8:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG9:.*]] = %[[EXTRACTED_SLICE_10]])
// CHECK:        %[[S13:.*]] = scf.for %[[ARG10:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG11:.*]] = %[[ARG9]])
// CHECK:          %[[EXTRACTED_SLICE_11:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE_9]][%[[ARG8]], 0, 0, %[[ARG10]]] [1, 6, 6, 1] [1, 1, 1, 1]
// CHECK:          %[[S15:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK:          %[[S16:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S15]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:          %[[S17:.*]] = linalg.matmul ins(%[[CST_3]], %[[EXTRACTED_SLICE_11]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S16]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:          %[[S18:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK:          %[[S19:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S18]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:          %[[S20:.*]] = linalg.matmul ins(%[[S17]], %[[CST_2]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S19]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:          %[[INSERTED_SLICE_12:.*]] = tensor.insert_slice %[[S20]] into %[[ARG11]][0, 0, 0, 0, %[[ARG8]], %[[ARG10]]] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1]
// CHECK:          scf.yield %[[INSERTED_SLICE_12]] : tensor<6x6x1x1x2x5xf32>
// CHECK:        scf.yield %[[S13]] : tensor<6x6x1x1x2x5xf32>
// CHECK:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S12]] into %[[ARG7]][0, 0, %[[ARG4]], %[[ARG6]], 0, 0] [6, 6, 1, 1, 2, 5] [1, 1, 1, 1, 1, 1]
// CHECK:      scf.yield %[[INSERTED_SLICE]]
// CHECK:    scf.yield %[[S9]]
// CHECK:  %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S1]] {{\[}}[0, 1], [2], [3]]
// CHECK:  %[[COLLAPSED_7:.*]] = tensor.collapse_shape %[[S4]] {{\[}}[0, 1], [2, 3, 4], [5]]
// CHECK:  %[[S6:.*]] = linalg.batch_matmul
// CHECK:  %[[EXPANDED:.*]] = tensor.expand_shape %[[S6]] {{\[}}[0, 1], [2, 3, 4], [5]] output_shape [6, 6, 3, 3, 2, 2]
// CHECK:  %[[PADDED_8:.*]] = tensor.pad %[[ARG2]] low[0, 0, 0, 0] high[0, 3, 3, 0]
// CHECK:  %[[S7:.*]] = tensor.empty() : tensor<2x12x12x2xf32>
// CHECK:  %[[S8:.*]] = scf.for %[[ARG4:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG5:.*]] = %[[S7]])
// CHECK:    %[[S9:.*]] = scf.for %[[ARG6:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG7:.*]] = %[[ARG5]])
// CHECK:      %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[EXPANDED]][0, 0, %[[ARG4]], %[[ARG6]], 0, 0] [6, 6, 1, 1, 2, 2] [1, 1, 1, 1, 1, 1]
// CHECK:      %[[S10:.*]] = affine.apply #[[$MAP0]](%[[ARG4]])
// CHECK:      %[[S11:.*]] = affine.apply #[[$MAP0]](%[[ARG6]])
// CHECK:      %[[EXTRACTED_SLICE_10:.*]] = tensor.extract_slice %[[PADDED_8]][0, %[[S10]], %[[S11]], 0] [2, 4, 4, 2] [1, 1, 1, 1]
// CHECK:      %[[S12:.*]] = scf.for %[[ARG8:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG9:.*]] = %[[EXTRACTED_SLICE_10]])
// CHECK:        %[[S15:.*]] = scf.for %[[ARG10:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG11:.*]] = %[[ARG9]])
// CHECK:          %[[EXTRACTED_SLICE_11:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE_9]][0, 0, 0, 0, %[[ARG8]], %[[ARG10]]] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1]
// CHECK:          %[[S17:.*]] = tensor.empty() : tensor<4x6xf32>
// CHECK:          %[[S18:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S17]] : tensor<4x6xf32>) -> tensor<4x6xf32>
// CHECK:          %[[S19:.*]] = linalg.matmul ins(%[[CST_1]], %[[EXTRACTED_SLICE_11]] : tensor<4x6xf32>, tensor<6x6xf32>) outs(%[[S18]] : tensor<4x6xf32>) -> tensor<4x6xf32>
// CHECK:          %[[S20:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK:          %[[S21:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S20]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:          %[[S22:.*]] = linalg.matmul ins(%[[S19]], %[[CST_0]] : tensor<4x6xf32>, tensor<6x4xf32>) outs(%[[S21]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:          %[[S23:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK:          %[[S24:.*]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%[[CST]] : f32) outs(%[[S23]] : tensor<4x4xf32>) {
// CHECK:          ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:            linalg.yield %[[IN]] : f32
// CHECK:          } -> tensor<4x4xf32>
// CHECK:          %[[S25:.*]] = linalg.mul ins(%[[S24]], %[[S22]] : tensor<4x4xf32>, tensor<4x4xf32>) outs(%[[S23]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:          %[[INSERTED_SLICE_12:.*]] = tensor.insert_slice %[[S25]] into %[[ARG11]][%[[ARG8]], 0, 0, %[[ARG10]]] [1, 4, 4, 1] [1, 1, 1, 1]
// CHECK:          scf.yield %[[INSERTED_SLICE_12]]
// CHECK:        scf.yield %[[S15]] : tensor<2x4x4x2xf32>
// CHECK:      %[[S13:.*]] = affine.apply #[[$MAP0]](%[[ARG4]])
// CHECK:      %[[S14:.*]] = affine.apply #[[$MAP0]](%[[ARG6]])
// CHECK:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S12]] into %[[ARG7]][0, %[[S13]], %[[S14]], 0] [2, 4, 4, 2] [1, 1, 1, 1]
// CHECK:      scf.yield %[[INSERTED_SLICE]]
// CHECK:    scf.yield %[[S9]]
// CHECK:  %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[S8]][0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1]
// CHECK:  return %[[EXTRACTED_SLICE]]

// -----

func.func @conv2d_mx1_rx1(%arg0: tensor<2x6x1x5xf32>, %arg1: tensor<2x3x1x5xf32>, %arg2: tensor<2x4x1x2xf32>) -> tensor<2x4x1x2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<6x1x5x2xf32>
  %1 = linalg.winograd_filter_transform m(4) r(3) ins(%arg1 : tensor<2x3x1x5xf32>) outs(%0 : tensor<6x1x5x2xf32>) -> tensor<6x1x5x2xf32>
  %2 = tensor.empty() : tensor<6x1x1x1x2x5xf32>
  %3 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x6x1x5xf32>) outs(%2 : tensor<6x1x1x1x2x5xf32>) -> tensor<6x1x1x1x2x5xf32>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<6x1x5x2xf32> into tensor<6x5x2xf32>
  %collapsed_0 = tensor.collapse_shape %3 [[0, 1], [2, 3, 4], [5]] : tensor<6x1x1x1x2x5xf32> into tensor<6x2x5xf32>
  %4 = tensor.empty() : tensor<6x2x2xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
  %6 = linalg.batch_matmul ins(%collapsed_0, %collapsed : tensor<6x2x5xf32>, tensor<6x5x2xf32>) outs(%5 : tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
  %expanded = tensor.expand_shape %6 [[0, 1], [2, 3, 4], [5]] output_shape [6, 1, 1, 1, 2, 2] : tensor<6x2x2xf32> into tensor<6x1x1x1x2x2xf32>
  %7 = linalg.winograd_output_transform m(4) r(3) ins(%expanded : tensor<6x1x1x1x2x2xf32>) outs(%arg2 : tensor<2x4x1x2xf32>) -> tensor<2x4x1x2xf32>
  return %7 : tensor<2x4x1x2xf32>
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

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @conv2d_mx1_rx1
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x6x1x5xf32>, %[[ARG1:.*]]: tensor<2x3x1x5xf32>, %[[ARG2:.*]]: tensor<2x4x1x2xf32>) -> tensor<2x4x1x2xf32> {
// CHECK:   %[[CST:.*]] = arith.constant 3.200000e+01 : f32
// CHECK:  %[[CST_0:.*]] = arith.constant dense<{{.*}}> : tensor<4x6xf32>
// CHECK:  %[[CST_1:.*]] = arith.constant dense<{{.*}}> : tensor<6x6xf32>
// CHECK:  %[[CST_2:.*]] = arith.constant dense<{{.*}}> : tensor<6x3xf32>
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[C5:.*]] = arith.constant 5 : index
// CHECK:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[CST_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:   %[[S0:.*]] = tensor.empty() : tensor<6x1x5x2xf32>
// CHECK:   %[[S1:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S0]])
// CHECK:     %[[S7:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]])
// CHECK:       %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], 0, 0, %[[ARG5]]] [1, 3, 1, 1] [1, 1, 1, 1]
// CHECK:       %[[S8:.*]] = tensor.empty() : tensor<6x1xf32>
// CHECK:       %[[S9:.*]] = linalg.fill ins(%[[CST_3]] : f32) outs(%[[S8]] : tensor<6x1xf32>) -> tensor<6x1xf32>
// CHECK:       %[[S10:.*]] = linalg.matmul ins(%[[CST_2]], %[[EXTRACTED_SLICE]] : tensor<6x3xf32>, tensor<3x1xf32>) outs(%[[S9]] : tensor<6x1xf32>) -> tensor<6x1xf32>
// CHECK:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S10]] into %[[ARG6]][0, 0, %[[ARG5]], %[[ARG3]]] [6, 1, 1, 1] [1, 1, 1, 1]
// CHECK:       scf.yield %[[INSERTED_SLICE]]
// CHECK:     scf.yield %[[S7]]
// CHECK:   %[[S2:.*]] = tensor.empty() : tensor<6x1x1x1x2x5xf32>
// CHECK:   %[[S3:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S2]])
// CHECK:     %[[S7:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]])
// CHECK:       %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], 0, 0, %[[ARG5]]] [1, 6, 1, 1] [1, 1, 1, 1]
// CHECK:       %[[S8:.*]] = tensor.empty() : tensor<6x1xf32>
// CHECK:       %[[S9:.*]] = linalg.fill ins(%[[CST_3]] : f32) outs(%[[S8]] : tensor<6x1xf32>) -> tensor<6x1xf32>
// CHECK:       %[[S10:.*]] = linalg.matmul ins(%[[CST_1]], %[[EXTRACTED_SLICE]] : tensor<6x6xf32>, tensor<6x1xf32>) outs(%[[S9]] : tensor<6x1xf32>) -> tensor<6x1xf32>
// CHECK:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S10]] into %[[ARG6]][0, 0, 0, 0, %[[ARG3]], %[[ARG5]]] [6, 1, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1]
// CHECK:       scf.yield %[[INSERTED_SLICE]]
// CHECK:     scf.yield %[[S7]]
// CHECK:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S1]] {{\[}}[0, 1], [2], [3]]
// CHECK:   %[[COLLAPSED_3:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1], [2, 3, 4], [5]]
// CHECK:   %[[S4:.*]] = tensor.empty() : tensor<6x2x2xf32>
// CHECK:   %[[S5:.*]] = linalg.fill ins(%[[CST_3]] : f32) outs(%[[S4]] : tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
// CHECK:   %[[S6:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_3]], %[[COLLAPSED]] : tensor<6x2x5xf32>, tensor<6x5x2xf32>) outs(%[[S5]] : tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
// CHECK:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S6]] {{\[}}[0, 1], [2, 3, 4], [5]] output_shape [6, 1, 1, 1, 2, 2]
// CHECK:   %[[S6:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[ARG2]])
// CHECK:     %[[S7:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]])
// CHECK:       %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[EXPANDED]][0, 0, 0, 0, %[[ARG3]], %[[ARG5]]] [6, 1, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1]
// CHECK:       %[[S9:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK:       %[[S10:.*]] = linalg.fill ins(%[[CST_3]] : f32) outs(%[[S9]] : tensor<4x1xf32>) -> tensor<4x1xf32>
// CHECK:       %[[S11:.*]] = linalg.matmul ins(%[[CST_0]], %[[EXTRACTED_SLICE]] : tensor<4x6xf32>, tensor<6x1xf32>) outs(%[[S10]] : tensor<4x1xf32>) -> tensor<4x1xf32>
// CHECK:       %[[S12:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK:       %[[S13:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%[[CST]] : f32) outs(%[[S12]] : tensor<4x1xf32>) {
// CHECK:       ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:         linalg.yield %[[IN]] : f32
// CHECK:       } -> tensor<4x1xf32>
// CHECK:       %[[S14:.*]] = linalg.mul ins(%[[S13]], %[[S11]] : tensor<4x1xf32>, tensor<4x1xf32>) outs(%[[S12]] : tensor<4x1xf32>) -> tensor<4x1xf32>
// CHECK:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S14]] into %[[ARG6]][%[[ARG3]], 0, 0, %[[ARG5]]] [1, 4, 1, 1] [1, 1, 1, 1]
// CHECK:       scf.yield %[[INSERTED_SLICE]]
// CHECK:     scf.yield %[[S7]]
// CHECK:   return %[[S6]]
