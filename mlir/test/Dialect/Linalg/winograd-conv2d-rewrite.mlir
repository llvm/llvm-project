// RUN: mlir-opt %s -split-input-file -test-linalg-transform-patterns=test-decompose-winograd-ops | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @conv2d_4x4_3x3(%arg0: tensor<2x6x6x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x4x4x2xf32> {
  %0 = tensor.empty() : tensor<2x4x4x2xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1xf32>) outs(%0 : tensor<2x4x4x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x4x4x2xf32>
  %2 = tensor.empty() : tensor<1x1x6x6x5x2xf32>
  %3 = linalg.winograd_filter_transform m(4) r(3) ins(%arg1 : tensor<2x3x3x5xf32>) outs(%2 : tensor<1x1x6x6x5x2xf32>) -> tensor<1x1x6x6x5x2xf32>
  %4 = tensor.empty() : tensor<1x1x6x6x2x5xf32>
  %5 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x6x6x5xf32>) outs(%4 : tensor<1x1x6x6x2x5xf32>) -> tensor<1x1x6x6x2x5xf32>
  %collapsed = tensor.collapse_shape %3 [[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x6x5x2xf32> into tensor<36x5x2xf32>
  %collapsed_0 = tensor.collapse_shape %5 [[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x6x2x5xf32> into tensor<36x2x5xf32>
  %6 = tensor.empty() : tensor<36x2x2xf32>
  %7 = linalg.batch_matmul ins(%collapsed_0, %collapsed : tensor<36x2x5xf32>, tensor<36x5x2xf32>) outs(%6 : tensor<36x2x2xf32>) -> tensor<36x2x2xf32>
  %expanded = tensor.expand_shape %7 [[0, 1, 2, 3], [4], [5]] output_shape [1, 1, 6, 6, 2, 2] : tensor<36x2x2xf32> into tensor<1x1x6x6x2x2xf32>
  %8 = linalg.winograd_output_transform m(4) r(3) ins(%expanded : tensor<1x1x6x6x2x2xf32>) outs(%1 : tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32>
  return %8 : tensor<2x4x4x2xf32>
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @conv2d_4x4_3x3
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x6x6x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<1xf32>) -> tensor<2x4x4x2xf32> {
// CHECK-DAG:   %[[CST:.*]] = arith.constant 1.024000e+03 : f32
// CHECK-DAG:   %[[CST_0:.*]] = arith.constant dense<{{\[}}[1.250000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00], [2.500000e-01, -2.500000e-01, 2.500000e-01, -2.500000e-01], [2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01], [1.250000e-01, -2.500000e-01, 5.000000e-01, -1.000000e+00], [1.250000e-01, 2.500000e-01, 5.000000e-01, 1.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 5.000000e-01]]> : tensor<6x4xf32>
// CHECK-DAG:   %[[CST_1:.*]] = arith.constant dense<{{\[}}[1.250000e-01, 2.500000e-01, 2.500000e-01, 1.250000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -2.500000e-01, 2.500000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 2.500000e-01, 5.000000e-01, 5.000000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -1.000000e+00, 1.000000e+00, 5.000000e-01]]> : tensor<4x6xf32>
// CHECK-DAG:   %[[CST_2:.*]] = arith.constant dense<{{\[}}[2.500000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 2.500000e-01, -2.500000e-01, 2.500000e-01, -2.500000e-01, 2.500000e-01], [-3.125000e-01, -2.500000e-01, -2.500000e-01, -1.250000e-01, -1.250000e-01, 0.000000e+00], [0.000000e+00, -6.250000e-02, 6.250000e-02, -2.500000e-01, 2.500000e-01, -3.125000e-01], [6.250000e-02, 6.250000e-02, 6.250000e-02, 1.250000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 6.250000e-02]]> : tensor<6x6xf32>
// CHECK-DAG:   %[[CST_3:.*]] = arith.constant dense<{{\[}}[2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -2.500000e-01, -6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, -2.500000e-01, -2.500000e-01, 6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -1.250000e-01, -2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, -1.250000e-01, 2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02]]> : tensor<6x6xf32>
// CHECK-DAG:   %[[CST_4:.*]] = arith.constant dense<{{\[}}[1.000000e+00, -0.333333343, -0.333333343, 0.0833333358, 0.0833333358, 0.000000e+00], [0.000000e+00, 0.333333343, -0.333333343, -0.166666672, 0.166666672, 0.000000e+00], [0.000000e+00, -0.333333343, -0.333333343, 0.333333343, 0.333333343, 1.000000e+00]]> : tensor<3x6xf32>
// CHECK-DAG:   %[[CST_5:.*]] = arith.constant dense<{{\[}}[1.000000e+00, 0.000000e+00, 0.000000e+00], [-0.333333343, 0.333333343, -0.333333343], [-0.333333343, -0.333333343, -0.333333343], [0.0833333358, -0.166666672, 0.333333343], [0.0833333358, 0.166666672, 0.333333343], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<6x3xf32>
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[S0:.*]] = tensor.empty() : tensor<2x4x4x2xf32>
// CHECK-NEXT:  %[[S1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG2]] : tensor<1xf32>) outs(%[[S0]] : tensor<2x4x4x2xf32>) {
// CHECK-NEXT:  ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:    linalg.yield %[[IN]] : f32
// CHECK-NEXT:  } -> tensor<2x4x4x2xf32>
// CHECK-NEXT:  %[[S2:.*]] = tensor.empty() : tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:  %[[S3:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S2]]) -> (tensor<1x1x6x6x5x2xf32>) {
// CHECK-NEXT:    %[[S9:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<1x1x6x6x5x2xf32>) {
// CHECK-NEXT:      %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], 0, 0, %[[ARG5]]] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<2x3x3x5xf32> to tensor<1x3x3x1xf32>
// CHECK-NEXT:      %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, 0, 0] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<1x3x3x1xf32> to tensor<3x3xf32>
// CHECK-NEXT:      %[[S10:.*]] = tensor.empty() : tensor<6x3xf32>
// CHECK-NEXT:      %[[S11:.*]] = linalg.matmul ins(%[[CST_5]], %[[EXTRACTED_SLICE_7]] : tensor<6x3xf32>, tensor<3x3xf32>) outs(%[[S10]] : tensor<6x3xf32>) -> tensor<6x3xf32>
// CHECK-NEXT:      %[[S12:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:      %[[S13:.*]] = linalg.matmul ins(%[[S11]], %[[CST_4]] : tensor<6x3xf32>, tensor<3x6xf32>) outs(%[[S12]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:      %[[S14:.*]] = tensor.empty() : tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S13]] into %[[S14]][0, 0, 0, 0, 0, 0] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6xf32> into tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:      %[[INSERTED_SLICE_8:.*]] = tensor.insert_slice %[[INSERTED_SLICE]] into %[[ARG6]][0, 0, 0, 0, %[[ARG5]], %[[ARG3]]] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x1x1xf32> into tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:      scf.yield %[[INSERTED_SLICE_8]] : tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield %[[S9]] : tensor<1x1x6x6x5x2xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[S4:.*]] = tensor.empty() : tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:  %[[S5:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S4]]) -> (tensor<1x1x6x6x2x5xf32>) {
// CHECK-NEXT:    %[[S9:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<1x1x6x6x2x5xf32>) {
// CHECK-NEXT:      %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], 0, 0, %[[ARG5]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<2x6x6x5xf32> to tensor<1x6x6x1xf32>
// CHECK-NEXT:      %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, 0, 0] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<1x6x6x1xf32> to tensor<6x6xf32>
// CHECK-NEXT:      %[[S10:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:      %[[S11:.*]] = linalg.matmul ins(%[[CST_3]], %[[EXTRACTED_SLICE_7]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S10]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:      %[[S12:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:      %[[S13:.*]] = linalg.matmul ins(%[[S11]], %[[CST_2]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S12]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:      %[[S14:.*]] = tensor.empty() : tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S13]] into %[[S14]][0, 0, 0, 0, 0, 0] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6xf32> into tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:      %[[INSERTED_SLICE_8:.*]] = tensor.insert_slice %[[INSERTED_SLICE]] into %[[ARG6]][0, 0, 0, 0, %[[ARG3]], %[[ARG5]]] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x1x1xf32> into tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:      scf.yield %[[INSERTED_SLICE_8]] : tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield %[[S9]] : tensor<1x1x6x6x2x5xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x6x5x2xf32> into tensor<36x5x2xf32>
// CHECK-NEXT:  %[[COLLAPSED_6:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1, 2, 3], [4], [5]] : tensor<1x1x6x6x2x5xf32> into tensor<36x2x5xf32>
// CHECK-NEXT:  %[[S6:.*]] = tensor.empty() : tensor<36x2x2xf32>
// CHECK-NEXT:  %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_6]], %[[COLLAPSED]] : tensor<36x2x5xf32>, tensor<36x5x2xf32>) outs(%[[S6]] : tensor<36x2x2xf32>) -> tensor<36x2x2xf32>
// CHECK-NEXT:  %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1, 2, 3], [4], [5]] output_shape [1, 1, 6, 6, 2, 2] : tensor<36x2x2xf32> into tensor<1x1x6x6x2x2xf32>
// CHECK-NEXT:  %[[S8:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S1]]) -> (tensor<2x4x4x2xf32>) {
// CHECK-NEXT:    %[[S9:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<2x4x4x2xf32>) {
// CHECK-NEXT:      %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[EXPANDED]][0, 0, 0, 0, %[[ARG3]], %[[ARG5]]] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x2x2xf32> to tensor<1x1x6x6x1x1xf32>
// CHECK-NEXT:      %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, 0, 0, 0, 0] [1, 1, 6, 6, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x6x6x1x1xf32> to tensor<6x6xf32>
// CHECK-NEXT:      %[[S10:.*]] = tensor.empty() : tensor<4x6xf32>
// CHECK-NEXT:      %[[S11:.*]] = linalg.matmul ins(%[[CST_1]], %[[EXTRACTED_SLICE_7]] : tensor<4x6xf32>, tensor<6x6xf32>) outs(%[[S10]] : tensor<4x6xf32>) -> tensor<4x6xf32>
// CHECK-NEXT:      %[[S12:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:      %[[S13:.*]] = linalg.matmul ins(%[[S11]], %[[CST_0]] : tensor<4x6xf32>, tensor<6x4xf32>) outs(%[[S12]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      %[[S14:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:      %[[S15:.*]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP3]], #[[$MAP3]]], iterator_types = ["parallel", "parallel"]} ins(%[[CST]], %[[S13]] : f32, tensor<4x4xf32>) outs(%[[S14]] : tensor<4x4xf32>) {
// CHECK-NEXT:      ^bb0(%[[IN:.*]]: f32, %[[IN_9:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:        %[[S17:.*]] = arith.mulf %[[IN]], %[[IN_9]] : f32
// CHECK-NEXT:        linalg.yield %[[S17]] : f32
// CHECK-NEXT:      } -> tensor<4x4xf32>
// CHECK-NEXT:      %[[S16:.*]] = tensor.empty() : tensor<1x4x4x1xf32>
// CHECK-NEXT:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S15]] into %[[S16]][0, 0, 0, 0] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<4x4xf32> into tensor<1x4x4x1xf32>
// CHECK-NEXT:      %[[INSERTED_SLICE_8:.*]] = tensor.insert_slice %[[INSERTED_SLICE]] into %[[ARG6]][%[[ARG3]], 0, 0, %[[ARG5]]] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<1x4x4x1xf32> into tensor<2x4x4x2xf32>
// CHECK-NEXT:      scf.yield %[[INSERTED_SLICE_8]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield %[[S9]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[S8]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:}
