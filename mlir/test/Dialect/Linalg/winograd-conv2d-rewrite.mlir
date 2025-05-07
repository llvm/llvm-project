// RUN: mlir-opt %s -split-input-file -test-linalg-transform-patterns=test-decompose-winograd-ops | FileCheck %s

func.func @conv2d(%arg0: tensor<2x11x11x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %2 = tensor.empty() : tensor<6x6x5x2xf32>
  %3 = linalg.winograd_filter_transform m(4) r(3) ins(%arg1 : tensor<2x3x3x5xf32>) outs(%2 : tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32>
  %padded = tensor.pad %arg0 low[0, 0, 0, 0] high[0, 3, 3, 0] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %cst : f32
  } : tensor<2x11x11x5xf32> to tensor<2x14x14x5xf32>
  %4 = tensor.empty() : tensor<6x6x3x3x2x5xf32>
  %5 = linalg.winograd_input_transform m(4) r(3) ins(%padded : tensor<2x14x14x5xf32>) outs(%4 : tensor<6x6x3x3x2x5xf32>) -> tensor<6x6x3x3x2x5xf32>
  %collapsed = tensor.collapse_shape %3 [[0, 1], [2], [3]] : tensor<6x6x5x2xf32> into tensor<36x5x2xf32>
  %collapsed_0 = tensor.collapse_shape %5 [[0, 1], [2, 3, 4], [5]] : tensor<6x6x3x3x2x5xf32> into tensor<36x18x5xf32>
  %6 = tensor.empty() : tensor<36x18x2xf32>
  %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<36x18x2xf32>) -> tensor<36x18x2xf32>
  %8 = linalg.batch_matmul ins(%collapsed_0, %collapsed : tensor<36x18x5xf32>, tensor<36x5x2xf32>) outs(%7 : tensor<36x18x2xf32>) -> tensor<36x18x2xf32>
  %expanded = tensor.expand_shape %8 [[0, 1], [2, 3, 4], [5]] output_shape [6, 6, 3, 3, 2, 2] : tensor<36x18x2xf32> into tensor<6x6x3x3x2x2xf32>
  %padded_1 = tensor.pad %arg2 low[0, 0, 0, 0] high[0, 3, 3, 0] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %cst : f32
  } : tensor<2x9x9x2xf32> to tensor<2x12x12x2xf32>
  %9 = linalg.winograd_output_transform m(4) r(3) ins(%expanded : tensor<6x6x3x3x2x2xf32>) outs(%padded_1 : tensor<2x12x12x2xf32>) -> tensor<2x12x12x2xf32>
  %extracted_slice = tensor.extract_slice %9[0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1] : tensor<2x12x12x2xf32> to tensor<2x9x9x2xf32>
  return %extracted_slice : tensor<2x9x9x2xf32>
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @conv2d
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x11x11x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32> {
// CHECK-DAG:   %[[CST:.*]] = arith.constant 1.024000e+03 : f32
// CHECK-DAG:   %[[CST_0:.*]] = arith.constant dense<{{\[}}[1.250000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00], [2.500000e-01, -2.500000e-01, 2.500000e-01, -2.500000e-01], [2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01], [1.250000e-01, -2.500000e-01, 5.000000e-01, -1.000000e+00], [1.250000e-01, 2.500000e-01, 5.000000e-01, 1.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 5.000000e-01]]> : tensor<6x4xf32>
// CHECK-DAG:   %[[CST_1:.*]] = arith.constant dense<{{\[}}[1.250000e-01, 2.500000e-01, 2.500000e-01, 1.250000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -2.500000e-01, 2.500000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 2.500000e-01, 5.000000e-01, 5.000000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -1.000000e+00, 1.000000e+00, 5.000000e-01]]> : tensor<4x6xf32>
// CHECK-DAG:   %[[CST_2:.*]] = arith.constant dense<{{\[}}[2.500000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 2.500000e-01, -2.500000e-01, 2.500000e-01, -2.500000e-01, 2.500000e-01], [-3.125000e-01, -2.500000e-01, -2.500000e-01, -1.250000e-01, -1.250000e-01, 0.000000e+00], [0.000000e+00, -6.250000e-02, 6.250000e-02, -2.500000e-01, 2.500000e-01, -3.125000e-01], [6.250000e-02, 6.250000e-02, 6.250000e-02, 1.250000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 6.250000e-02]]> : tensor<6x6xf32>
// CHECK-DAG:   %[[CST_3:.*]] = arith.constant dense<{{\[}}[2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -2.500000e-01, -6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, -2.500000e-01, -2.500000e-01, 6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -1.250000e-01, -2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, -1.250000e-01, 2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02]]> : tensor<6x6xf32>
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[CST_4:.*]] = arith.constant dense<{{\[}}[1.000000e+00, -0.333333343, -0.333333343, 0.0833333358, 0.0833333358, 0.000000e+00], [0.000000e+00, 0.333333343, -0.333333343, -0.166666672, 0.166666672, 0.000000e+00], [0.000000e+00, -0.333333343, -0.333333343, 0.333333343, 0.333333343, 1.000000e+00]]> : tensor<3x6xf32>
// CHECK-DAG:   %[[CST_5:.*]] = arith.constant dense<{{\[}}[1.000000e+00, 0.000000e+00, 0.000000e+00], [-0.333333343, 0.333333343, -0.333333343], [-0.333333343, -0.333333343, -0.333333343], [0.0833333358, -0.166666672, 0.333333343], [0.0833333358, 0.166666672, 0.333333343], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<6x3xf32>
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[CST_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[S0:.*]] = tensor.empty() : tensor<6x6x5x2xf32>
// CHECK-NEXT:   %[[S1:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S0]]) -> (tensor<6x6x5x2xf32>) {
// CHECK-NEXT:     %[[S7:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<6x6x5x2xf32>) {
// CHECK-NEXT:       %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], %[[C0]], %[[C0]], %[[ARG5]]] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<2x3x3x5xf32> to tensor<3x3xf32>
// CHECK-NEXT:       %[[S9:.*]] = tensor.empty() : tensor<6x3xf32>
// CHECK-NEXT:       %[[S10:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S9]] : tensor<6x3xf32>) -> tensor<6x3xf32>
// CHECK-NEXT:       %[[S11:.*]] = linalg.matmul ins(%[[CST_5]], %[[EXTRACTED_SLICE_9]] : tensor<6x3xf32>, tensor<3x3xf32>) outs(%[[S10]] : tensor<6x3xf32>) -> tensor<6x3xf32>
// CHECK-NEXT:       %[[S12:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:       %[[S13:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S12]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:       %[[S14:.*]] = linalg.matmul ins(%[[S11]], %[[CST_4]] : tensor<6x3xf32>, tensor<3x6xf32>) outs(%[[S13]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S14]] into %[[ARG6]][%[[C0]], %[[C0]], %[[ARG5]], %[[ARG3]]] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6xf32> into tensor<6x6x5x2xf32>
// CHECK-NEXT:       scf.yield %[[INSERTED_SLICE]] : tensor<6x6x5x2xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S7]] : tensor<6x6x5x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[PADDED:.*]] = tensor.pad %[[ARG0]] low[0, 0, 0, 0] high[0, 3, 3, 0] {
// CHECK-NEXT:   ^bb0(%[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index, %[[ARG6:.*]]: index):
// CHECK-NEXT:     tensor.yield %[[CST_6]] : f32
// CHECK-NEXT:   } : tensor<2x11x11x5xf32> to tensor<2x14x14x5xf32>
// CHECK-NEXT:   %[[S2:.*]] = tensor.empty() : tensor<6x6x3x3x2x5xf32>
// CHECK-NEXT:   %[[S3:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[S2]]) -> (tensor<6x6x3x3x2x5xf32>) {
// CHECK-NEXT:     %[[S7:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<6x6x3x3x2x5xf32>) {
// CHECK-NEXT:       %[[S8:.*]] = scf.for %[[ARG7:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.*]] = %[[ARG6]]) -> (tensor<6x6x3x3x2x5xf32>) {
// CHECK-NEXT:         %[[S9:.*]] = scf.for %[[ARG9:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG10:.*]] = %[[ARG8]]) -> (tensor<6x6x3x3x2x5xf32>) {
// CHECK-NEXT:           %[[S10:.*]] = affine.apply #[[$MAP0]](%[[ARG3]])
// CHECK-NEXT:           %[[S11:.*]] = affine.apply #[[$MAP0]](%[[ARG5]])
// CHECK-NEXT:           %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[PADDED]][%[[ARG7]], %[[S10]], %[[S11]], %[[ARG9]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<2x14x14x5xf32> to tensor<6x6xf32>
// CHECK-NEXT:           %[[S13:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:           %[[S14:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S13]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:           %[[S15:.*]] = linalg.matmul ins(%[[CST_3]], %[[EXTRACTED_SLICE_9]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S14]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:           %[[S16:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:           %[[S17:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S16]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:           %[[S18:.*]] = linalg.matmul ins(%[[S15]], %[[CST_2]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S17]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S18]] into %[[ARG10]][0, 0, %[[ARG3]], %[[ARG5]], %[[ARG7]], %[[ARG9]]] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6xf32> into tensor<6x6x3x3x2x5xf32>
// CHECK-NEXT:           scf.yield %[[INSERTED_SLICE]] : tensor<6x6x3x3x2x5xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S9]] : tensor<6x6x3x3x2x5xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[S8]] : tensor<6x6x3x3x2x5xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S7]] : tensor<6x6x3x3x2x5xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S1]] {{\[}}[0, 1], [2], [3]] : tensor<6x6x5x2xf32> into tensor<36x5x2xf32>
// CHECK-NEXT:   %[[COLLAPSED_7:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1], [2, 3, 4], [5]] : tensor<6x6x3x3x2x5xf32> into tensor<36x18x5xf32>
// CHECK-NEXT:   %[[S4:.*]] = tensor.empty() : tensor<36x18x2xf32>
// CHECK-NEXT:   %[[S5:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S4]] : tensor<36x18x2xf32>) -> tensor<36x18x2xf32>
// CHECK-NEXT:   %[[S6:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_7]], %[[COLLAPSED]] : tensor<36x18x5xf32>, tensor<36x5x2xf32>) outs(%[[S5]] : tensor<36x18x2xf32>) -> tensor<36x18x2xf32>
// CHECK-NEXT:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S6]] {{\[}}[0, 1], [2, 3, 4], [5]] output_shape [6, 6, 3, 3, 2, 2] : tensor<36x18x2xf32> into tensor<6x6x3x3x2x2xf32>
// CHECK-NEXT:   %[[PADDED_8:.*]] = tensor.pad %[[ARG2]] low[0, 0, 0, 0] high[0, 3, 3, 0] {
// CHECK-NEXT:   ^bb0(%[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index, %[[ARG6:.*]]: index):
// CHECK-NEXT:     tensor.yield %[[CST_6]] : f32
// CHECK-NEXT:   } : tensor<2x9x9x2xf32> to tensor<2x12x12x2xf32>
// CHECK-NEXT:   %[[S6:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG4:.*]] = %[[PADDED_8]]) -> (tensor<2x12x12x2xf32>) {
// CHECK-NEXT:     %[[S7:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<2x12x12x2xf32>) {
// CHECK-NEXT:       %[[S8:.*]] = scf.for %[[ARG7:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.*]] = %[[ARG6]]) -> (tensor<2x12x12x2xf32>) {
// CHECK-NEXT:         %[[S9:.*]] = scf.for %[[ARG9:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG10:.*]] = %[[ARG8]]) -> (tensor<2x12x12x2xf32>) {
// CHECK-NEXT:           %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[EXPANDED]][0, 0, %[[ARG3]], %[[ARG5]], %[[ARG7]], %[[ARG9]]] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6x3x3x2x2xf32> to tensor<6x6xf32>
// CHECK-NEXT:           %[[S20:.*]] = affine.apply #[[$MAP0]](%[[ARG3]])
// CHECK-NEXT:           %[[S21:.*]] = affine.apply #[[$MAP0]](%[[ARG5]])
// CHECK-NEXT:           %[[S22:.*]] = tensor.extract_slice %[[ARG10]][%[[ARG7]], %[[S20]], %[[S21]], %[[ARG9]]] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<2x12x12x2xf32> to tensor<4x4xf32>
// CHECK-NEXT:           %[[S11:.*]] = tensor.empty() : tensor<4x6xf32>
// CHECK-NEXT:           %[[S12:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S11]] : tensor<4x6xf32>) -> tensor<4x6xf32>
// CHECK-NEXT:           %[[S13:.*]] = linalg.matmul ins(%[[CST_1]], %[[EXTRACTED_SLICE_9]] : tensor<4x6xf32>, tensor<6x6xf32>) outs(%[[S12]] : tensor<4x6xf32>) -> tensor<4x6xf32>
// CHECK-NEXT:           %[[S14:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:           %[[S15:.*]] = linalg.fill ins(%[[CST_6]] : f32) outs(%[[S14]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:           %[[S16:.*]] = linalg.matmul ins(%[[S13]], %[[CST_0]] : tensor<4x6xf32>, tensor<6x4xf32>) outs(%[[S15]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:           %[[S18:.*]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%[[CST]], %[[S16]] : f32, tensor<4x4xf32>) outs(%[[S22]] : tensor<4x4xf32>) {
// CHECK-NEXT:           ^bb0(%[[IN1:.*]]: f32, %[[IN2:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:             %[[VAL_98:.*]] = arith.mulf %[[IN1]], %[[IN2]] : f32
// CHECK-NEXT:             %[[VAL_99:.*]] = arith.addf %[[VAL_98]], %[[OUT]] : f32
// CHECK-NEXT:             linalg.yield %[[VAL_99]] : f32
// CHECK-NEXT:           } -> tensor<4x4xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S18]] into %[[ARG10]][%[[ARG7]], %[[S20]], %[[S21]], %[[ARG9]]] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<4x4xf32> into tensor<2x12x12x2xf32>
// CHECK-NEXT:           scf.yield %[[INSERTED_SLICE]] : tensor<2x12x12x2xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S9]] : tensor<2x12x12x2xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[S8]] : tensor<2x12x12x2xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S7]] : tensor<2x12x12x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[S6]][0, 0, 0, 0] [2, 9, 9, 2] [1, 1, 1, 1] : tensor<2x12x12x2xf32> to tensor<2x9x9x2xf32>
// CHECK-NEXT:   return %[[EXTRACTED_SLICE]] : tensor<2x9x9x2xf32>
// CHECK-NEXT: }
