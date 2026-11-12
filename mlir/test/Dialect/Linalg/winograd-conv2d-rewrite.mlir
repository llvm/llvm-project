// RUN: mlir-opt %s -split-input-file -test-linalg-transform-patterns=test-decompose-winograd-ops | FileCheck %s

func.func @conv2d(%arg0: tensor<2x11x11x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %2 = tensor.empty() : tensor<6x6x5x2xf32>
  %3 = linalg.winograd_filter_transform fmr(F_4_3) ins(%arg1 : tensor<2x3x3x5xf32>) outs(%2 : tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32>
  %padded = tensor.pad %arg0 low[0, 0, 0, 0] high[0, 3, 3, 0] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %cst : f32
  } : tensor<2x11x11x5xf32> to tensor<2x14x14x5xf32>
  %4 = tensor.empty() : tensor<6x6x3x3x2x5xf32>
  %5 = linalg.winograd_input_transform fmr(F_4_3) ins(%padded : tensor<2x14x14x5xf32>) outs(%4 : tensor<6x6x3x3x2x5xf32>) -> tensor<6x6x3x3x2x5xf32>
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
  %9 = linalg.winograd_output_transform fmr(F_4_3) ins(%expanded : tensor<6x6x3x3x2x2xf32>) outs(%padded_1 : tensor<2x12x12x2xf32>) -> tensor<2x12x12x2xf32>
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

// -----

func.func @conv2d_type_promotion(%arg0: tensor<2x6x6x5xf16>, %arg1: tensor<2x3x3x5xf16>, %arg2: tensor<1xf32>, %arg3: tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<6x6x5x2xf16>
  %1 = linalg.winograd_filter_transform fmr(F_4_3) ins(%arg1 : tensor<2x3x3x5xf16>) outs(%0 : tensor<6x6x5x2xf16>) -> tensor<6x6x5x2xf16> // no-crash
  %2 = tensor.empty() : tensor<6x6x1x1x2x5xf16>
  %3 = linalg.winograd_input_transform fmr(F_4_3) ins(%arg0 : tensor<2x6x6x5xf16>) outs(%2 : tensor<6x6x1x1x2x5xf16>) -> tensor<6x6x1x1x2x5xf16> // no-crash
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<6x6x5x2xf16> into tensor<36x5x2xf16>
  %collapsed_0 = tensor.collapse_shape %3 [[0, 1], [2, 3, 4], [5]] : tensor<6x6x1x1x2x5xf16> into tensor<36x2x5xf16>
  %4 = tensor.empty() : tensor<36x2x2xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<36x2x2xf32>) -> tensor<36x2x2xf32>
  %6 = linalg.batch_matmul ins(%collapsed_0, %collapsed : tensor<36x2x5xf16>, tensor<36x5x2xf16>) outs(%5 : tensor<36x2x2xf32>) -> tensor<36x2x2xf32>
  %expanded = tensor.expand_shape %6 [[0, 1], [2, 3, 4], [5]] output_shape [6, 6, 1, 1, 2, 2] : tensor<36x2x2xf32> into tensor<6x6x1x1x2x2xf32>
  %7 = linalg.winograd_output_transform fmr(F_4_3) ins(%expanded : tensor<6x6x1x1x2x2xf32>) outs(%arg3 : tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32>
  return %7 : tensor<2x4x4x2xf32>
}


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @conv2d_type_promotion(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x6x6x5xf16>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<2x3x3x5xf16>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<1xf32>,
// CHECK-SAME:      %[[ARG3:.*]]: tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32> {
// CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 1.024000e+03 : f32
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant dense<{{\[\[}}1.250000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00], [2.500000e-01, -2.500000e-01, 2.500000e-01, -2.500000e-01], [2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01], [1.250000e-01, -2.500000e-01, 5.000000e-01, -1.000000e+00], [1.250000e-01, 2.500000e-01, 5.000000e-01, 1.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 5.000000e-01]]> : tensor<6x4xf32>
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant dense<{{\[\[}}1.250000e-01, 2.500000e-01, 2.500000e-01, 1.250000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -2.500000e-01, 2.500000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 2.500000e-01, 5.000000e-01, 5.000000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -1.000000e+00, 1.000000e+00, 5.000000e-01]]> : tensor<4x6xf32>
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant dense<{{\[\[}}2.500000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 2.500000e-01, -2.500000e-01, 2.500000e-01, -2.500000e-01, 2.500000e-01], [-3.125000e-01, -2.500000e-01, -2.500000e-01, -1.250000e-01, -1.250000e-01, 0.000000e+00], [0.000000e+00, -6.250000e-02, 6.250000e-02, -2.500000e-01, 2.500000e-01, -3.125000e-01], [6.250000e-02, 6.250000e-02, 6.250000e-02, 1.250000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 6.250000e-02]]> : tensor<6x6xf16>
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant dense<{{\[\[}}2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -2.500000e-01, -6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, -2.500000e-01, -2.500000e-01, 6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -1.250000e-01, -2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, -1.250000e-01, 2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02]]> : tensor<6x6xf16>
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant dense<{{\[\[}}1.000000e+00, -3.332520e-01, -3.332520e-01, 8.331300e-02, 8.331300e-02, 0.000000e+00], [0.000000e+00, 3.332520e-01, -3.332520e-01, -1.666260e-01, 1.666260e-01, 0.000000e+00], [0.000000e+00, -3.332520e-01, -3.332520e-01, 3.332520e-01, 3.332520e-01, 1.000000e+00]]> : tensor<3x6xf16>
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00, 0.000000e+00], [-3.332520e-01, 3.332520e-01, -3.332520e-01], [-3.332520e-01, -3.332520e-01, -3.332520e-01], [8.331300e-02, -1.666260e-01, 3.332520e-01], [8.331300e-02, 1.666260e-01, 3.332520e-01], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<6x3xf16>
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:           %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_9:.*]] = arith.constant 5 : index
// CHECK-DAG:           %[[VAL_10:.*]] = arith.constant 2 : index
// CHECK-DAG:           %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_12:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_13:.*]] = tensor.empty() : tensor<6x6x5x2xf16>
// CHECK-NEXT:           %[[VAL_14:.*]] = scf.for %[[VAL_15:.*]] = %[[VAL_11]] to %[[VAL_10]] step %[[VAL_8]] iter_args(%[[VAL_16:.*]] = %[[VAL_13]]) -> (tensor<6x6x5x2xf16>) {
// CHECK-NEXT:             %[[VAL_17:.*]] = scf.for %[[VAL_18:.*]] = %[[VAL_11]] to %[[VAL_9]] step %[[VAL_8]] iter_args(%[[VAL_19:.*]] = %[[VAL_16]]) -> (tensor<6x6x5x2xf16>) {
// CHECK-NEXT:               %[[VAL_20:.*]] = tensor.extract_slice %[[ARG1]]{{\[}}%[[VAL_15]], %[[VAL_11]], %[[VAL_11]], %[[VAL_18]]] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<2x3x3x5xf16> to tensor<3x3xf16>
// CHECK-NEXT:               %[[VAL_21:.*]] = tensor.empty() : tensor<6x3xf16>
// CHECK-NEXT:               %[[VAL_22:.*]] = linalg.fill ins(%[[VAL_7]] : f16) outs(%[[VAL_21]] : tensor<6x3xf16>) -> tensor<6x3xf16>
// CHECK-NEXT:               %[[VAL_23:.*]] = linalg.matmul ins(%[[VAL_6]], %[[VAL_20]] : tensor<6x3xf16>, tensor<3x3xf16>) outs(%[[VAL_22]] : tensor<6x3xf16>) -> tensor<6x3xf16>
// CHECK-NEXT:               %[[VAL_24:.*]] = tensor.empty() : tensor<6x6xf16>
// CHECK-NEXT:               %[[VAL_25:.*]] = linalg.fill ins(%[[VAL_7]] : f16) outs(%[[VAL_24]] : tensor<6x6xf16>) -> tensor<6x6xf16>
// CHECK-NEXT:               %[[VAL_26:.*]] = linalg.matmul ins(%[[VAL_23]], %[[VAL_5]] : tensor<6x3xf16>, tensor<3x6xf16>) outs(%[[VAL_25]] : tensor<6x6xf16>) -> tensor<6x6xf16>
// CHECK-NEXT:               %[[VAL_27:.*]] = tensor.insert_slice %[[VAL_26]] into %[[VAL_19]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_18]], %[[VAL_15]]] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6xf16> into tensor<6x6x5x2xf16>
// CHECK-NEXT:               scf.yield %[[VAL_27]] : tensor<6x6x5x2xf16>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %[[VAL_17]] : tensor<6x6x5x2xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:           %[[VAL_28:.*]] = tensor.empty() : tensor<6x6x1x1x2x5xf16>
// CHECK-NEXT:           %[[VAL_29:.*]] = scf.for %[[VAL_30:.*]] = %[[VAL_11]] to %[[VAL_8]] step %[[VAL_8]] iter_args(%[[VAL_31:.*]] = %[[VAL_28]]) -> (tensor<6x6x1x1x2x5xf16>) {
// CHECK-NEXT:             %[[VAL_32:.*]] = scf.for %[[VAL_33:.*]] = %[[VAL_11]] to %[[VAL_8]] step %[[VAL_8]] iter_args(%[[VAL_34:.*]] = %[[VAL_31]]) -> (tensor<6x6x1x1x2x5xf16>) {
// CHECK-NEXT:               %[[VAL_35:.*]] = scf.for %[[VAL_36:.*]] = %[[VAL_11]] to %[[VAL_10]] step %[[VAL_8]] iter_args(%[[VAL_37:.*]] = %[[VAL_34]]) -> (tensor<6x6x1x1x2x5xf16>) {
// CHECK-NEXT:                 %[[VAL_38:.*]] = scf.for %[[VAL_39:.*]] = %[[VAL_11]] to %[[VAL_9]] step %[[VAL_8]] iter_args(%[[VAL_40:.*]] = %[[VAL_37]]) -> (tensor<6x6x1x1x2x5xf16>) {
// CHECK-NEXT:                   %[[VAL_41:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_30]])
// CHECK-NEXT:                   %[[VAL_42:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_33]])
// CHECK-NEXT:                   %[[VAL_43:.*]] = tensor.extract_slice %[[ARG0]]{{\[}}%[[VAL_36]], %[[VAL_41]], %[[VAL_42]], %[[VAL_39]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<2x6x6x5xf16> to tensor<6x6xf16>
// CHECK-NEXT:                   %[[VAL_44:.*]] = tensor.empty() : tensor<6x6xf16>
// CHECK-NEXT:                   %[[VAL_45:.*]] = linalg.fill ins(%[[VAL_7]] : f16) outs(%[[VAL_44]] : tensor<6x6xf16>) -> tensor<6x6xf16>
// CHECK-NEXT:                   %[[VAL_46:.*]] = linalg.matmul ins(%[[VAL_4]], %[[VAL_43]] : tensor<6x6xf16>, tensor<6x6xf16>) outs(%[[VAL_45]] : tensor<6x6xf16>) -> tensor<6x6xf16>
// CHECK-NEXT:                   %[[VAL_47:.*]] = tensor.empty() : tensor<6x6xf16>
// CHECK-NEXT:                   %[[VAL_48:.*]] = linalg.fill ins(%[[VAL_7]] : f16) outs(%[[VAL_47]] : tensor<6x6xf16>) -> tensor<6x6xf16>
// CHECK-NEXT:                   %[[VAL_49:.*]] = linalg.matmul ins(%[[VAL_46]], %[[VAL_3]] : tensor<6x6xf16>, tensor<6x6xf16>) outs(%[[VAL_48]] : tensor<6x6xf16>) -> tensor<6x6xf16>
// CHECK-NEXT:                   %[[VAL_50:.*]] = tensor.insert_slice %[[VAL_49]] into %[[VAL_40]][0, 0, %[[VAL_30]], %[[VAL_33]], %[[VAL_36]], %[[VAL_39]]] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6xf16> into tensor<6x6x1x1x2x5xf16>
// CHECK-NEXT:                   scf.yield %[[VAL_50]] : tensor<6x6x1x1x2x5xf16>
// CHECK-NEXT:                 }
// CHECK-NEXT:                 scf.yield %[[VAL_38]] : tensor<6x6x1x1x2x5xf16>
// CHECK-NEXT:               }
// CHECK-NEXT:               scf.yield %[[VAL_35]] : tensor<6x6x1x1x2x5xf16>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %[[VAL_32]] : tensor<6x6x1x1x2x5xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:           %[[VAL_51:.*]] = tensor.collapse_shape %[[VAL_14]] {{\[\[}}0, 1], [2], [3]] : tensor<6x6x5x2xf16> into tensor<36x5x2xf16>
// CHECK-NEXT:           %[[VAL_52:.*]] = tensor.collapse_shape %[[VAL_29]] {{\[\[}}0, 1], [2, 3, 4], [5]] : tensor<6x6x1x1x2x5xf16> into tensor<36x2x5xf16>
// CHECK-NEXT:           %[[VAL_53:.*]] = tensor.empty() : tensor<36x2x2xf32>
// CHECK-NEXT:           %[[VAL_54:.*]] = linalg.fill ins(%[[VAL_12]] : f32) outs(%[[VAL_53]] : tensor<36x2x2xf32>) -> tensor<36x2x2xf32>
// CHECK-NEXT:           %[[VAL_55:.*]] = linalg.batch_matmul ins(%[[VAL_52]], %[[VAL_51]] : tensor<36x2x5xf16>, tensor<36x5x2xf16>) outs(%[[VAL_54]] : tensor<36x2x2xf32>) -> tensor<36x2x2xf32>
// CHECK-NEXT:           %[[VAL_56:.*]] = tensor.expand_shape %[[VAL_55]] {{\[\[}}0, 1], [2, 3, 4], [5]] output_shape [6, 6, 1, 1, 2, 2] : tensor<36x2x2xf32> into tensor<6x6x1x1x2x2xf32>
// CHECK-NEXT:           %[[VAL_57:.*]] = scf.for %[[VAL_58:.*]] = %[[VAL_11]] to %[[VAL_8]] step %[[VAL_8]] iter_args(%[[VAL_59:.*]] = %[[ARG3]]) -> (tensor<2x4x4x2xf32>) {
// CHECK-NEXT:             %[[VAL_60:.*]] = scf.for %[[VAL_61:.*]] = %[[VAL_11]] to %[[VAL_8]] step %[[VAL_8]] iter_args(%[[VAL_62:.*]] = %[[VAL_59]]) -> (tensor<2x4x4x2xf32>) {
// CHECK-NEXT:               %[[VAL_63:.*]] = scf.for %[[VAL_64:.*]] = %[[VAL_11]] to %[[VAL_10]] step %[[VAL_8]] iter_args(%[[VAL_65:.*]] = %[[VAL_62]]) -> (tensor<2x4x4x2xf32>) {
// CHECK-NEXT:                 %[[VAL_66:.*]] = scf.for %[[VAL_67:.*]] = %[[VAL_11]] to %[[VAL_10]] step %[[VAL_8]] iter_args(%[[VAL_68:.*]] = %[[VAL_65]]) -> (tensor<2x4x4x2xf32>) {
// CHECK-NEXT:                   %[[VAL_69:.*]] = tensor.extract_slice %[[VAL_56]][0, 0, %[[VAL_58]], %[[VAL_61]], %[[VAL_64]], %[[VAL_67]]] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6x1x1x2x2xf32> to tensor<6x6xf32>
// CHECK-NEXT:                   %[[VAL_70:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_58]])
// CHECK-NEXT:                   %[[VAL_71:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_61]])
// CHECK-NEXT:                   %[[VAL_72:.*]] = tensor.extract_slice %[[VAL_68]]{{\[}}%[[VAL_64]], %[[VAL_70]], %[[VAL_71]], %[[VAL_67]]] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<2x4x4x2xf32> to tensor<4x4xf32>
// CHECK-NEXT:                   %[[VAL_73:.*]] = tensor.empty() : tensor<4x6xf32>
// CHECK-NEXT:                   %[[VAL_74:.*]] = linalg.fill ins(%[[VAL_12]] : f32) outs(%[[VAL_73]] : tensor<4x6xf32>) -> tensor<4x6xf32>
// CHECK-NEXT:                   %[[VAL_75:.*]] = linalg.matmul ins(%[[VAL_2]], %[[VAL_69]] : tensor<4x6xf32>, tensor<6x6xf32>) outs(%[[VAL_74]] : tensor<4x6xf32>) -> tensor<4x6xf32>
// CHECK-NEXT:                   %[[VAL_76:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:                   %[[VAL_77:.*]] = linalg.fill ins(%[[VAL_12]] : f32) outs(%[[VAL_76]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:                   %[[VAL_78:.*]] = linalg.matmul ins(%[[VAL_75]], %[[VAL_1]] : tensor<4x6xf32>, tensor<6x4xf32>) outs(%[[VAL_77]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:                   %[[VAL_79:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_78]] : f32, tensor<4x4xf32>) outs(%[[VAL_72]] : tensor<4x4xf32>) {
// CHECK-NEXT:                   ^bb0(%[[VAL_80:.*]]: f32, %[[VAL_81:.*]]: f32, %[[VAL_82:.*]]: f32):
// CHECK-NEXT:                     %[[VAL_83:.*]] = arith.mulf %[[VAL_80]], %[[VAL_81]] : f32
// CHECK-NEXT:                     %[[VAL_84:.*]] = arith.addf %[[VAL_83]], %[[VAL_82]] : f32
// CHECK-NEXT:                     linalg.yield %[[VAL_84]] : f32
// CHECK-NEXT:                   } -> tensor<4x4xf32>
// CHECK-NEXT:                   %[[VAL_85:.*]] = tensor.insert_slice %[[VAL_79]] into %[[VAL_68]]{{\[}}%[[VAL_64]], %[[VAL_70]], %[[VAL_71]], %[[VAL_67]]] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<4x4xf32> into tensor<2x4x4x2xf32>
// CHECK-NEXT:                   scf.yield %[[VAL_85]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:                 }
// CHECK-NEXT:                 scf.yield %[[VAL_66]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:               }
// CHECK-NEXT:               scf.yield %[[VAL_63]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %[[VAL_60]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           return %[[VAL_57]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:         }