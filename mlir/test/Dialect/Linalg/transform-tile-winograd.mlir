// RUN: mlir-opt %s -transform-interpreter --split-input-file | FileCheck %s

func.func @tile_winograd_filter(%arg0: tensor<2x3x3x5xf32>, %arg1: tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32> {
  %0 = linalg.winograd_filter_transform m(4) r(3) ins(%arg0 : tensor<2x3x3x5xf32>) outs(%arg1 : tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32>
  return %0 : tensor<6x6x5x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:4 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @tile_winograd_filter(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x3x3x5xf32>, %[[ARG1:.*]]: tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32> {
// CHECK:       %[[S0:.*]] = tensor.empty() : tensor<6x6x5x2xf32>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_2:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C3_3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_4:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_5:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_6:.*]] = arith.constant 1 : index
// CHECK:       %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG3:.*]] = %[[S0]]) -> (tensor<6x6x5x2xf32>) {
// CHECK-NEXT:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C3]] step %[[C1_4]] iter_args(%[[ARG5:.*]] = %[[ARG3]]) -> (tensor<6x6x5x2xf32>) {
// CHECK-NEXT:       %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_1]] to %[[C3_3]] step %[[C1_5]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<6x6x5x2xf32>) {
// CHECK-NEXT:         %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_2]] to %[[C5]] step %[[C1_6]] iter_args(%[[ARG9:.*]] = %[[ARG7]]) -> (tensor<6x6x5x2xf32>) {
// CHECK-NEXT:           %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0, 0, %[[ARG8]]] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<2x3x3x5xf32> to tensor<1x3x3x1xf32>
// CHECK-NEXT:           %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[ARG1]][0, 0, %[[ARG8]], %[[ARG2]]] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6x5x2xf32> to tensor<6x6x1x1xf32>
// CHECK-NEXT:           %[[S5:.*]] = linalg.winograd_filter_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<1x3x3x1xf32>) outs(%[[EXTRACTED_SLICE_7]] : tensor<6x6x1x1xf32>) -> tensor<6x6x1x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S5]] into %[[ARG9]][0, 0, %[[ARG8]], %[[ARG2]]] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6x1x1xf32> into tensor<6x6x5x2xf32>
// CHECK-NEXT:           scf.yield %[[INSERTED_SLICE]] : tensor<6x6x5x2xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S4]] : tensor<6x6x5x2xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[S3]] : tensor<6x6x5x2xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S2]] : tensor<6x6x5x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[S1]] : tensor<6x6x5x2xf32>

// -----

func.func @tile_winograd_filter(%arg0: tensor<2x3x3x5xf32>, %arg1: tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32> {
  %0 = linalg.winograd_filter_transform m(4) r(3) ins(%arg0 : tensor<2x3x3x5xf32>) outs(%arg1 : tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32>
  return %0 : tensor<6x6x5x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:4 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (-d0 + 5, 2)>

// CHECK-LABEL: func.func @tile_winograd_filter(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x3x3x5xf32>, %[[ARG1:.*]]: tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32> {
// CHECK:       %[[S0:.*]] = tensor.empty() : tensor<6x6x5x2xf32>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_2:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C3_3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_4:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_5:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2_6:.*]] = arith.constant 2 : index
// CHECK:       %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG3:.*]] = %[[S0]]) -> (tensor<6x6x5x2xf32>) {
// CHECK-NEXT:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C3]] step %[[C1_4]] iter_args(%[[ARG5:.*]] = %[[ARG3]]) -> (tensor<6x6x5x2xf32>) {
// CHECK-NEXT:       %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_1]] to %[[C3_3]] step %[[C1_5]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<6x6x5x2xf32>) {
// CHECK-NEXT:         %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_2]] to %[[C5]] step %[[C2_6]] iter_args(%[[ARG9:.*]] = %[[ARG7]]) -> (tensor<6x6x5x2xf32>) {
// CHECK-NEXT:           %[[C5_7:.*]] = arith.constant 5 : index
// CHECK-NEXT:           %[[S5:.*]] = affine.min #[[$MAP0]](%[[ARG8]])
// CHECK-NEXT:           %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0, 0, %[[ARG8]]] [1, 3, 3, %[[S5]]] [1, 1, 1, 1] : tensor<2x3x3x5xf32> to tensor<1x3x3x?xf32>
// CHECK-NEXT:           %[[EXTRACTED_SLICE_8:.*]] = tensor.extract_slice %[[ARG1]][0, 0, %[[ARG8]], %[[ARG2]]] [6, 6, %[[S5]], 1] [1, 1, 1, 1] : tensor<6x6x5x2xf32> to tensor<6x6x?x1xf32>
// CHECK-NEXT:           %[[S6:.*]] = linalg.winograd_filter_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<1x3x3x?xf32>) outs(%[[EXTRACTED_SLICE_8]] : tensor<6x6x?x1xf32>) -> tensor<6x6x?x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S6]] into %[[ARG9]][0, 0, %[[ARG8]], %[[ARG2]]] [6, 6, %[[S5]], 1] [1, 1, 1, 1] : tensor<6x6x?x1xf32> into tensor<6x6x5x2xf32>
// CHECK-NEXT:           scf.yield %[[INSERTED_SLICE]] : tensor<6x6x5x2xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S4]] : tensor<6x6x5x2xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[S3]] : tensor<6x6x5x2xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S2]] : tensor<6x6x5x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[S1]] : tensor<6x6x5x2xf32>

// -----

func.func @tile_winograd_filter(%arg0: tensor<2x3x1x5xf32>, %arg1: tensor<6x1x5x2xf32>) -> tensor<6x1x5x2xf32> {
  %0 = linalg.winograd_filter_transform m(4) r(3) ins(%arg0 : tensor<2x3x1x5xf32>) outs(%arg1 : tensor<6x1x5x2xf32>) -> tensor<6x1x5x2xf32>
  return %0 : tensor<6x1x5x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:4 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @tile_winograd_filter(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x3x1x5xf32>, %[[ARG1:.*]]: tensor<6x1x5x2xf32>) -> tensor<6x1x5x2xf32> {
// CHECK:       %[[S0:.*]] = tensor.empty() : tensor<6x1x5x2xf32>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_2:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1_3:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_4:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_5:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_6:.*]] = arith.constant 1 : index
// CHECK:       %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C2]] step %[[C1_3]] iter_args(%[[ARG3:.*]] = %[[S0]]) -> (tensor<6x1x5x2xf32>) {
// CHECK-NEXT:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C3]] step %[[C1_4]] iter_args(%[[ARG5:.*]] = %[[ARG3]]) -> (tensor<6x1x5x2xf32>) {
// CHECK-NEXT:       %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_1]] to %[[C1]] step %[[C1_5]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<6x1x5x2xf32>) {
// CHECK-NEXT:         %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_2]] to %[[C5]] step %[[C1_6]] iter_args(%[[ARG9:.*]] = %[[ARG7]]) -> (tensor<6x1x5x2xf32>) {
// CHECK-NEXT:           %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0, 0, %[[ARG8]]] [1, 3, 1, 1] [1, 1, 1, 1] : tensor<2x3x1x5xf32> to tensor<1x3x1x1xf32>
// CHECK-NEXT:           %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[ARG1]][0, 0, %[[ARG8]], %[[ARG2]]] [6, 1, 1, 1] [1, 1, 1, 1] : tensor<6x1x5x2xf32> to tensor<6x1x1x1xf32>
// CHECK-NEXT:           %[[S5:.*]] = linalg.winograd_filter_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<1x3x1x1xf32>) outs(%[[EXTRACTED_SLICE_7]] : tensor<6x1x1x1xf32>) -> tensor<6x1x1x1xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S5]] into %[[ARG9]][0, 0, %[[ARG8]], %[[ARG2]]] [6, 1, 1, 1] [1, 1, 1, 1] : tensor<6x1x1x1xf32> into tensor<6x1x5x2xf32>
// CHECK-NEXT:           scf.yield %[[INSERTED_SLICE]] : tensor<6x1x5x2xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S4]] : tensor<6x1x5x2xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[S3]] : tensor<6x1x5x2xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S2]] : tensor<6x1x5x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[S1]] : tensor<6x1x5x2xf32>

// -----

func.func @tile_winograd_input(%arg0: tensor<2x10x10x5xf32>, %arg1: tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32> {
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x10x10x5xf32>) outs(%arg1 : tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32>
  return %0 : tensor<6x6x2x2x2x5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop3:4 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP1:.+]] = affine_map<() -> (6)>

// CHECK-LABEL: func.func @tile_winograd_input(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x10x10x5xf32>, %[[ARG1:.*]]: tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32> {
// CHECK:       %[[S0:.*]] = tensor.empty() : tensor<6x6x2x2x2x5xf32>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_2:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C6_3:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_4:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_5:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_6:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_7:.*]] = arith.constant 1 : index
// CHECK:       %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C6]] step %[[C1]] iter_args(%[[ARG3:.*]] = %[[S0]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C6_3]] step %[[C1_5]] iter_args(%[[ARG5:.*]] = %[[ARG3]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:       %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_1]] to %[[C2]] step %[[C1_6]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:         %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_2]] to %[[C2_4]] step %[[C1_7]] iter_args(%[[ARG9:.*]] = %[[ARG7]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:           %[[S5:.*]] = affine.apply #[[$MAP0]](%[[ARG6]])
// CHECK-NEXT:           %[[S6:.*]] = affine.apply #[[$MAP0]](%[[ARG8]])
// CHECK-NEXT:           %[[S7:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:           %[[S8:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:           %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, %[[S5]], %[[S6]], 0] [2, %[[S7]], %[[S8]], 5] [1, 1, 1, 1] : tensor<2x10x10x5xf32> to tensor<2x?x?x5xf32>
// CHECK-NEXT:           %[[EXTRACTED_SLICE_8:.*]] = tensor.extract_slice %[[ARG1]][0, 0, %[[ARG6]], %[[ARG8]], 0, 0] [6, 6, 1, 1, 2, 5] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x2x5xf32> to tensor<6x6x1x1x2x5xf32>
// CHECK-NEXT:           %[[S9:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<2x?x?x5xf32>) outs(%[[EXTRACTED_SLICE_8]] : tensor<6x6x1x1x2x5xf32>) -> tensor<6x6x1x1x2x5xf32>
// CHECK-NEXT:           %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S9]] into %[[ARG9]][0, 0, %[[ARG6]], %[[ARG8]], 0, 0] [6, 6, 1, 1, 2, 5] [1, 1, 1, 1, 1, 1] : tensor<6x6x1x1x2x5xf32> into tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:           scf.yield %[[INSERTED_SLICE]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S4]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[S3]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S2]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[S1]] : tensor<6x6x2x2x2x5xf32>

// -----

func.func @tile_winograd_input(%arg0: tensor<2x10x10x5xf32>, %arg1: tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32> {
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x10x10x5xf32>) outs(%arg1 : tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32>
  return %0 : tensor<6x6x2x2x2x5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop3:6 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP1:.+]] = affine_map<() -> (6)>

// CHECK-LABEL: func.func @tile_winograd_input(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x10x10x5xf32>, %[[ARG1:.*]]: tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32> {
// CHECK:       %[[S0:.*]] = tensor.empty() : tensor<6x6x2x2x2x5xf32>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_2:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_3:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_4:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C6_5:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_6:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_7:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_8:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_9:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_10:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_11:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_12:.*]] = arith.constant 1 : index
// CHECK:       %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C6]] step %[[C1]] iter_args(%[[ARG3:.*]] = %[[S0]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C6_5]] step %[[C1_8]] iter_args(%[[ARG5:.*]] = %[[ARG3]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:       %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_1]] to %[[C2]] step %[[C1_9]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:         %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_2]] to %[[C2_6]] step %[[C1_10]] iter_args(%[[ARG9:.*]] = %[[ARG7]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:           %[[S5:.*]] = scf.for %[[ARG10:.*]] = %[[C0_3]] to %[[C2_7]] step %[[C1_11]] iter_args(%[[ARG11:.*]] = %[[ARG9]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:             %[[S6:.*]] = scf.for %[[ARG12:.*]] = %[[C0_4]] to %[[C5]] step %[[C1_12]] iter_args(%[[ARG13:.*]] = %[[ARG11]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:               %[[S7:.*]] = affine.apply #[[$MAP0]](%[[ARG6]])
// CHECK-NEXT:               %[[S8:.*]] = affine.apply #[[$MAP0]](%[[ARG8]])
// CHECK-NEXT:               %[[S9:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:               %[[S10:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:               %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG10]], %[[S7]], %[[S8]], %[[ARG12]]] [1, %[[S9]], %[[S10]], 1] [1, 1, 1, 1] : tensor<2x10x10x5xf32> to tensor<1x?x?x1xf32>
// CHECK-NEXT:               %[[EXTRACTED_SLICE_13:.*]] = tensor.extract_slice %[[ARG1]][0, 0, %[[ARG6]], %[[ARG8]], %[[ARG10]], %[[ARG12]]] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x2x5xf32> to tensor<6x6x1x1x1x1xf32>
// CHECK-NEXT:               %[[S11:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<1x?x?x1xf32>) outs(%[[EXTRACTED_SLICE_13]] : tensor<6x6x1x1x1x1xf32>) -> tensor<6x6x1x1x1x1xf32>
// CHECK-NEXT:               %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S11]] into %[[ARG13]][0, 0, %[[ARG6]], %[[ARG8]], %[[ARG10]], %[[ARG12]]] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6x1x1x1x1xf32> into tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:               scf.yield %[[INSERTED_SLICE]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %[[S6]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %[[S5]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S4]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[S3]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S2]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[S1]] : tensor<6x6x2x2x2x5xf32>

// -----

func.func @tile_winograd_input(%arg0: tensor<2x10x10x5xf32>, %arg1: tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32> {
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x10x10x5xf32>) outs(%arg1 : tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32>
  return %0 : tensor<6x6x2x2x2x5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop3:6 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 2, 2, 2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (-d0 + 5, 2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP2:.+]] = affine_map<() -> (10)>

// CHECK-LABEL: func.func @tile_winograd_input(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x10x10x5xf32>, %[[ARG1:.*]]: tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32> {
// CHECK:       %[[S0:.*]] = tensor.empty() : tensor<6x6x2x2x2x5xf32>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_2:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_3:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_4:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C6_5:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_6:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_7:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_8:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2_9:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_10:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_11:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_12:.*]] = arith.constant 2 : index
// CHECK:       %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C6]] step %[[C1]] iter_args(%[[ARG3:.*]] = %[[S0]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C6_5]] step %[[C1_8]] iter_args(%[[ARG5:.*]] = %[[ARG3]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:       %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_1]] to %[[C2]] step %[[C2_9]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:         %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_2]] to %[[C2_6]] step %[[C2_10]] iter_args(%[[ARG9:.*]] = %[[ARG7]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:           %[[S5:.*]] = scf.for %[[ARG10:.*]] = %[[C0_3]] to %[[C2_7]] step %[[C2_11]] iter_args(%[[ARG11:.*]] = %[[ARG9]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:             %[[S6:.*]] = scf.for %[[ARG12:.*]] = %[[C0_4]] to %[[C5]] step %[[C2_12]] iter_args(%[[ARG13:.*]] = %[[ARG11]]) -> (tensor<6x6x2x2x2x5xf32>) {
// CHECK-NEXT:               %[[C5_13:.*]] = arith.constant 5 : index
// CHECK-NEXT:               %[[S7:.*]] = affine.min #[[$MAP0]](%[[ARG12]])
// CHECK-NEXT:               %[[S8:.*]] = affine.apply #[[$MAP1]](%[[ARG6]])
// CHECK-NEXT:               %[[S9:.*]] = affine.apply #[[$MAP1]](%[[ARG8]])
// CHECK-NEXT:               %[[S10:.*]] = affine.apply #[[$MAP2]]()
// CHECK-NEXT:               %[[S11:.*]] = affine.apply #[[$MAP2]]()
// CHECK-NEXT:               %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG10]], %[[S8]], %[[S9]], %[[ARG12]]] [2, %[[S10]], %[[S11]], %[[S7]]] [1, 1, 1, 1] : tensor<2x10x10x5xf32> to tensor<2x?x?x?xf32>
// CHECK-NEXT:               %[[EXTRACTED_SLICE_14:.*]] = tensor.extract_slice %[[ARG1]][0, 0, %[[ARG6]], %[[ARG8]], %[[ARG10]], %[[ARG12]]] [6, 6, 2, 2, 2, %[[S7]]] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x2x5xf32> to tensor<6x6x2x2x2x?xf32>
// CHECK-NEXT:               %[[S12:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<2x?x?x?xf32>) outs(%[[EXTRACTED_SLICE_14]] : tensor<6x6x2x2x2x?xf32>) -> tensor<6x6x2x2x2x?xf32>
// CHECK-NEXT:               %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S12]] into %[[ARG13]][0, 0, %[[ARG6]], %[[ARG8]], %[[ARG10]], %[[ARG12]]] [6, 6, 2, 2, 2, %[[S7]]] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x2x?xf32> into tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:               scf.yield %[[INSERTED_SLICE]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %[[S6]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %[[S5]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S4]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[S3]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S2]] : tensor<6x6x2x2x2x5xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[S1]] : tensor<6x6x2x2x2x5xf32>

// -----

func.func @tile_winograd_input(%arg0: tensor<2x1x10x5xf32>, %arg1: tensor<1x6x1x2x2x5xf32>) -> tensor<1x6x1x2x2x5xf32> {
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x1x10x5xf32>) outs(%arg1 : tensor<1x6x1x2x2x5xf32>) -> tensor<1x6x1x2x2x5xf32>
  return %0 : tensor<1x6x1x2x2x5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop3:6 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP1:.+]] = affine_map<() -> (6)>

// CHECK-LABEL: func.func @tile_winograd_input(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x1x10x5xf32>, %[[ARG1:.*]]: tensor<1x6x1x2x2x5xf32>) -> tensor<1x6x1x2x2x5xf32> {
// CHECK:       %[[S0:.*]] = tensor.empty() : tensor<1x6x1x2x2x5xf32>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_2:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_3:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_4:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C1_5:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_6:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1_7:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_8:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_9:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_10:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_11:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_12:.*]] = arith.constant 1 : index
// CHECK:       %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C1]] step %[[C1_7]] iter_args(%[[ARG3:.*]] = %[[S0]]) -> (tensor<1x6x1x2x2x5xf32>) {
// CHECK-NEXT:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C6]] step %[[C1_8]] iter_args(%[[ARG5:.*]] = %[[ARG3]]) -> (tensor<1x6x1x2x2x5xf32>) {
// CHECK-NEXT:       %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_1]] to %[[C1_5]] step %[[C1_9]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<1x6x1x2x2x5xf32>) {
// CHECK-NEXT:         %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_2]] to %[[C2]] step %[[C1_10]] iter_args(%[[ARG9:.*]] = %[[ARG7]]) -> (tensor<1x6x1x2x2x5xf32>) {
// CHECK-NEXT:           %[[S5:.*]] = scf.for %[[ARG10:.*]] = %[[C0_3]] to %[[C2_6]] step %[[C1_11]] iter_args(%[[ARG11:.*]] = %[[ARG9]]) -> (tensor<1x6x1x2x2x5xf32>) {
// CHECK-NEXT:             %[[S6:.*]] = scf.for %[[ARG12:.*]] = %[[C0_4]] to %[[C5]] step %[[C1_12]] iter_args(%[[ARG13:.*]] = %[[ARG11]]) -> (tensor<1x6x1x2x2x5xf32>) {
// CHECK-NEXT:               %[[S7:.*]] = affine.apply #[[$MAP0]](%[[ARG6]])
// CHECK-NEXT:               %[[S8:.*]] = affine.apply #[[$MAP0]](%[[ARG8]])
// CHECK-NEXT:               %[[S9:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:               %[[S10:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:               %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG10]], 0, %[[S8]], %[[ARG12]]] [1, 1, %[[S10]], 1] [1, 1, 1, 1] : tensor<2x1x10x5xf32> to tensor<1x1x?x1xf32>
// CHECK-NEXT:               %[[EXTRACTED_SLICE_13:.*]] = tensor.extract_slice %[[ARG1]][0, 0, %[[ARG6]], %[[ARG8]], %[[ARG10]], %[[ARG12]]] [1, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x6x1x2x2x5xf32> to tensor<1x6x1x1x1x1xf32>
// CHECK-NEXT:               %[[S11:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<1x1x?x1xf32>) outs(%[[EXTRACTED_SLICE_13]] : tensor<1x6x1x1x1x1xf32>) -> tensor<1x6x1x1x1x1xf32>
// CHECK-NEXT:               %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S11]] into %[[ARG13]][0, 0, %[[ARG6]], %[[ARG8]], %[[ARG10]], %[[ARG12]]] [1, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x6x1x1x1x1xf32> into tensor<1x6x1x2x2x5xf32>
// CHECK-NEXT:               scf.yield %[[INSERTED_SLICE]] : tensor<1x6x1x2x2x5xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %[[S6]] : tensor<1x6x1x2x2x5xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %[[S5]] : tensor<1x6x1x2x2x5xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S4]] : tensor<1x6x1x2x2x5xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[S3]] : tensor<1x6x1x2x2x5xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S2]] : tensor<1x6x1x2x2x5xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[S1]] : tensor<1x6x1x2x2x5xf32>

// -----

func.func @tile_winograd_output(%arg0 : tensor<6x6x2x2x2x2xf32>, %arg1: tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32> {
  %0 = linalg.winograd_output_transform m(4) r(3) ins(%arg0 : tensor<6x6x2x2x2x2xf32>) outs(%arg1 : tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:4 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP1:.+]] = affine_map<() -> (4)>

// CHECK-LABEL: func.func @tile_winograd_output(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<6x6x2x2x2x2xf32>, %[[ARG1:.*]]: tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32> {
// CHECK:       %[[S0:.*]] = tensor.empty() : tensor<2x8x8x2xf32>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_2:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C6_3:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_4:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_5:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_6:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_7:.*]] = arith.constant 1 : index
// CHECK:       %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C6]] step %[[C1]] iter_args(%[[ARG3:.*]] = %[[S0]]) -> (tensor<2x8x8x2xf32>) {
// CHECK-NEXT:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C6_3]] step %[[C1_5]] iter_args(%[[ARG5:.*]] = %[[ARG3]]) -> (tensor<2x8x8x2xf32>) {
// CHECK-NEXT:       %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_1]] to %[[C2]] step %[[C1_6]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<2x8x8x2xf32>) {
// CHECK-NEXT:         %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_2]] to %[[C2_4]] step %[[C1_7]] iter_args(%[[ARG9:.*]] = %[[ARG7]]) -> (tensor<2x8x8x2xf32>) {
// CHECK-NEXT:           %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG6]], %[[ARG8]], 0, 0] [6, 6, 1, 1, 2, 2] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x2x2xf32> to tensor<6x6x1x1x2x2xf32>
// CHECK-NEXT:           %[[S5:.*]] = affine.apply #[[$MAP0]](%[[ARG6]])
// CHECK-NEXT:           %[[S6:.*]] = affine.apply #[[$MAP0]](%[[ARG8]])
// CHECK-NEXT:           %[[S7:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:           %[[S8:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:           %[[EXTRACTED_SLICE_8:.*]] = tensor.extract_slice %[[ARG1]][0, %[[S5]], %[[S6]], 0] [2, %[[S7]], %[[S8]], 2] [1, 1, 1, 1] : tensor<2x8x8x2xf32> to tensor<2x?x?x2xf32>
// CHECK-NEXT:           %[[S9:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<6x6x1x1x2x2xf32>) outs(%[[EXTRACTED_SLICE_8]] : tensor<2x?x?x2xf32>) -> tensor<2x?x?x2xf32>
// CHECK-NEXT:           %[[S10:.*]] = affine.apply #[[$MAP0]](%[[ARG6]])
// CHECK-NEXT:           %[[S11:.*]] = affine.apply #[[$MAP0]](%[[ARG8]])
// CHECK-NEXT:           %[[S12:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:           %[[S13:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:           %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S9]] into %[[ARG9]][0, %[[S10]], %[[S11]], 0] [2, %[[S12]], %[[S13]], 2] [1, 1, 1, 1] : tensor<2x?x?x2xf32> into tensor<2x8x8x2xf32>
// CHECK-NEXT:           scf.yield %[[INSERTED_SLICE]] : tensor<2x8x8x2xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S4]] : tensor<2x8x8x2xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[S3]] : tensor<2x8x8x2xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S2]] : tensor<2x8x8x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[S1]] : tensor<2x8x8x2xf32>

// -----

func.func @tile_winograd_output(%arg0 : tensor<6x6x2x2x3x5xf32>, %arg1: tensor<3x8x8x5xf32>) -> tensor<3x8x8x5xf32> {
  %0 = linalg.winograd_output_transform m(4) r(3) ins(%arg0 : tensor<6x6x2x2x3x5xf32>) outs(%arg1 : tensor<3x8x8x5xf32>) -> tensor<3x8x8x5xf32>
  return %0 : tensor<3x8x8x5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:6 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 2, 2, 2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (-d0 + 3, 2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 5, 2)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP3:.+]] = affine_map<() -> (8)>

// CHECK-LABEL: func.func @tile_winograd_output(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<6x6x2x2x3x5xf32>, %[[ARG1:.*]]: tensor<3x8x8x5xf32>) -> tensor<3x8x8x5xf32> {
// CHECK:       %[[S0:.*]] = tensor.empty() : tensor<3x8x8x5xf32>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_2:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_3:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_4:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C6_5:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_6:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_7:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2_8:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_9:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_10:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_11:.*]] = arith.constant 2 : index
// CHECK:       %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C6]] step %[[C1]] iter_args(%[[ARG3:.*]] = %[[S0]]) -> (tensor<3x8x8x5xf32>) {
// CHECK-NEXT:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C6_5]] step %[[C1_7]] iter_args(%[[ARG5:.*]] = %[[ARG3]]) -> (tensor<3x8x8x5xf32>) {
// CHECK-NEXT:       %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_1]] to %[[C2]] step %[[C2_8]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<3x8x8x5xf32>) {
// CHECK-NEXT:         %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_2]] to %[[C2_6]] step %[[C2_9]] iter_args(%[[ARG9:.*]] = %[[ARG7]]) -> (tensor<3x8x8x5xf32>) {
// CHECK-NEXT:           %[[S5:.*]] = scf.for %[[ARG10:.*]] = %[[C0_3]] to %[[C3]] step %[[C2_10]] iter_args(%[[ARG11:.*]] = %[[ARG9]]) -> (tensor<3x8x8x5xf32>) {
// CHECK-NEXT:             %[[S6:.*]] = scf.for %[[ARG12:.*]] = %[[C0_4]] to %[[C5]] step %[[C2_11]] iter_args(%[[ARG13:.*]] = %[[ARG11]]) -> (tensor<3x8x8x5xf32>) {
// CHECK-NEXT:               %[[C3_12:.*]] = arith.constant 3 : index
// CHECK-NEXT:               %[[S7:.*]] = affine.min #[[$MAP0]](%[[ARG10]])
// CHECK-NEXT:               %[[C5_13:.*]] = arith.constant 5 : index
// CHECK-NEXT:               %[[S8:.*]] = affine.min #[[$MAP1]](%[[ARG12]])
// CHECK-NEXT:               %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG6]], %[[ARG8]], %[[ARG10]], %[[ARG12]]] [6, 6, 2, 2, %[[S7]], %[[S8]]] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x3x5xf32> to tensor<6x6x2x2x?x?xf32>
// CHECK-NEXT:               %[[S9:.*]] = affine.apply #[[$MAP2]](%[[ARG6]])
// CHECK-NEXT:               %[[S10:.*]] = affine.apply #[[$MAP2]](%[[ARG8]])
// CHECK-NEXT:               %[[S11:.*]] = affine.apply #[[$MAP3]]()
// CHECK-NEXT:               %[[S12:.*]] = affine.apply #[[$MAP3]]()
// CHECK-NEXT:               %[[EXTRACTED_SLICE_14:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG10]], %[[S9]], %[[S10]], %[[ARG12]]] [%[[S7]], %[[S11]], %[[S12]], %[[S8]]] [1, 1, 1, 1] : tensor<3x8x8x5xf32> to tensor<?x?x?x?xf32>
// CHECK-NEXT:               %[[S13:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<6x6x2x2x?x?xf32>) outs(%[[EXTRACTED_SLICE_14]] : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK-NEXT:               %[[S14:.*]] = affine.apply #[[$MAP2]](%[[ARG6]])
// CHECK-NEXT:               %[[S15:.*]] = affine.apply #[[$MAP2]](%[[ARG8]])
// CHECK-NEXT:               %[[S16:.*]] = affine.apply #[[$MAP3]]()
// CHECK-NEXT:               %[[S17:.*]] = affine.apply #[[$MAP3]]()
// CHECK-NEXT:               %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S13]] into %[[ARG13]][%[[ARG10]], %[[S14]], %[[S15]], %[[ARG12]]] [%[[S7]], %[[S16]], %[[S17]], %[[S8]]] [1, 1, 1, 1] : tensor<?x?x?x?xf32> into tensor<3x8x8x5xf32>
// CHECK-NEXT:               scf.yield %[[INSERTED_SLICE]] : tensor<3x8x8x5xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %[[S6]] : tensor<3x8x8x5xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %[[S5]] : tensor<3x8x8x5xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S4]] : tensor<3x8x8x5xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[S3]] : tensor<3x8x8x5xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S2]] : tensor<3x8x8x5xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[S1]] : tensor<3x8x8x5xf32>
// CHECK-NEXT: }

// -----

func.func @tile_winograd_output(%arg0 : tensor<6x1x2x1x3x5xf32>, %arg1: tensor<3x8x1x5xf32>) -> tensor<3x8x1x5xf32> {
  %0 = linalg.winograd_output_transform m(4) r(3) ins(%arg0 : tensor<6x1x2x1x3x5xf32>) outs(%arg1 : tensor<3x8x1x5xf32>) -> tensor<3x8x1x5xf32>
  return %0 : tensor<3x8x1x5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:6 = transform.structured.tile_using_for %0 tile_sizes [1, 1, 1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP1:.+]] = affine_map<() -> (4)>

// CHECK-LABEL: func.func @tile_winograd_output(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<6x1x2x1x3x5xf32>, %[[ARG1:.*]]: tensor<3x8x1x5xf32>) -> tensor<3x8x1x5xf32> {
// CHECK:       %[[S0:.*]] = tensor.empty() : tensor<3x8x1x5xf32>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_2:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_3:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_4:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1_5:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1_6:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_7:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_8:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_9:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_10:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_11:.*]] = arith.constant 1 : index
// CHECK:       %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C6]] step %[[C1_6]] iter_args(%[[ARG3:.*]] = %[[S0]]) -> (tensor<3x8x1x5xf32>) {
// CHECK-NEXT:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C1]] step %[[C1_7]] iter_args(%[[ARG5:.*]] = %[[ARG3]]) -> (tensor<3x8x1x5xf32>) {
// CHECK-NEXT:       %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_1]] to %[[C2]] step %[[C1_8]] iter_args(%[[ARG7:.*]] = %[[ARG5]]) -> (tensor<3x8x1x5xf32>) {
// CHECK-NEXT:         %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_2]] to %[[C1_5]] step %[[C1_9]] iter_args(%[[ARG9:.*]] = %[[ARG7]]) -> (tensor<3x8x1x5xf32>) {
// CHECK-NEXT:           %[[S5:.*]] = scf.for %[[ARG10:.*]] = %[[C0_3]] to %[[C3]] step %[[C1_10]] iter_args(%[[ARG11:.*]] = %[[ARG9]]) -> (tensor<3x8x1x5xf32>) {
// CHECK-NEXT:             %[[S6:.*]] = scf.for %[[ARG12:.*]] = %[[C0_4]] to %[[C5]] step %[[C1_11]] iter_args(%[[ARG13:.*]] = %[[ARG11]]) -> (tensor<3x8x1x5xf32>) {
// CHECK-NEXT:               %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG6]], %[[ARG8]], %[[ARG10]], %[[ARG12]]] [6, 1, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x1x2x1x3x5xf32> to tensor<6x1x1x1x1x1xf32>
// CHECK-NEXT:               %[[S7:.*]] = affine.apply #[[$MAP0]](%[[ARG6]])
// CHECK-NEXT:               %[[S8:.*]] = affine.apply #[[$MAP0]](%[[ARG8]])
// CHECK-NEXT:               %[[S9:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:               %[[S10:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:               %[[EXTRACTED_SLICE_12:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG10]], %[[S7]], 0, %[[ARG12]]] [1, %[[S9]], 1, 1] [1, 1, 1, 1] : tensor<3x8x1x5xf32> to tensor<1x?x1x1xf32>
// CHECK-NEXT:               %[[S11:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<6x1x1x1x1x1xf32>) outs(%[[EXTRACTED_SLICE_12]] : tensor<1x?x1x1xf32>) -> tensor<1x?x1x1xf32>
// CHECK-NEXT:               %[[S12:.*]] = affine.apply #[[$MAP0]](%[[ARG6]])
// CHECK-NEXT:               %[[S13:.*]] = affine.apply #[[$MAP0]](%[[ARG8]])
// CHECK-NEXT:               %[[S14:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:               %[[S15:.*]] = affine.apply #[[$MAP1]]()
// CHECK-NEXT:               %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S11]] into %[[ARG13]][%[[ARG10]], %[[S12]], 0, %[[ARG12]]] [1, %[[S14]], 1, 1] [1, 1, 1, 1] : tensor<1x?x1x1xf32> into tensor<3x8x1x5xf32>
// CHECK-NEXT:               scf.yield %[[INSERTED_SLICE]] : tensor<3x8x1x5xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %[[S6]] : tensor<3x8x1x5xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %[[S5]] : tensor<3x8x1x5xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[S4]] : tensor<3x8x1x5xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[S3]] : tensor<3x8x1x5xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[S2]] : tensor<3x8x1x5xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[S1]] : tensor<3x8x1x5xf32>
