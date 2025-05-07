// RUN: mlir-opt %s -transform-interpreter --split-input-file | FileCheck %s

func.func @tile_winograd_filter(%arg0: tensor<2x3x3x5xf32>, %arg1: tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32> {
  %0 = linalg.winograd_filter_transform m(4) r(3) ins(%arg0 : tensor<2x3x3x5xf32>) outs(%arg1 : tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32>
  return %0 : tensor<6x6x5x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:2 = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @tile_winograd_filter(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x3x3x5xf32>, %[[ARG1:.*]]: tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32> {
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:  %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:  %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:  %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG3:.*]] = %[[ARG1]])
// CHECK:    %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C5]] step %[[C1_1]] iter_args(%[[ARG5:.*]] = %[[ARG3]])
// CHECK:      %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0, 0, %[[ARG4]]] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<2x3x3x5xf32> to tensor<1x3x3x1xf32>
// CHECK:      %[[EXTRACTED_SLICE_2:.*]] = tensor.extract_slice %[[ARG5]][0, 0, %[[ARG4]], %[[ARG2]]] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6x5x2xf32> to tensor<6x6x1x1xf32>
// CHECK:      %[[S3:.*]] = linalg.winograd_filter_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<1x3x3x1xf32>) outs(%[[EXTRACTED_SLICE_2]] : tensor<6x6x1x1xf32>) -> tensor<6x6x1x1xf32>
// CHECK:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S3]] into %[[ARG5]][0, 0, %[[ARG4]], %[[ARG2]]] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6x1x1xf32> into tensor<6x6x5x2xf32>

// -----

func.func @tile_winograd_filter(%arg0: tensor<2x3x3x5xf32>, %arg1: tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32> {
  %0 = linalg.winograd_filter_transform m(4) r(3) ins(%arg0 : tensor<2x3x3x5xf32>) outs(%arg1 : tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32>
  return %0 : tensor<6x6x5x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:2 = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (-d0 + 5, 2)>
// CHECK-LABEL: func.func @tile_winograd_filter(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x3x3x5xf32>, %[[ARG1:.*]]: tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32> {
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2_1:.*]] = arith.constant 2 : index
// CHECK:   %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG3:.*]] = %[[ARG1]])
// CHECK:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C5]] step %[[C2_1]] iter_args(%[[ARG5:.*]] = %[[ARG3]])
// CHECK:       %[[C5_2:.*]] = arith.constant 5 : index
// CHECK:       %[[S3:.*]] = affine.min #[[$MAP0]](%[[ARG4]])
// CHECK:       %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0, 0, %[[ARG4]]] [1, 3, 3, %[[S3]]] [1, 1, 1, 1] : tensor<2x3x3x5xf32> to tensor<1x3x3x?xf32>
// CHECK:       %[[EXTRACTED_SLICE_3:.*]] = tensor.extract_slice %[[ARG5]][0, 0, %[[ARG4]], %[[ARG2]]] [6, 6, %[[S3]], 1] [1, 1, 1, 1] : tensor<6x6x5x2xf32> to tensor<6x6x?x1xf32>
// CHECK:       %[[S4:.*]] = linalg.winograd_filter_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<1x3x3x?xf32>) outs(%[[EXTRACTED_SLICE_3]] : tensor<6x6x?x1xf32>) -> tensor<6x6x?x1xf32>
// CHECK:      %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S4]] into %[[ARG5]][0, 0, %[[ARG4]], %[[ARG2]]] [6, 6, %[[S3]], 1] [1, 1, 1, 1] : tensor<6x6x?x1xf32> into tensor<6x6x5x2xf32>
// -----

func.func @tile_winograd_filter(%arg0: tensor<2x3x1x5xf32>, %arg1: tensor<6x1x5x2xf32>) -> tensor<6x1x5x2xf32> {
  %0 = linalg.winograd_filter_transform m(4) r(3) ins(%arg0 : tensor<2x3x1x5xf32>) outs(%arg1 : tensor<6x1x5x2xf32>) -> tensor<6x1x5x2xf32>
  return %0 : tensor<6x1x5x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_filter_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:2 = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @tile_winograd_filter(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x3x1x5xf32>, %[[ARG1:.*]]: tensor<6x1x5x2xf32>) -> tensor<6x1x5x2xf32> {
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:   %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG3:.*]] = %[[ARG1]])
// CHECK:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C5]] step %[[C1_1]] iter_args(%[[ARG5:.*]] = %[[ARG3]])
// CHECK:       %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0, 0, %[[ARG4]]] [1, 3, 1, 1] [1, 1, 1, 1] : tensor<2x3x1x5xf32> to tensor<1x3x1x1xf32>
// CHECK:       %[[EXTRACTED_SLICE_2:.*]] = tensor.extract_slice %[[ARG5]][0, 0, %[[ARG4]], %[[ARG2]]] [6, 1, 1, 1] [1, 1, 1, 1] : tensor<6x1x5x2xf32> to tensor<6x1x1x1xf32>
// CHECK:       %[[S3:.*]] = linalg.winograd_filter_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<1x3x1x1xf32>) outs(%[[EXTRACTED_SLICE_2]] : tensor<6x1x1x1xf32>) -> tensor<6x1x1x1xf32>
// CHECK:       %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S3]] into %[[ARG5]][0, 0, %[[ARG4]], %[[ARG2]]] [6, 1, 1, 1] [1, 1, 1, 1] : tensor<6x1x1x1xf32> into tensor<6x1x5x2xf32>

// -----

func.func @tile_winograd_input(%arg0: tensor<2x10x10x5xf32>, %arg1: tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32> {
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x10x10x5xf32>) outs(%arg1 : tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32>
  return %0 : tensor<6x6x2x2x2x5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop3:2 = transform.structured.tile_using_for %0 tile_sizes [0, 0, 1, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP1:.+]] = affine_map<() -> (6)>
// CHECK-LABEL: func.func @tile_winograd_input(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x10x10x5xf32>, %[[ARG1:.*]]: tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32> {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C2_1:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C1_2:.*]] = arith.constant 1 : index
// CHECK: %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG3:.*]] = %[[ARG1]])
// CHECK: %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C2_1]] step %[[C1_2]] iter_args(%[[ARG5:.*]] = %[[ARG3]])
// CHECK:   %[[S3:.*]] = affine.apply #[[$MAP0]](%[[ARG2]])
// CHECK:   %[[S4:.*]] = affine.apply #[[$MAP0]](%[[ARG4]])
// CHECK:   %[[S5:.*]] = affine.apply #[[$MAP1]]()
// CHECK:   %[[S6:.*]] = affine.apply #[[$MAP1]]()
// CHECK:   %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, %[[S3]], %[[S4]], 0] [2, %[[S5]], %[[S6]], 5] [1, 1, 1, 1] : tensor<2x10x10x5xf32> to tensor<2x?x?x5xf32>
// CHECK:   %[[EXTRACTED_SLICE_5:.*]] = tensor.extract_slice %[[ARG5]][0, 0, %[[ARG2]], %[[ARG4]], 0, 0] [6, 6, 1, 1, 2, 5] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x2x5xf32> to tensor<6x6x1x1x2x5xf32>
// CHECK:   %[[S7:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<2x?x?x5xf32>) outs(%[[EXTRACTED_SLICE_5]] : tensor<6x6x1x1x2x5xf32>) -> tensor<6x6x1x1x2x5xf32>
// CHECK:   %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S7]] into %[[ARG5]][0, 0, %[[ARG2]], %[[ARG4]], 0, 0] [6, 6, 1, 1, 2, 5] [1, 1, 1, 1, 1, 1] : tensor<6x6x1x1x2x5xf32> into tensor<6x6x2x2x2x5xf32>

// -----

func.func @tile_winograd_input(%arg0: tensor<2x10x10x5xf32>, %arg1: tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32> {
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x10x10x5xf32>) outs(%arg1 : tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32>
  return %0 : tensor<6x6x2x2x2x5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop3:4 = transform.structured.tile_using_for %0 tile_sizes [0, 0, 1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP1:.+]] = affine_map<() -> (6)>
// CHECK-LABEL: func.func @tile_winograd_input(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x10x10x5xf32>, %[[ARG1:.*]]: tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32> {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C0_3:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C0_6:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C2_1:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C2_4:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C1_2:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C1_5:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C1_7:.*]] = arith.constant 1 : index
// CHECK: %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG3:.*]] = %[[ARG1]])
// CHECK:   %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C2_1]] step %[[C1_2]] iter_args(%[[ARG5:.*]] = %[[ARG3]])
// CHECK:     %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_3]] to %[[C2_4]] step %[[C1_5]] iter_args(%[[ARG7:.*]] = %[[ARG5]])
// CHECK:       %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_6]] to %[[C5]] step %[[C1_7]] iter_args(%[[ARG9:.*]] = %[[ARG7]])
// CHECK:         %[[S5:.*]] = affine.apply #[[$MAP0]](%[[ARG2]])
// CHECK:         %[[S6:.*]] = affine.apply #[[$MAP0]](%[[ARG4]])
// CHECK:         %[[S7:.*]] = affine.apply #[[$MAP1]]()
// CHECK:         %[[S8:.*]] = affine.apply #[[$MAP1]]()
// CHECK:         %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG6]], %[[S5]], %[[S6]], %[[ARG8]]] [1, %[[S7]], %[[S8]], 1] [1, 1, 1, 1] : tensor<2x10x10x5xf32> to tensor<1x?x?x1xf32>
// CHECK:         %[[EXTRACTED_SLICE_10:.*]] = tensor.extract_slice %[[ARG9]][0, 0, %[[ARG2]], %[[ARG4]], %[[ARG6]], %[[ARG8]]] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x2x5xf32> to tensor<6x6x1x1x1x1xf32>
// CHECK:         %[[S9:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<1x?x?x1xf32>) outs(%[[EXTRACTED_SLICE_10]] : tensor<6x6x1x1x1x1xf32>) -> tensor<6x6x1x1x1x1xf32>
// CHECK:         %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S9]] into %[[ARG9]][0, 0, %[[ARG2]], %[[ARG4]], %[[ARG6]], %[[ARG8]]] [6, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x6x1x1x1x1xf32> into tensor<6x6x2x2x2x5xf32>

// -----

func.func @tile_winograd_input(%arg0: tensor<2x10x10x5xf32>, %arg1: tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32> {
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x10x10x5xf32>) outs(%arg1 : tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32>
  return %0 : tensor<6x6x2x2x2x5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop3:4 = transform.structured.tile_using_for %0 tile_sizes [0, 0, 2, 2, 2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (-d0 + 5, 2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP2:.+]] = affine_map<() -> (10)>
// CHECK-LABEL: func.func @tile_winograd_input(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x10x10x5xf32>, %[[ARG1:.*]]: tensor<6x6x2x2x2x5xf32>) -> tensor<6x6x2x2x2x5xf32> {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C0_4:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C0_7:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C2_2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C2_5:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG: %[[C2_0:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C2_3:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C2_6:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C2_8:.*]] = arith.constant 2 : index
// CHECK: %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C2]] step %[[C2_0]] iter_args(%[[ARG3:.*]] = %[[ARG1]])
// CHECK:   %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_1]] to %[[C2_2]] step %[[C2_3]] iter_args(%[[ARG5:.*]] = %[[ARG3]])
// CHECK:     %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_4]] to %[[C2_5]] step %[[C2_6]] iter_args(%[[ARG7:.*]] = %[[ARG5]])
// CHECK:       %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_7]] to %[[C5]] step %[[C2_8]] iter_args(%[[ARG9:.*]] = %[[ARG7]])
// CHECK:         %[[S5:.*]] = affine.min #[[$MAP0]](%[[ARG8]])
// CHECK:         %[[S6:.*]] = affine.apply #[[$MAP1]](%[[ARG2]])
// CHECK:         %[[S7:.*]] = affine.apply #[[$MAP1]](%[[ARG4]])
// CHECK:         %[[S8:.*]] = affine.apply #[[$MAP2]]()
// CHECK:         %[[S9:.*]] = affine.apply #[[$MAP2]]()
// CHECK:         %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG6]], %[[S6]], %[[S7]], %[[ARG8]]] [2, %[[S8]], %[[S9]], %[[S5]]] [1, 1, 1, 1] : tensor<2x10x10x5xf32> to tensor<2x?x?x?xf32>
// CHECK:         %[[EXTRACTED_SLICE_12:.*]] = tensor.extract_slice %[[ARG9]][0, 0, %[[ARG2]], %[[ARG4]], %[[ARG6]], %[[ARG8]]] [6, 6, 2, 2, 2, %[[S5]]] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x2x5xf32> to tensor<6x6x2x2x2x?xf32>
// CHECK:         %[[S10:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<2x?x?x?xf32>) outs(%[[EXTRACTED_SLICE_12]] : tensor<6x6x2x2x2x?xf32>) -> tensor<6x6x2x2x2x?xf32>
// CHECK:         %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S10]] into %[[ARG9]][0, 0, %[[ARG2]], %[[ARG4]], %[[ARG6]], %[[ARG8]]] [6, 6, 2, 2, 2, %[[S5]]] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x2x?xf32> into tensor<6x6x2x2x2x5xf32>

// -----

func.func @tile_winograd_input(%arg0: tensor<2x1x10x5xf32>, %arg1: tensor<1x6x1x2x2x5xf32>) -> tensor<1x6x1x2x2x5xf32> {
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x1x10x5xf32>) outs(%arg1 : tensor<1x6x1x2x2x5xf32>) -> tensor<1x6x1x2x2x5xf32>
  return %0 : tensor<1x6x1x2x2x5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_input_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop3:4 = transform.structured.tile_using_for %0 tile_sizes [0, 0, 1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP2:.+]] = affine_map<() -> (6)>
// CHECK-LABEL: func.func @tile_winograd_input(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x1x10x5xf32>, %[[ARG1:.*]]: tensor<1x6x1x2x2x5xf32>) -> tensor<1x6x1x2x2x5xf32> {
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_3:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_6:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_4:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1_0:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_2:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_5:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_7:.*]] = arith.constant 1 : index
// CHECK:   %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C1]] step %[[C1_0]] iter_args(%[[ARG3:.*]] = %[[ARG1]])
// CHECK:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_1]] to %[[C2]] step %[[C1_2]] iter_args(%[[ARG5:.*]] = %[[ARG3]])
// CHECK:       %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_3]] to %[[C2_4]] step %[[C1_5]] iter_args(%[[ARG7:.*]] = %[[ARG5]])
// CHECK:         %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_6]] to %[[C5]] step %[[C1_7]] iter_args(%[[ARG9:.*]] = %[[ARG7]])
// CHECK:           %[[S5:.*]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[S6:.*]] = affine.apply #[[$MAP1]](%[[ARG4]])
// CHECK:           %[[S7:.*]] = affine.apply #[[$MAP2]]()
// CHECK:           %[[S8:.*]] = affine.apply #[[$MAP2]]()
// CHECK:           %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[ARG6]], %[[S5]], %[[S6]], %[[ARG8]]] [1, 1, %[[S8]], 1] [1, 1, 1, 1] : tensor<2x1x10x5xf32> to tensor<1x1x?x1xf32>
// CHECK:           %[[EXTRACTED_SLICE_10:.*]] = tensor.extract_slice %[[ARG9]][0, 0, %[[ARG2]], %[[ARG4]], %[[ARG6]], %[[ARG8]]] [1, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x6x1x2x2x5xf32> to tensor<1x6x1x1x1x1xf32>
// CHECK:           %[[S9:.*]] = linalg.winograd_input_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<1x1x?x1xf32>) outs(%[[EXTRACTED_SLICE_10]] : tensor<1x6x1x1x1x1xf32>) -> tensor<1x6x1x1x1x1xf32>
// CHECK:         %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[S9]] into %[[ARG9]][0, 0, %[[ARG2]], %[[ARG4]], %[[ARG6]], %[[ARG8]]] [1, 6, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<1x6x1x1x1x1xf32> into tensor<1x6x1x2x2x5xf32>

// -----

func.func @tile_winograd_output(%arg0 : tensor<6x6x2x2x2x2xf32>, %arg1: tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32> {
  %0 = linalg.winograd_output_transform m(4) r(3) ins(%arg0 : tensor<6x6x2x2x2x2xf32>) outs(%arg1 : tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:2 = transform.structured.tile_using_for %0 tile_sizes [0, 0, 1, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP1:.+]] = affine_map<() -> (4)>
// CHECK-LABEL: func.func @tile_winograd_output(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<6x6x2x2x2x2xf32>, %[[ARG1:.*]]: tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32> {
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C2_1:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:   %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG5:.*]] = %[[ARG1]]) -> (tensor<2x8x8x2xf32>)
// CHECK:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C2_1]] step %[[C1_2]] iter_args(%[[ARG6:.*]] = %[[ARG5]]) -> (tensor<2x8x8x2xf32>)
// CHECK:       %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG2]], %[[ARG4]], 0, 0] [6, 6, 1, 1, 2, 2] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x2x2xf32> to tensor<6x6x1x1x2x2xf32>
// CHECK:       %[[S3:.*]] = affine.apply #[[$MAP0]](%[[ARG2]])
// CHECK:       %[[S4:.*]] = affine.apply #[[$MAP0]](%[[ARG4]])
// CHECK:       %[[S5:.*]] = affine.apply #[[$MAP1]]()
// CHECK:       %[[S6:.*]] = affine.apply #[[$MAP1]]()
// CHECK:       %[[EXTRACTED_SLICE_5:.*]] = tensor.extract_slice %[[ARG6]][0, %[[S3]], %[[S4]], 0] [2, %[[S5]], %[[S6]], 2] [1, 1, 1, 1] : tensor<2x8x8x2xf32> to tensor<2x?x?x2xf32>

// -----

func.func @tile_winograd_output(%arg0 : tensor<6x6x2x2x3x5xf32>, %arg1: tensor<3x8x8x5xf32>) -> tensor<3x8x8x5xf32> {
  %0 = linalg.winograd_output_transform m(4) r(3) ins(%arg0 : tensor<6x6x2x2x3x5xf32>) outs(%arg1 : tensor<3x8x8x5xf32>) -> tensor<3x8x8x5xf32>
  return %0 : tensor<3x8x8x5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:4 = transform.structured.tile_using_for %0 tile_sizes [0, 0, 2, 2, 2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (-d0 + 3, 2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 5, 2)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP3:.+]] = affine_map<() -> (8)>
// CHECK-LABEL:  func.func @tile_winograd_output(
// CHECK-SAME:   %[[ARG0:.*]]: tensor<6x6x2x2x3x5xf32>, %[[ARG1:.*]]: tensor<3x8x8x5xf32>) -> tensor<3x8x8x5xf32> {
// CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[C0_4:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[C0_6:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[C2_2:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:    %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:    %[[C2_0:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[C2_3:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[C2_5:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[C2_7:.*]] = arith.constant 2 : index
// CHECK:    %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C2]] step %[[C2_0]] iter_args(%[[ARG9:.*]] = %[[ARG1]]) -> (tensor<3x8x8x5xf32>)
// CHECK:      %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_1]] to %[[C2_2]] step %[[C2_3]] iter_args(%[[ARG10:.*]] = %[[ARG9]]) -> (tensor<3x8x8x5xf32>)
// CHECK:        %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_4]] to %[[C3]] step %[[C2_5]] iter_args(%[[ARG11:.*]] = %[[ARG10]])
// CHECK:          %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_6]] to %[[C5]] step %[[C2_7]] iter_args(%[[ARG12:.*]] = %[[ARG11]])
// CHECK:            %[[C3_8:.*]] = arith.constant 3 : index
// CHECK:            %[[S5:.*]] = affine.min #[[$MAP0]](%[[ARG6]])
// CHECK:            %[[C5_9:.*]] = arith.constant 5 : index
// CHECK:            %[[S6:.*]] = affine.min #[[$MAP1]](%[[ARG8]])
// CHECK:            %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG2]], %[[ARG4]], %[[ARG6]], %[[ARG8]]] [6, 6, 2, 2, %[[S5]], %[[S6]]] [1, 1, 1, 1, 1, 1] : tensor<6x6x2x2x3x5xf32> to tensor<6x6x2x2x?x?xf32>
// CHECK:            %[[S7:.*]] = affine.apply #[[$MAP2]](%[[ARG2]])
// CHECK:            %[[S8:.*]] = affine.apply #[[$MAP2]](%[[ARG4]])
// CHECK:            %[[S9:.*]] = affine.apply #[[$MAP3]]()
// CHECK:            %[[S10:.*]] = affine.apply #[[$MAP3]]()
// CHECK:            %[[EXTRACTED_SLICE_12:.*]] = tensor.extract_slice %[[ARG12]][%[[ARG6]], %[[S7]], %[[S8]], %[[ARG8]]] [%[[S5]], %[[S9]], %[[S10]], %[[S6]]] [1, 1, 1, 1] : tensor<3x8x8x5xf32> to tensor<?x?x?x?xf32>

// -----

func.func @tile_winograd_output(%arg0 : tensor<6x1x2x1x3x5xf32>, %arg1: tensor<3x8x1x5xf32>) -> tensor<3x8x1x5xf32> {
  %0 = linalg.winograd_output_transform m(4) r(3) ins(%arg0 : tensor<6x1x2x1x3x5xf32>) outs(%arg1 : tensor<3x8x1x5xf32>) -> tensor<3x8x1x5xf32>
  return %0 : tensor<3x8x1x5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.winograd_output_transform"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop1:4 = transform.structured.tile_using_for %0 tile_sizes [0, 0, 1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP2:.+]] = affine_map<() -> (4)>
// CHECK-LABEL: func.func @tile_winograd_output(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<6x1x2x1x3x5xf32>, %[[ARG1:.*]]: tensor<3x8x1x5xf32>) -> tensor<3x8x1x5xf32> {
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_3:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C0_5:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1_1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_2:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_4:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C1_6:.*]] = arith.constant 1 : index
// CHECK:   %[[S1:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG9:.*]] = %[[ARG1]]) -> (tensor<3x8x1x5xf32>)
// CHECK:     %[[S2:.*]] = scf.for %[[ARG4:.*]] = %[[C0_0]] to %[[C1_1]] step %[[C1_2]] iter_args(%[[ARG10:.*]] = %[[ARG9]]) -> (tensor<3x8x1x5xf32>)
// CHECK:       %[[S3:.*]] = scf.for %[[ARG6:.*]] = %[[C0_3]] to %[[C3]] step %[[C1_4]] iter_args(%[[ARG11:.*]] = %[[ARG10]]) -> (tensor<3x8x1x5xf32>)
// CHECK:         %[[S4:.*]] = scf.for %[[ARG8:.*]] = %[[C0_5]] to %[[C5]] step %[[C1_6]] iter_args(%[[ARG12:.*]] = %[[ARG11]]) -> (tensor<3x8x1x5xf32>)
// CHECK:           %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG2]], %[[ARG4]], %[[ARG6]], %[[ARG8]]] [6, 1, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<6x1x2x1x3x5xf32> to tensor<6x1x1x1x1x1xf32>
// CHECK:           %[[S5:.*]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[S6:.*]] = affine.apply #[[$MAP1]](%[[ARG4]])
// CHECK:           %[[S7:.*]] = affine.apply #[[$MAP2]]()
// CHECK:           %[[S8:.*]] = affine.apply #[[$MAP2]]()
// CHECK:           %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[ARG12]][%[[ARG6]], %[[S5]], %[[S6]], %[[ARG8]]] [1, %[[S7]], 1, 1] [1, 1, 1, 1] : tensor<3x8x1x5xf32> to tensor<1x?x1x1xf32>
// CHECK:           %[[S9:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXTRACTED_SLICE]] : tensor<6x1x1x1x1x1xf32>) outs(%[[EXTRACTED_SLICE_9]] : tensor<1x?x1x1xf32>) -> tensor<1x?x1x1xf32>
