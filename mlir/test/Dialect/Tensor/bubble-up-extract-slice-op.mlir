// RUN: mlir-opt -split-input-file -transform-interpreter  %s | FileCheck %s

///----------------------------------------------------------------------------------------
/// [Pattern: BubbleUpExpandShapeThroughExtractSlice]
///
/// IN: tensor.expand_shape(tensor.extract_slice)
/// OUT:tensor.extract_slice(tensor.expand_shape)
///
/// Note: tensor.extract_slice is bubbled up to be before tensor.expand_shape.
///       Some tests are negative tests for cases where the pattern cannot be applied.
///----------------------------------------------------------------------------------------

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_expand_shape(
// CHECK-SAME:                %[[SRC:.*]]: tensor<60xf32>) -> tensor<1x1x5xf32> {
// CHECK:           %[[C1:.+]] = arith.constant 5 : index
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][%[[C1]]] [5] [1] : tensor<60xf32> to tensor<5xf32>
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]] output_shape [1, 1, 5] : tensor<5xf32> into tensor<1x1x5xf32>
// CHECK:           return %[[EXPAND]] : tensor<1x1x5xf32>

func.func @bubble_up_extract_slice_through_expand_shape(%src: tensor<60xf32>) -> tensor<1x1x5xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [2, 3, 10] : tensor<60xf32> into tensor<2x3x10xf32>
  %extract = tensor.extract_slice %expand[0, 0, 5][1, 1, 5][1, 1, 1] : tensor<2x3x10xf32> to tensor<1x1x5xf32>
  return %extract : tensor<1x1x5xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_on_non_contiguous(
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape 
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice 
// CHECK:           return %[[EXTRACT]]

func.func @no_bubble_up_extract_slice_on_non_contiguous(%src: tensor<60xf32>) -> tensor<1x2x5xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [2, 3, 10] : tensor<60xf32> into tensor<2x3x10xf32>
  %extract = tensor.extract_slice %expand[0, 0, 5][1, 2, 5][1, 1, 1] : tensor<2x3x10xf32> to tensor<1x2x5xf32>
  return %extract : tensor<1x2x5xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_on_stride(
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape 
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice 
// CHECK:           return %[[EXTRACT]]

func.func @no_bubble_up_extract_slice_on_stride(%src: tensor<60xf32>) -> tensor<1x1x5xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [2, 3, 10] : tensor<60xf32> into tensor<2x3x10xf32>
  %extract = tensor.extract_slice %expand[0, 0, 0][1, 1, 5][1, 1, 2] : tensor<2x3x10xf32> to tensor<1x1x5xf32>
  return %extract : tensor<1x1x5xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_on_rank_reducing(
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape 
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice 
// CHECK:           return %[[EXTRACT]]

func.func @no_bubble_up_extract_slice_on_rank_reducing(%src: tensor<60xf32>) -> tensor<1x5xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [2, 3, 10] : tensor<60xf32> into tensor<2x3x10xf32>
  %extract = tensor.extract_slice %expand[0, 0, 5][1, 1, 5][1, 1, 1] : tensor<2x3x10xf32> to tensor<1x5xf32>
  return %extract : tensor<1x5xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_expand_shape_multiple_expanded_dims(
// CHECK-SAME:                  %[[SRC:.*]]: tensor<120x56xf32>) -> tensor<1x2x10x1x4xf32> {
// CHECK:           %[[C0:.+]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][%[[C0]], %[[C0]]] [20, 4] [1, 1] : tensor<120x56xf32> to tensor<20x4xf32>
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]] {{\[\[}}0, 1, 2], [3, 4]] output_shape [1, 2, 10, 1, 4] : tensor<20x4xf32> into tensor<1x2x10x1x4xf32>
// CHECK:           return %[[EXPAND]] : tensor<1x2x10x1x4xf32>

func.func @bubble_up_extract_slice_through_expand_shape_multiple_expanded_dims(%src: tensor<120x56xf32>) -> tensor<1x2x10x1x4xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2], [3, 4]] output_shape [3, 4, 10, 7, 8] : tensor<120x56xf32> into tensor<3x4x10x7x8xf32>
  %extract = tensor.extract_slice %expand[0, 0, 0, 0, 0][1, 2, 10, 1, 4][1, 1, 1, 1, 1] : tensor<3x4x10x7x8xf32> to tensor<1x2x10x1x4xf32>
  return %extract : tensor<1x2x10x1x4xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_with_trailing_full_dims(
// CHECK-SAME:                %[[SRC:.*]]: tensor<60xf32>) -> tensor<2x5x2xf32> {
// CHECK:           %[[C0:.+]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][%[[C0]]] [20] [1] : tensor<60xf32> to tensor<20xf32>
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]] output_shape [2, 5, 2] : tensor<20xf32> into tensor<2x5x2xf32>
// CHECK:           return %[[EXPAND]] : tensor<2x5x2xf32>
func.func @bubble_up_extract_slice_with_trailing_full_dims(%src: tensor<60xf32>) -> tensor<2x5x2xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [6, 5, 2] : tensor<60xf32> into tensor<6x5x2xf32>
  %extract = tensor.extract_slice %expand[0, 0, 0][2, 5, 2][1, 1, 1] : tensor<6x5x2xf32> to tensor<2x5x2xf32>
  return %extract : tensor<2x5x2xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_dont_fold_linearize_index(
// CHECK-SAME:                 %[[SRC:.*]]: tensor<60xf32>,
// CHECK-SAME:                 %[[OFFSET_0:.*]]: index,
// CHECK-SAME:                 %[[OFFSET_1:.*]]: index) -> tensor<1x1x5xf32> {
// CHECK:           %[[C1:.+]] = arith.constant 5 : index
// CHECK:           %[[LINEARIZE:.*]] = affine.linearize_index disjoint {{\[}}%[[OFFSET_0]], %[[OFFSET_1]], %[[C1]]] by (2, 3, 10) : index
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][%[[LINEARIZE]]] [5] [1] : tensor<60xf32> to tensor<5xf32>
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]] output_shape [1, 1, 5] : tensor<5xf32> into tensor<1x1x5xf32>
// CHECK:           return %[[EXPAND]] : tensor<1x1x5xf32>
func.func @bubble_up_extract_slice_dont_fold_linearize_index(%src: tensor<60xf32>, %offset_0 : index, %offset_1 : index) -> tensor<1x1x5xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [2, 3, 10] : tensor<60xf32> into tensor<2x3x10xf32>
  %extract = tensor.extract_slice %expand[%offset_0, %offset_1, 5][1, 1, 5][1, 1, 1] : tensor<2x3x10xf32> to tensor<1x1x5xf32>
  return %extract : tensor<1x1x5xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_not_all_dims_expanded(
// CHECK-SAME:                %[[SRC:.*]]: tensor<60x12xf32>) -> tensor<1x1x5x12xf32> {
// CHECK-DAG:       %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]]{{\[}}%[[C5]], %[[C0]]] [5, 12] [1, 1] : tensor<60x12xf32> to tensor<5x12xf32>
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]] {{\[\[}}0, 1, 2], [3]] output_shape [1, 1, 5, 12] : tensor<5x12xf32> into tensor<1x1x5x12xf32>
// CHECK:           return %[[EXPAND]] : tensor<1x1x5x12xf32>
func.func @bubble_up_extract_slice_not_all_dims_expanded(%src: tensor<60x12xf32>) -> tensor<1x1x5x12xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2], [3]] output_shape [2, 3, 10, 12] : tensor<60x12xf32> into tensor<2x3x10x12xf32>
  %extract = tensor.extract_slice %expand[0, 0, 5, 0][1, 1, 5, 12][1, 1, 1, 1] : tensor<2x3x10x12xf32> to tensor<1x1x5x12xf32>
  return %extract : tensor<1x1x5x12xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_affine_apply_not_folded(
// CHECK-SAME:                   %[[SRC:.*]]: tensor<60xf32>,
// CHECK-SAME:                   %[[SLICE_SIZE:.*]]: index) -> tensor<?x5x2xf32> {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[AFFINE_APPLY:.*]] = affine.apply #map(){{\[}}%[[SLICE_SIZE]]]
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]]{{\[}}%[[C0]]] {{\[}}%[[AFFINE_APPLY]]] [1] : tensor<60xf32> to tensor<?xf32>
// CHECK:           %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]] output_shape {{\[}}%[[SLICE_SIZE]], 5, 2] : tensor<?xf32> into tensor<?x5x2xf32>
// CHECK:           return %[[EXPAND]] : tensor<?x5x2xf32>
func.func @bubble_up_extract_slice_affine_apply_not_folded(%src: tensor<60xf32>, %slice_size : index) -> tensor<?x5x2xf32> {
  %expand = tensor.expand_shape %src [[0, 1, 2]] output_shape [6, 5, 2] : tensor<60xf32> into tensor<6x5x2xf32>
  %extract = tensor.extract_slice %expand[0, 0, 0][%slice_size, 5, 2][1, 1, 1] : tensor<6x5x2xf32> to tensor<?x5x2xf32>
  return %extract : tensor<?x5x2xf32>
}

///----------------------------------------------------------------------------------------
/// [Pattern: BubbleUpCollapseShapeThroughExtractSlice]
///
/// IN: tensor.collapse_shape(tensor.extract_slice)
/// OUT:tensor.extract_slice(tensor.collapse_shape)
///
/// Note: tensor.extract_slice is bubbled up to be before tensor.collapse_shape.
///       Some tests are negative tests for cases where the pattern cannot be applied.
///----------------------------------------------------------------------------------------

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_single_reassoc_group(
// CHECK-SAME:                   %[[SRC:.*]]: tensor<6x5x2xf32>) -> tensor<1xf32> {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][%[[C0]], %[[C0]], %[[C0]]] [1, 1, 1] [1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_single_reassoc_group(%src: tensor<6x5x2xf32>) -> tensor<1xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<6x5x2xf32> into tensor<60xf32>
  %extract = tensor.extract_slice %collapse[0][1][1] : tensor<60xf32> to tensor<1xf32>
  return %extract : tensor<1xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_multiple_reassoc_group(
// CHECK-SAME:                      %[[SRC:.*]]: tensor<6x5x3x10xf32>) -> tensor<15x10xf32> {
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][%[[C1]], %[[C0]], %[[C1]], %[[C0]]] [3, 5, 1, 10] [1, 1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1], [2, 3]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_multiple_reassoc_group(%src: tensor<6x5x3x10xf32>) -> tensor<15x10xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1], [2, 3]] : tensor<6x5x3x10xf32> into tensor<30x30xf32>
  %extract = tensor.extract_slice %collapse[5, 10][15, 10][1, 1] : tensor<30x30xf32> to tensor<15x10xf32>
  return %extract : tensor<15x10xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_offset_on_leading_dim(
// CHECK-SAME:                         %[[SRC:.*]]: tensor<6x5x2xf32>) -> tensor<4xf32> {
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][%[[C2]], %[[C0]], %[[C0]]] [1, 2, 2] [1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_offset_on_leading_dim(%src: tensor<6x5x2xf32>) -> tensor<4xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<6x5x2xf32> into tensor<60xf32>
  %extract = tensor.extract_slice %collapse[20][4][1] : tensor<60xf32> to tensor<4xf32>
  return %extract : tensor<4xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_dynamic_size(
// CHECK-SAME:                    %[[SRC:.*]]: tensor<1x5x1xf32>,
// CHECK-SAME:                    %[[SIZE:.*]]: index) -> tensor<?xf32> {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][%[[C0]], %[[C0]], %[[C0]]] [1, %[[SIZE]], 1] [1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_dynamic_size(%src: tensor<1x5x1xf32>, %size : index) -> tensor<?xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<1x5x1xf32> into tensor<5xf32>
  %extract = tensor.extract_slice %collapse[0][%size][1] : tensor<5xf32> to tensor<?xf32>
  return %extract : tensor<?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_dynamic_size_and_src(
// CHECK-SAME:                    %[[SRC:.*]]: tensor<1x?x1xf32>,
// CHECK-SAME:                    %[[SIZE:.*]]: index) -> tensor<?xf32> {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM:.*]] = tensor.dim %[[SRC]], %[[C1]]
// CHECK:           %[[DELIN:.*]]:3 = affine.delinearize_index %[[C0]] into (1, %[[DIM]], 1)
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][%[[DELIN]]#0, %[[DELIN]]#1, %[[DELIN]]#2] [1, %[[SIZE]], 1] [1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_dynamic_size_and_src(%src: tensor<1x?x1xf32>, %size : index) -> tensor<?xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<1x?x1xf32> into tensor<?xf32>
  %extract = tensor.extract_slice %collapse[0][%size][1] : tensor<?xf32> to tensor<?xf32>
  return %extract : tensor<?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_dynamic_offset(
// CHECK-SAME:                       %[[SRC:.*]]: tensor<1x5x1xf32>,
// CHECK-SAME:                       %[[OFFSET:.*]]: index) -> tensor<3xf32> {
// CHECK:           %[[DELIN:.*]]:3 = affine.delinearize_index %[[OFFSET]] into (1, 5, 1)
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][%[[DELIN]]#0, %[[DELIN]]#1, %[[DELIN]]#2] [1, 3, 1] [1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_dynamic_offset(%src: tensor<1x5x1xf32>, %offset : index) -> tensor<3xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<1x5x1xf32> into tensor<5xf32>
  %extract = tensor.extract_slice %collapse[%offset][3][1] : tensor<5xf32> to tensor<3xf32>
  return %extract : tensor<3xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_dynamic_offset_and_size(
// CHECK-SAME:                     %[[SRC:.*]]: tensor<14x1xf32>,
// CHECK-SAME:                     %[[OFFSET:.*]]: index,
// CHECK-SAME:                     %[[SIZE:.*]]: index) -> tensor<?xf32> {
// CHECK:           %[[DELIN:.*]]:2 = affine.delinearize_index %[[OFFSET]] into (14, 1)
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]]{{\[}}%[[DELIN]]#0, %[[DELIN]]#1] {{\[}}%[[SIZE]], 1] [1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_dynamic_offset_and_size(%src: tensor<14x1xf32>, %offset : index, %size : index) -> tensor<?xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1]] : tensor<14x1xf32> into tensor<14xf32>
  %extract = tensor.extract_slice %collapse[%offset][%size][1] : tensor<14xf32> to tensor<?xf32>
  return %extract : tensor<?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_dynamic_and_static_groups(
// CHECK-SAME:                      %[[SRC:.*]]: tensor<5x10x1x1x40xf32>,
// CHECK-SAME:                      %[[OFFSET:.*]]: index,
// CHECK-SAME:                      %[[SIZE:.*]]: index) -> tensor<20x?xf32> {
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[DELIN:.*]]:3 = affine.delinearize_index %[[OFFSET]] into (1, 1, 40)
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][%[[C1]], %[[C0]], %[[DELIN]]#0, %[[DELIN]]#1, %[[DELIN]]#2] [2, 10, 1, 1, %[[SIZE]]] [1, 1, 1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1], [2, 3, 4]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_dynamic_and_static_groups(%src: tensor<5x10x1x1x40xf32>, %offset : index, %size : index) -> tensor<20x?xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1], [2, 3, 4]] : tensor<5x10x1x1x40xf32> into tensor<50x40xf32>
  %extract = tensor.extract_slice %collapse[10, %offset][20, %size][1, 1] : tensor<50x40xf32> to tensor<20x?xf32>
  return %extract : tensor<20x?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_unit_size_dynamic_offset(
// CHECK-SAME:                      %[[SRC:.*]]: tensor<2x3x10xf32>,
// CHECK-SAME:                      %[[OFFSET:.*]]: index) -> tensor<1xf32> {
// CHECK:           %[[DELINEARIZE:.*]]:3 = affine.delinearize_index %[[OFFSET]] into (2, 3, 10)
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]]{{\[}}%[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[DELINEARIZE]]#2] [1, 1, 1] [1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_unit_size_dynamic_offset(%src: tensor<2x3x10xf32>, %offset : index) -> tensor<1xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<2x3x10xf32> into tensor<60xf32>
  %extract = tensor.extract_slice %collapse[%offset][1][1] : tensor<60xf32> to tensor<1xf32>
  return %extract : tensor<1xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_unit_size_dynamic_offset_multi_group(
// CHECK-SAME:                      %[[SRC:.*]]: tensor<2x3x4x5xf32>,
// CHECK-SAME:                      %[[OFFSET0:.*]]: index,
// CHECK-SAME:                      %[[OFFSET1:.*]]: index) -> tensor<1x1xf32> {
// CHECK-DAG:       %[[DELINEARIZE0:.*]]:2 = affine.delinearize_index %[[OFFSET0]] into (2, 3)
// CHECK-DAG:       %[[DELINEARIZE1:.*]]:2 = affine.delinearize_index %[[OFFSET1]] into (4, 5)
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]]{{\[}}%[[DELINEARIZE0]]#0, %[[DELINEARIZE0]]#1, %[[DELINEARIZE1]]#0, %[[DELINEARIZE1]]#1] [1, 1, 1, 1] [1, 1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1], [2, 3]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_unit_size_dynamic_offset_multi_group(%src: tensor<2x3x4x5xf32>, %offset0 : index, %offset1 : index) -> tensor<1x1xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1], [2, 3]] : tensor<2x3x4x5xf32> into tensor<6x20xf32>
  %extract = tensor.extract_slice %collapse[%offset0, %offset1][1, 1][1, 1] : tensor<6x20xf32> to tensor<1x1xf32>
  return %extract : tensor<1x1xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_affine_apply_offset(
// CHECK-SAME:                      %[[SRC:.*]]: tensor<2x3x10xf32>,
// CHECK-SAME:                      %[[IDX:.*]]: index) -> tensor<5xf32> {
// CHECK:           %[[OFFSET:.*]] = affine.apply
// CHECK:           %[[DELINEARIZE:.*]]:3 = affine.delinearize_index %[[OFFSET]] into (2, 3, 10)
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]]{{\[}}%[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[DELINEARIZE]]#2] [1, 1, 5] [1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_affine_apply_offset(%src: tensor<2x3x10xf32>, %idx : index) -> tensor<5xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<2x3x10xf32> into tensor<60xf32>
  %offset = affine.apply affine_map<()[s0] -> (s0 * 5)>()[%idx]
  %extract = tensor.extract_slice %collapse[%offset][5][1] : tensor<60xf32> to tensor<5xf32>
  return %extract : tensor<5xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_affine_apply_offset_full_inner_dim(
// CHECK-SAME:                      %[[SRC:.*]]: tensor<4x5x8xf32>,
// CHECK-SAME:                      %[[IDX:.*]]: index) -> tensor<8xf32> {
// CHECK:           %[[OFFSET:.*]] = affine.apply
// CHECK:           %[[DELINEARIZE:.*]]:3 = affine.delinearize_index %[[OFFSET]] into (4, 5, 8)
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]]{{\[}}%[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[DELINEARIZE]]#2] [1, 1, 8] [1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_affine_apply_offset_full_inner_dim(%src: tensor<4x5x8xf32>, %idx : index) -> tensor<8xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<4x5x8xf32> into tensor<160xf32>
  // offset = idx * 8, size = 8, innermost dim = 8, 8 % 8 == 0, should work
  %offset = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%idx]
  %extract = tensor.extract_slice %collapse[%offset][8][1] : tensor<160xf32> to tensor<8xf32>
  return %extract : tensor<8xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_affine_apply_complex_expr(
// CHECK-SAME:                      %[[SRC:.*]]: tensor<3x4x8xf32>,
// CHECK-SAME:                      %[[IDX:.*]]: index) -> tensor<4xf32> {
// CHECK:           %[[OFFSET:.*]] = affine.apply
// CHECK:           %[[DELINEARIZE:.*]]:3 = affine.delinearize_index %[[OFFSET]] into (3, 4, 8)
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]]{{\[}}%[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[DELINEARIZE]]#2] [1, 1, 4] [1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_affine_apply_complex_expr(%src: tensor<3x4x8xf32>, %idx : index) -> tensor<4xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<3x4x8xf32> into tensor<96xf32>
  // offset = idx * 4, size = 4, innermost dim = 8, 8 % 4 == 0, should work
  %offset = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%idx]
  %extract = tensor.extract_slice %collapse[%offset][4][1] : tensor<96xf32> to tensor<4xf32>
  return %extract : tensor<4xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_affine_apply_multi_group(
// CHECK-SAME:                      %[[SRC:.*]]: tensor<2x3x4x6xf32>,
// CHECK-SAME:                      %[[IDX0:.*]]: index,
// CHECK-SAME:                      %[[IDX1:.*]]: index) -> tensor<3x2xf32> {
// CHECK-DAG:       %[[OFFSET0:.*]] = affine.apply {{.*}}{{\[}}%[[IDX0]]]
// CHECK-DAG:       %[[OFFSET1:.*]] = affine.apply {{.*}}{{\[}}%[[IDX1]]]
// CHECK-DAG:       %[[DELINEARIZE0:.*]]:2 = affine.delinearize_index %[[OFFSET0]] into (2, 3)
// CHECK-DAG:       %[[DELINEARIZE1:.*]]:2 = affine.delinearize_index %[[OFFSET1]] into (4, 6)
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]]{{\[}}%[[DELINEARIZE0]]#0, %[[DELINEARIZE0]]#1, %[[DELINEARIZE1]]#0, %[[DELINEARIZE1]]#1] [1, 3, 1, 2] [1, 1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1], [2, 3]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_affine_apply_multi_group(%src: tensor<2x3x4x6xf32>, %idx0 : index, %idx1 : index) -> tensor<3x2xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1], [2, 3]] : tensor<2x3x4x6xf32> into tensor<6x24xf32>
  // For first group: size=3, innermost=3, 3%3==0, offset=idx0*3, works
  // For second group: size=2, innermost=6, 6%2==0, offset=idx1*2, works
  %offset0 = affine.apply affine_map<()[s0] -> (s0 * 3)>()[%idx0]
  %offset1 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%idx1]
  %extract = tensor.extract_slice %collapse[%offset0, %offset1][3, 2][1, 1] : tensor<6x24xf32> to tensor<3x2xf32>
  return %extract : tensor<3x2xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_affine_apply_nested(
// CHECK-SAME:                      %[[SRC:.*]]: tensor<2x4x8xf32>,
// CHECK-SAME:                      %[[IDX:.*]]: index) -> tensor<4xf32> {
// CHECK:           affine.apply
// CHECK:           %[[OFFSET:.*]] = affine.apply
// CHECK:           %[[DELINEARIZE:.*]]:3 = affine.delinearize_index %[[OFFSET]] into (2, 4, 8)
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]]{{\[}}%[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[DELINEARIZE]]#2] [1, 1, 4] [1, 1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1, 2]]
// CHECK:           return %[[COLLAPSE]]
func.func @bubble_up_extract_slice_through_collapse_shape_affine_apply_nested(%src: tensor<2x4x8xf32>, %idx : index) -> tensor<4xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<2x4x8xf32> into tensor<64xf32>
  // Nested affine.apply: offset = (idx * 2) * 2 = idx * 4, multiple of 4
  %tmp = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%idx]
  %offset = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%tmp]
  %extract = tensor.extract_slice %collapse[%offset][4][1] : tensor<64xf32> to tensor<4xf32>
  return %extract : tensor<4xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_affine_apply_offset_not_multiple(
// CHECK-SAME:                      %[[SRC:.*]]: tensor<2x3x10xf32>,
// CHECK-SAME:                      %[[IDX:.*]]: index) -> tensor<4xf32> {
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape
// CHECK:           %[[OFFSET:.*]] = affine.apply
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice
// CHECK:           return %[[EXTRACT]]
func.func @no_bubble_up_extract_slice_affine_apply_offset_not_multiple(%src: tensor<2x3x10xf32>, %idx : index) -> tensor<4xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<2x3x10xf32> into tensor<60xf32>
  // offset = idx * 3, but size = 4, 3 is not a multiple of 4
  %offset = affine.apply affine_map<()[s0] -> (s0 * 3)>()[%idx]
  %extract = tensor.extract_slice %collapse[%offset][4][1] : tensor<60xf32> to tensor<4xf32>
  return %extract : tensor<4xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_affine_apply_size_not_dividing_inner(
// CHECK-SAME:                      %[[SRC:.*]]: tensor<2x3x10xf32>,
// CHECK-SAME:                      %[[IDX:.*]]: index) -> tensor<3xf32> {
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape
// CHECK:           %[[OFFSET:.*]] = affine.apply
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice
// CHECK:           return %[[EXTRACT]]
func.func @no_bubble_up_extract_slice_affine_apply_size_not_dividing_inner(%src: tensor<2x3x10xf32>, %idx : index) -> tensor<3xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<2x3x10xf32> into tensor<60xf32>
  // offset = idx * 3 (multiple of 3), but size = 3, innermost = 10, 10 % 3 != 0
  %offset = affine.apply affine_map<()[s0] -> (s0 * 3)>()[%idx]
  %extract = tensor.extract_slice %collapse[%offset][3][1] : tensor<60xf32> to tensor<3xf32>
  return %extract : tensor<3xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_dynamic_offset_not_affine_apply(
// CHECK-SAME:                      %[[SRC:.*]]: tensor<2x3x10xf32>,
// CHECK-SAME:                      %[[OFFSET:.*]]: index) -> tensor<5xf32> {
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice
// CHECK:           return %[[EXTRACT]]
func.func @no_bubble_up_extract_slice_dynamic_offset_not_affine_apply(%src: tensor<2x3x10xf32>, %offset : index) -> tensor<5xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<2x3x10xf32> into tensor<60xf32>
  // offset is a plain Value (not from affine.apply), can't prove it's a multiple of 5
  %extract = tensor.extract_slice %collapse[%offset][5][1] : tensor<60xf32> to tensor<5xf32>
  return %extract : tensor<5xf32>
}

/// The 2 following tests are cases where the bubble up cannot occur because the contiguous size extracted 
/// from the collapsed shape cannot be expressed via a single extract_slice op.
/// In the first test it is because the size extracted cannot be expressed as a slice
/// of the form [ 1, 1, ..., 1, Sk, Ak + 1, Ak + 2, ...,An ] (see the pattern documentation for more details).
/// In the second test, the size can be expressed as the required form, but the offset is such that the pattern
/// cannot be applied.

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_through_collapse_shape_on_non_contiguous_1(
// CHECK-SAME:                              %[[SRC:.*]]: tensor<2x3x10xf32>) -> tensor<15xf32> {
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice
func.func @no_bubble_up_extract_slice_through_collapse_shape_on_non_contiguous_1(%src: tensor<2x3x10xf32>) -> tensor<15xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<2x3x10xf32> into tensor<60xf32>
  %extract = tensor.extract_slice %collapse[0][15][1] : tensor<60xf32> to tensor<15xf32>
  return %extract : tensor<15xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_through_collapse_shape_on_non_contiguous_2(
// CHECK-SAME:                              %[[SRC:.*]]: tensor<2x3x10xf32>) -> tensor<20xf32> {
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice
func.func @no_bubble_up_extract_slice_through_collapse_shape_on_non_contiguous_2(%src: tensor<2x3x10xf32>) -> tensor<20xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<2x3x10xf32> into tensor<60xf32>
  %extract = tensor.extract_slice %collapse[20][20][1] : tensor<60xf32> to tensor<20xf32>
  return %extract : tensor<20xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_through_collapse_shape_on_stride(
// CHECK-SAME:                              %[[SRC:.*]]: tensor<2x3x10xf32>) -> tensor<5xf32> {
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice
func.func @no_bubble_up_extract_slice_through_collapse_shape_on_stride(%src: tensor<2x3x10xf32>) -> tensor<5xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<2x3x10xf32> into tensor<60xf32>
  %extract = tensor.extract_slice %collapse[0][5][2] : tensor<60xf32> to tensor<5xf32>
  return %extract : tensor<5xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_through_collapse_shape_on_rank_reducing(
// CHECK-SAME:                              %[[SRC:.*]]: tensor<6x5x2x1xf32>) -> tensor<1xf32> {
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice
func.func @no_bubble_up_extract_slice_through_collapse_shape_on_rank_reducing(%src: tensor<6x5x2x1xf32>) -> tensor<1xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2], [3]] : tensor<6x5x2x1xf32> into tensor<60x1xf32>
  %extract = tensor.extract_slice %collapse[0, 0][1, 1][1, 1] : tensor<60x1xf32> to tensor<1xf32>
  return %extract : tensor<1xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_through_collapse_shape_on_unsupported_dynamic(
// CHECK-SAME:                              %[[SRC:.*]]: tensor<1x5x2xf32>,
// CHECK-SAME:                              %[[SIZE:.*]]: index) -> tensor<?xf32> {
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice
func.func @no_bubble_up_extract_slice_through_collapse_shape_on_unsupported_dynamic(%src: tensor<1x5x2xf32>, %size : index) -> tensor<?xf32> {
  %collapse = tensor.collapse_shape %src [[0, 1, 2]] : tensor<1x5x2xf32> into tensor<10xf32>
  %extract = tensor.extract_slice %collapse[0][%size][1] : tensor<10xf32> to tensor<?xf32>
  return %extract : tensor<?xf32>
}

// CHECK-LABEL:   func.func @bubble_up_extract_slice_through_collapse_shape_boundary_offset(
// CHECK-SAME:                      %[[SRC:.*]]: tensor<3x10xf32>) -> tensor<5xf32> {
// CHECK-DAG:       %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[SRC]][%[[C0]], %[[C5]]] [1, 5] [1, 1]
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape %[[EXTRACT]] {{\[\[}}0, 1]]
// CHECK:           return %[[COLLAPSE]]

func.func @bubble_up_extract_slice_through_collapse_shape_boundary_offset(%src: tensor<3x10xf32>) -> tensor<5xf32> {
  %collapsed = tensor.collapse_shape %src [[0, 1]] : tensor<3x10xf32> into tensor<30xf32>
  %extracted_slice = tensor.extract_slice %collapsed[5] [5] [1] : tensor<30xf32> to tensor<5xf32>
  return %extracted_slice : tensor<5xf32>
}

// CHECK-LABEL:   func.func @no_bubble_up_extract_slice_affine_apply_dynamic_offset_multi_trailing
// CHECK:           %[[COLLAPSE:.*]] = tensor.collapse_shape
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice

func.func @no_bubble_up_extract_slice_affine_apply_dynamic_offset_multi_trailing(%arg0: tensor<2x8x6xf32>, %arg1: index) -> tensor<24xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<2x8x6xf32> into tensor<96xf32>
  %0 = affine.apply affine_map<()[s0] -> (s0 * 12)>()[%arg1]
  %extracted_slice = tensor.extract_slice %collapsed[%0] [24] [1] : tensor<96xf32> to tensor<24xf32>
  return %extracted_slice : tensor<24xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.tensor.bubble_up_extract_slice
    } : !transform.op<"func.func">
    transform.yield
  }
}
