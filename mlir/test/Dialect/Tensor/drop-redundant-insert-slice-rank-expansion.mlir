// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-drop-redundant-insert-slice-rank-expansion %s | FileCheck %s

// CHECK-LABEL: func @test_drop_rank_expansion(
//  CHECK-SAME:     %[[src:.*]]: tensor<128x480xf32>,
//       CHECK:   %[[extract:.*]] = tensor.extract_slice %[[src]][0, 0] [123, 456] [1, 1] : tensor<128x480xf32> to tensor<123x456xf32>
//       CHECK:   return %[[extract]]
func.func @test_drop_rank_expansion(%src: tensor<128x480xf32>, %dest: tensor<1x1x128x480xf32>) -> tensor<123x456xf32> {
  %inserted_slice = tensor.insert_slice %src into %dest[0, 0, 0, 0] [1, 1, 128, 480] [1, 1, 1, 1] : tensor<128x480xf32> into tensor<1x1x128x480xf32>
  %extracted_slice = tensor.extract_slice %inserted_slice[0, 0, 0, 0] [1, 1, 123, 456] [1, 1, 1, 1] : tensor<1x1x128x480xf32> to tensor<123x456xf32>
  return %extracted_slice : tensor<123x456xf32>
}

// -----

func.func @fold_casting_insert_slice_of_extract_slice(%in : tensor<?x8x2x8xf32>, %dest : tensor<8x1x8xf32>) -> tensor<8x1x8xf32> {
  %extracted_slice = tensor.extract_slice %in[0, 0, 0, 0] [1, 8, 1, 8] [1, 1, 1, 1] : tensor<?x8x2x8xf32> to tensor<8x8xf32>
  %inserted_slice = tensor.insert_slice %extracted_slice into %dest[0, 0, 0] [8, 1, 8] [1, 1, 1] : tensor<8x8xf32> into tensor<8x1x8xf32>
  return %inserted_slice : tensor<8x1x8xf32>
}
// CHECK-LABEL: func.func @fold_casting_insert_slice_of_extract_slice(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?x8x2x8xf32>
// CHECK:         %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0, 0, 0] [1, 8, 1, 8] [1, 1, 1, 1]
// CHECK-SAME:      : tensor<?x8x2x8xf32> to tensor<8x1x8xf32>
// CHECK:         return %[[EXTRACTED_SLICE]] : tensor<8x1x8xf32>

// -----

func.func @fold_casting_insert_slice_of_strided_extract_slice(%in : tensor<?x8x2x8xf32>, %dest : tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %extracted_slice = tensor.extract_slice %in[0, 0, 0, 0] [1, 4, 1, 8] [1, 2, 1, 1] : tensor<?x8x2x8xf32> to tensor<4x8xf32>
  %inserted_slice = tensor.insert_slice %extracted_slice into %dest[0, 0, 0] [1, 4, 8] [1, 1, 1] : tensor<4x8xf32> into tensor<1x4x8xf32>
  return %inserted_slice : tensor<1x4x8xf32>
}
// CHECK-LABEL: func.func @fold_casting_insert_slice_of_strided_extract_slice(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?x8x2x8xf32>
// CHECK:         %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0, 0, 0] [1, 4, 1, 8] [1, 2, 1, 1]
// CHECK-SAME:      : tensor<?x8x2x8xf32> to tensor<1x4x8xf32>
// CHECK:         return %[[EXTRACTED_SLICE]] : tensor<1x4x8xf32>

// -----

func.func @no_fold_more_unit_dims_insert_slice_of_extract_slice(%in : tensor<?x8x8xf32>, %dest : tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32> {
  %extracted_slice = tensor.extract_slice %in[0, 0, 0] [1, 8, 8] [1, 1, 1] : tensor<?x8x8xf32> to tensor<8x8xf32>
  %inserted_slice = tensor.insert_slice %extracted_slice into %dest[0, 0, 0, 0] [1, 1, 8, 8] [1, 1, 1, 1] : tensor<8x8xf32> into tensor<1x1x8x8xf32>
  return %inserted_slice : tensor<1x1x8x8xf32>
}
// CHECK-LABEL: func.func @no_fold_more_unit_dims_insert_slice_of_extract_slice(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?x8x8xf32>
// CHECK:         %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]]
// CHECK:         %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[EXTRACTED_SLICE]]
// CHECK:         return %[[INSERTED_SLICE]] : tensor<1x1x8x8xf32>

// -----

func.func @no_fold_strided_insert_slice_of_extract_slice(%in : tensor<?x8x2x8xf32>, %dest : tensor<1x4x4xf32>) -> tensor<1x4x4xf32> {
  %extracted_slice = tensor.extract_slice %in[0, 0, 0, 0] [1, 8, 1, 8] [1, 1, 1, 1] : tensor<?x8x2x8xf32> to tensor<8x8xf32>
  %inserted_slice = tensor.insert_slice %extracted_slice into %dest[0, 0, 0] [1, 8, 8] [1, 2, 2] : tensor<8x8xf32> into tensor<1x4x4xf32>
  return %inserted_slice : tensor<1x4x4xf32>
}
// CHECK-LABEL: func.func @no_fold_strided_insert_slice_of_extract_slice(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?x8x2x8xf32>
// CHECK:         %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]]
// CHECK:         %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[EXTRACTED_SLICE]]
// CHECK:         return %[[INSERTED_SLICE]] : tensor<1x4x4xf32>

// -----

func.func @no_fold_non_casting_insert_slice_of_extract_slice(%in : tensor<1x1x1x8x8xf32>, %dest : tensor<2x8x8xf32>) -> tensor<2x8x8xf32> {
  %extracted_slice = tensor.extract_slice %in[0, 0, 0, 0, 0] [1, 1, 1, 8, 8] [1, 1, 1, 1, 1] : tensor<1x1x1x8x8xf32> to tensor<8x8xf32>
  %inserted_slice = tensor.insert_slice %extracted_slice into %dest[0, 0, 0] [1, 8, 8] [1, 1, 1] : tensor<8x8xf32> into tensor<2x8x8xf32>
  return %inserted_slice : tensor<2x8x8xf32>
}
// CHECK-LABEL: func.func @no_fold_non_casting_insert_slice_of_extract_slice(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x1x1x8x8xf32>
// CHECK:         %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[ARG0]]
// CHECK:         %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[EXTRACTED_SLICE]]
// CHECK:         return %[[INSERTED_SLICE]] : tensor<2x8x8xf32>
