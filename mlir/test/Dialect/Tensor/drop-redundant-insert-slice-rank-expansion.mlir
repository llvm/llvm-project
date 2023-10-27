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
