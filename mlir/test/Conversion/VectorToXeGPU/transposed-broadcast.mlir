// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s

// broadcast + transpose is rewritten to shape_cast + broadcast.

gpu.module @xevm_module {
gpu.func @row_vector_along_columns(%v: vector<16xf32>) -> vector<16x64xf32> {
  %b = vector.broadcast %v : vector<16xf32> to vector<64x16xf32>
  %t = vector.transpose %b, [1, 0] : vector<64x16xf32> to vector<16x64xf32>
  gpu.return %t : vector<16x64xf32>
}

// CHECK-LABEL: @row_vector_along_columns(
// CHECK-SAME:    %[[V:.+]]: vector<16xf32>
// CHECK:         %[[CAST:.+]] = vector.shape_cast %[[V]] : vector<16xf32> to vector<16x1xf32>
// CHECK:         %[[RES:.+]] = vector.broadcast %[[CAST]] : vector<16x1xf32> to vector<16x64xf32>
// CHECK-NOT:     vector.transpose
// CHECK:         gpu.return %[[RES]] : vector<16x64xf32>
}

// -----

// The source's own dims are shuffled, so the transpose is real data movement
// and has to stay.
gpu.module @xevm_module {
gpu.func @negative_source_dims_shuffled(%v: vector<8x16xf32>) -> vector<16x8x4xf32> {
  %b = vector.broadcast %v : vector<8x16xf32> to vector<4x8x16xf32>
  %t = vector.transpose %b, [2, 1, 0] : vector<4x8x16xf32> to vector<16x8x4xf32>
  gpu.return %t : vector<16x8x4xf32>
}

// CHECK-LABEL: @negative_source_dims_shuffled(
// CHECK:         %[[B:.+]] = vector.broadcast %{{.+}} : vector<8x16xf32> to vector<4x8x16xf32>
// CHECK:         %[[RES:.+]] = vector.transpose %[[B]], [2, 1, 0]
// CHECK:         gpu.return %[[RES]] : vector<16x8x4xf32>
}
