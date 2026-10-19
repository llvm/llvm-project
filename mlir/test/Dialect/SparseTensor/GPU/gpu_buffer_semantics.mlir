// RUN: mlir-opt %s --split-input-file --sparse-gpu-codegen="num-threads=0" | FileCheck %s

// Verifies that the sparse GPU rewrites leave linalg operations with buffer
// semantics alone. Their operands have memref types, which do not carry a
// sparse tensor encoding, so the ops must not be matched by the rewrites.

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @matmul_buffer_semantics
// CHECK-NOT:     gpu.
// CHECK:         linalg.generic
// CHECK:         return
func.func @matmul_buffer_semantics(%arga: memref<4x8xf32>,
                                   %argb: memref<8x6xf32>,
                                   %argc: memref<4x6xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]
  }
  ins(%arga, %argb : memref<4x8xf32>, memref<8x6xf32>)
  outs(%argc : memref<4x6xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %0 = arith.mulf %a, %b : f32
    %1 = arith.addf %c, %0 : f32
    linalg.yield %1 : f32
  }
  return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @matvec_buffer_semantics
// CHECK-NOT:     gpu.
// CHECK:         linalg.generic
// CHECK:         return
func.func @matvec_buffer_semantics(%arga: memref<4x8xf32>,
                                   %argb: memref<8xf32>,
                                   %argc: memref<4xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction"]
  }
  ins(%arga, %argb : memref<4x8xf32>, memref<8xf32>)
  outs(%argc : memref<4xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %0 = arith.mulf %a, %b : f32
    %1 = arith.addf %c, %0 : f32
    linalg.yield %1 : f32
  }
  return
}
