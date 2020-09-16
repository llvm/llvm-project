// RUN: mlir-opt %s -inline | FileCheck %s

// These tests verify that regions with operations from Lingalg dialect
// can be inlined.

#accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>
]

#trait = {
  args_in = 1,
  args_out = 1,
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

func @inline_into(%arg0: memref<?xf32>) {
  // CHECK: linalg.generic
  call @inlined_fn(%arg0) : (memref<?xf32>) -> ()
  return
}

func @inlined_fn(%arg0: memref<?xf32>) {
  // CHECK: linalg.generic
  linalg.generic #trait %arg0, %arg0 {
    ^bb(%0 : f32, %1 : f32) :
      linalg.yield %0 : f32
  } : memref<?xf32>, memref<?xf32>
  return
}
