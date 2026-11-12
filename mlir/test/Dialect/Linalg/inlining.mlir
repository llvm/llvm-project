// RUN: mlir-opt %s -inline | FileCheck %s

// These tests verify that regions with operations from Lingalg dialect
// can be inlined.

#accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

func.func @inline_into(%arg0: memref<?xf32>) {
  // CHECK: linalg.generic
  call @inlined_fn(%arg0) : (memref<?xf32>) -> ()
  return
}

func.func @inlined_fn(%arg0: memref<?xf32>) {
  // CHECK: linalg.generic
  linalg.generic #trait
     ins(%arg0 : memref<?xf32>)
    outs(%arg0 : memref<?xf32>) {
    ^bb(%0 : f32, %1 : f32) :
      %2 = arith.addf %0, %0: f32
      linalg.yield %2 : f32
  }
  return
}
