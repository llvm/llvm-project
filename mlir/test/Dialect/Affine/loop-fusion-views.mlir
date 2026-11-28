// RUN: mlir-opt %s -affine-loop-fusion -split-input-file | FileCheck %s

// Accesses through views (memref.subview, memref.cast, ...) of a memref written
// by a loop nest must keep their order relative to that loop nest.

// CHECK-LABEL: func @load_through_subview_between_loops
func.func @load_through_subview_between_loops(%in: memref<4xf32>, %buf: memref<4xf32>, %out: memref<4xf32>) -> f32 {
  affine.for %i = 0 to 4 {
    %v = affine.load %in[%i] : memref<4xf32>
    affine.store %v, %buf[%i] : memref<4xf32>
  }
  %c0 = arith.constant 0 : index
  %sv = memref.subview %buf[1] [2] [1] : memref<4xf32> to memref<2xf32, strided<[1], offset: 1>>
  %x = memref.load %sv[%c0] : memref<2xf32, strided<[1], offset: 1>>
  affine.for %i = 0 to 4 {
    %v = affine.load %buf[%i] : memref<4xf32>
    %w = arith.addf %v, %v : f32
    affine.store %w, %out[%i] : memref<4xf32>
  }
  return %x : f32
}
// The fused loop nest must be placed before the load through the subview.
// CHECK:       affine.for
// CHECK:       }
// CHECK-NOT:   affine.for
// CHECK:       memref.load
// CHECK-NOT:   affine.for
// CHECK:       return

// -----

// CHECK-LABEL: func @store_through_subview_between_loops
func.func @store_through_subview_between_loops(%in: memref<4xf32>, %buf: memref<4xf32>, %out: memref<4xf32>, %x: f32) {
  affine.for %i = 0 to 4 {
    %v = affine.load %in[%i] : memref<4xf32>
    affine.store %v, %buf[%i] : memref<4xf32>
  }
  %c0 = arith.constant 0 : index
  %sv = memref.subview %buf[1] [2] [1] : memref<4xf32> to memref<2xf32, strided<[1], offset: 1>>
  memref.store %x, %sv[%c0] : memref<2xf32, strided<[1], offset: 1>>
  affine.for %i = 0 to 4 {
    %v = affine.load %buf[%i] : memref<4xf32>
    affine.store %v, %out[%i] : memref<4xf32>
  }
  return
}
// The store through the subview must stay between the two loop nests.
// CHECK:       affine.for
// CHECK:       memref.store
// CHECK:       affine.for

// -----

// CHECK-LABEL: func @load_through_cast_and_subview_between_loops
func.func @load_through_cast_and_subview_between_loops(%in: memref<4xf32>, %buf: memref<4xf32>, %out: memref<4xf32>) -> f32 {
  affine.for %i = 0 to 4 {
    %v = affine.load %in[%i] : memref<4xf32>
    affine.store %v, %buf[%i] : memref<4xf32>
  }
  %c0 = arith.constant 0 : index
  %cast = memref.cast %buf : memref<4xf32> to memref<?xf32>
  %sv = memref.subview %cast[1] [2] [1] : memref<?xf32> to memref<2xf32, strided<[1], offset: 1>>
  %x = memref.load %sv[%c0] : memref<2xf32, strided<[1], offset: 1>>
  affine.for %i = 0 to 4 {
    %v = affine.load %buf[%i] : memref<4xf32>
    %w = arith.addf %v, %v : f32
    affine.store %w, %out[%i] : memref<4xf32>
  }
  return %x : f32
}
// Chained views are traced back to the base memref.
// CHECK:       affine.for
// CHECK:       }
// CHECK-NOT:   affine.for
// CHECK:       memref.load
// CHECK-NOT:   affine.for
// CHECK:       return

// -----

// Sibling loops reading %buf must not be fused across a store through a view
// of %buf.
// CHECK-LABEL: func @sibling_loops_across_store_through_subview
func.func @sibling_loops_across_store_through_subview(%buf: memref<4xf32>, %out0: memref<4xf32>, %out1: memref<4xf32>, %x: f32) {
  affine.for %i = 0 to 4 {
    %v = affine.load %buf[%i] : memref<4xf32>
    affine.store %v, %out0[%i] : memref<4xf32>
  }
  %c0 = arith.constant 0 : index
  %sv = memref.subview %buf[1] [2] [1] : memref<4xf32> to memref<2xf32, strided<[1], offset: 1>>
  memref.store %x, %sv[%c0] : memref<2xf32, strided<[1], offset: 1>>
  affine.for %i = 0 to 4 {
    %v = affine.load %buf[%i] : memref<4xf32>
    affine.store %v, %out1[%i] : memref<4xf32>
  }
  return
}
// CHECK:       affine.for
// CHECK:       memref.store
// CHECK:       affine.for