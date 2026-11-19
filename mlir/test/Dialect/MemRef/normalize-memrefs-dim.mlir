// RUN: mlir-opt %s -normalize-memrefs -split-input-file | FileCheck %s

// A rank-changing rewrite must not change the queried logical extent.
// CHECK-LABEL: func.func @reinterpret_changed_dim
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<3xi32>
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[ALLOC]] to offset: [0], sizes: [1, 3], strides: [3, 1] : memref<3xi32> to memref<1x3xi32, strided<[3, 1]>>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = memref.dim %[[VIEW]], %[[C0]] : memref<1x3xi32, strided<[3, 1]>>
// CHECK: return %[[DIM]] : index
func.func @reinterpret_changed_dim() -> index {
  %alloc = memref.alloc() : memref<3xi32>
  %view = memref.reinterpret_cast %alloc to offset: [0], sizes: [1, 3], strides: [3, 1] : memref<3xi32> to memref<1x3xi32, strided<[3, 1]>>
  %c0 = arith.constant 0 : index
  %dim = memref.dim %view, %c0 : memref<1x3xi32, strided<[3, 1]>>
  memref.dealloc %alloc : memref<3xi32>
  return %dim : index
}

// -----

// The queried dimension must still exist after normalization.
// CHECK-LABEL: func.func @reinterpret_removed_dim
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast {{.*}} to offset: [0], sizes: [3, 1], strides: [1, 1]
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: memref.dim %[[VIEW]], %[[C1]] : memref<3x1xi32, strided<[1, 1]>>
func.func @reinterpret_removed_dim(%source: memref<3xi32>) -> index {
  %view = memref.reinterpret_cast %source to offset: [0], sizes: [3, 1], strides: [1, 1] : memref<3xi32> to memref<3x1xi32, strided<[1, 1]>>
  %c1 = arith.constant 1 : index
  %dim = memref.dim %view, %c1 : memref<3x1xi32, strided<[1, 1]>>
  return %dim : index
}

// -----

// Even a rank-preserving rewrite can change a dimension's size.
// CHECK-LABEL: func.func @reinterpret_same_rank
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast {{.*}} to offset: [0], sizes: [4], strides: [2]
// CHECK: memref.dim %[[VIEW]], {{.*}} : memref<4xf32, strided<[2]>>
func.func @reinterpret_same_rank(%source: memref<7xf32>) -> index {
  %view = memref.reinterpret_cast %source to offset: [0], sizes: [4], strides: [2] : memref<7xf32> to memref<4xf32, strided<[2]>>
  %c0 = arith.constant 0 : index
  %dim = memref.dim %view, %c0 : memref<4xf32, strided<[2]>>
  return %dim : index
}

// -----

// A rank-changing rewrite is allowed if the queried extent stays the same.
// CHECK-LABEL: func.func @reinterpret_preserved_dim
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast {{.*}} to offset: [0], sizes: [3], strides: [1] : memref<3xi32> to memref<3xi32>
// CHECK: memref.dim %[[VIEW]], {{.*}} : memref<3xi32>
func.func @reinterpret_preserved_dim(%source: memref<3xi32>) -> index {
  %view = memref.reinterpret_cast %source to offset: [0], sizes: [3, 1], strides: [1, 1] : memref<3xi32> to memref<3x1xi32, strided<[1, 1]>>
  %c0 = arith.constant 0 : index
  %dim = memref.dim %view, %c0 : memref<3x1xi32, strided<[1, 1]>>
  return %dim : index
}

// -----

// A dynamic dimension index cannot be proven to query a preserved extent.
// CHECK-LABEL: func.func @reinterpret_dynamic_index
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast {{.*}} to offset: [0], sizes: [3, 1], strides: [1, 1]
// CHECK: memref.dim %[[VIEW]], {{.*}} : memref<3x1xi32, strided<[1, 1]>>
func.func @reinterpret_dynamic_index(%source: memref<3xi32>, %index: index) -> index {
  %view = memref.reinterpret_cast %source to offset: [0], sizes: [3, 1], strides: [1, 1] : memref<3xi32> to memref<3x1xi32, strided<[1, 1]>>
  %dim = memref.dim %view, %index : memref<3x1xi32, strided<[1, 1]>>
  return %dim : index
}

// -----

// A nonzero offset also contributes to the normalized size (155).
// CHECK-LABEL: func.func @reinterpret_offset
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast {{.*}} to offset: [100], sizes: [4, 4], strides: [16, 2]
// CHECK: memref.dim %[[VIEW]], {{.*}} : memref<4x4xf32, strided<[16, 2], offset: 100>>
func.func @reinterpret_offset(%source: memref<100x100xf32>) -> index {
  %view = memref.reinterpret_cast %source to offset: [100], sizes: [4, 4], strides: [16, 2] : memref<100x100xf32> to memref<4x4xf32, strided<[16, 2], offset: 100>>
  %c0 = arith.constant 0 : index
  %dim = memref.dim %view, %c0 : memref<4x4xf32, strided<[16, 2], offset: 100>>
  return %dim : index
}
