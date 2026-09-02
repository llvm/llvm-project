// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=4" | FileCheck %s

// This loop has a carried dependence through a view alias: iteration i stores
// to %arg0[i + 1] via %reinterpret_cast, and iteration i + 1 reads that cell
// through %arg0. It must not be vectorized as independent lanes.

// CHECK-LABEL: func.func @view_alias_carried_dependence_unsupported
// CHECK:         memref.reinterpret_cast
// CHECK:         affine.for
// CHECK:           affine.load
// CHECK:           affine.store
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
func.func @view_alias_carried_dependence_unsupported(%arg0: memref<8xi32>) {
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4], strides: [1] : memref<8xi32> to memref<4xi32, strided<[1], offset: 0>>
  affine.for %i = 0 to 3 {
    %0 = affine.load %arg0[%i] : memref<8xi32>
    %1 = arith.addi %0, %0 : i32
    affine.store %1, %reinterpret_cast[%i + 1] : memref<4xi32, strided<[1], offset: 0>>
  }
  return
}
