// RUN: mlir-opt %s -memref-copy-to-contiguous -split-input-file | FileCheck %s

// A copy into a padded buffer: the trailing dimension is contiguous in both
// layouts, the leading two are not in the target. Expect a loop nest over
// the leading two dimensions around a copy of the 163-element rows.

// CHECK-LABEL: func.func @pad_activation(
//  CHECK-SAME:     %[[SRC:.*]]: memref<15x64x163xf32>,
//  CHECK-SAME:     %[[DST:.*]]: memref<15x64x163xf32, strided<[11264, 176, 1]>>)
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C15:.*]] = arith.constant 15 : index
//   CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
//       CHECK:   scf.for %[[I:.*]] = %[[C0]] to %[[C15]] step %[[C1]] {
//       CHECK:     scf.for %[[J:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
//       CHECK:       %[[S:.*]] = memref.subview %[[SRC]][%[[I]], %[[J]], 0] [1, 1, 163] [1, 1, 1]
//       CHECK:       %[[D:.*]] = memref.subview %[[DST]][%[[I]], %[[J]], 0] [1, 1, 163] [1, 1, 1]
//       CHECK:       memref.copy %[[S]], %[[D]]
//       CHECK:     }
//       CHECK:   }
//   CHECK-NOT:   memref.copy
func.func @pad_activation(%src: memref<15x64x163xf32>,
                          %dst: memref<15x64x163xf32, strided<[11264, 176, 1]>>) {
  memref.copy %src, %dst
      : memref<15x64x163xf32> to memref<15x64x163xf32, strided<[11264, 176, 1]>>
  return
}

// -----

// Both operands contiguous: nothing to do.

// CHECK-LABEL: func.func @already_contiguous(
//   CHECK-NOT:   scf.for
//       CHECK:   memref.copy %{{.*}}, %{{.*}} : memref<4x64xf32> to memref<4x64xf32>
func.func @already_contiguous(%src: memref<4x64xf32>, %dst: memref<4x64xf32>) {
  memref.copy %src, %dst : memref<4x64xf32> to memref<4x64xf32>
  return
}

// -----

// A strided layout that is still contiguous (offset only) is accepted by the
// lowering's predicate and must be left alone.

// CHECK-LABEL: func.func @contiguous_with_offset(
//   CHECK-NOT:   scf.for
//       CHECK:   memref.copy
func.func @contiguous_with_offset(%src: memref<4x64xf32>,
                                  %dst: memref<4x64xf32, strided<[64, 1], offset: ?>>) {
  memref.copy %src, %dst
      : memref<4x64xf32> to memref<4x64xf32, strided<[64, 1], offset: ?>>
  return
}

// -----

// A unit dimension with a stride that does not match the running product
// breaks the lowering's contiguity check even though it has no effect on
// memory layout. The pass must follow the lowering, not the layout: the
// suffix starts after that dimension, so the leading two dimensions are
// looped over (the second with a trip count of one).

// CHECK-LABEL: func.func @unit_dim_odd_stride(
//  CHECK-SAME:     %[[SRC:.*]]: memref<4x1x64xf32>,
//  CHECK-SAME:     %[[DST:.*]]: memref<4x1x64xf32, strided<[64, 999, 1]>>)
//       CHECK:   scf.for %[[I:.*]] =
//       CHECK:     scf.for %[[J:.*]] =
//       CHECK:       memref.subview %[[SRC]][%[[I]], %[[J]], 0] [1, 1, 64] [1, 1, 1]
//       CHECK:       memref.subview %[[DST]][%[[I]], %[[J]], 0] [1, 1, 64] [1, 1, 1]
//       CHECK:       memref.copy
func.func @unit_dim_odd_stride(%src: memref<4x1x64xf32>,
                               %dst: memref<4x1x64xf32, strided<[64, 999, 1]>>) {
  memref.copy %src, %dst
      : memref<4x1x64xf32> to memref<4x1x64xf32, strided<[64, 999, 1]>>
  return
}

// -----

// Leading dimensions that are all unit extents are accepted by the lowering's
// second phase; leave the copy alone.

// CHECK-LABEL: func.func @leading_unit_dims(
//   CHECK-NOT:   scf.for
//       CHECK:   memref.copy
func.func @leading_unit_dims(%src: memref<1x1x64xf32>,
                             %dst: memref<1x1x64xf32, strided<[4096, 512, 1]>>) {
  memref.copy %src, %dst
      : memref<1x1x64xf32> to memref<1x1x64xf32, strided<[4096, 512, 1]>>
  return
}

// -----

// Dynamic shapes are not handled.

// CHECK-LABEL: func.func @dynamic_shape(
//   CHECK-NOT:   scf.for
//       CHECK:   memref.copy
func.func @dynamic_shape(%src: memref<?x64xf32>,
                         %dst: memref<?x64xf32, strided<[176, 1]>>) {
  memref.copy %src, %dst : memref<?x64xf32> to memref<?x64xf32, strided<[176, 1]>>
  return
}

// -----

// Only the source is non-contiguous.

// CHECK-LABEL: func.func @source_strided(
//       CHECK:   scf.for
//       CHECK:     memref.subview %{{.*}}[%{{.*}}, 0] [1, 64] [1, 1]
//       CHECK:     memref.subview %{{.*}}[%{{.*}}, 0] [1, 64] [1, 1]
//       CHECK:     memref.copy
func.func @source_strided(%src: memref<8x64xf32, strided<[128, 1]>>,
                          %dst: memref<8x64xf32>) {
  memref.copy %src, %dst : memref<8x64xf32, strided<[128, 1]>> to memref<8x64xf32>
  return
}
