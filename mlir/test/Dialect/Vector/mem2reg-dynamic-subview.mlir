// RUN: mlir-opt %s -mem2reg -split-input-file | FileCheck %s

// Reading a static buffer through a DYNAMIC subview: the read returns the value
// that was written, and takes the transfer's padding wherever it reads past the
// subview's extent.

// CHECK-LABEL: func.func @read_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.subview
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         %[[MASK:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<8x16xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[V]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @read_dyn_subview(%v: vector<8x16xf32>, %n: index, %pad: f32)
    -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]}
      : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1]
      : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %r = vector.transfer_read %sv[%c0, %c0], %pad {in_bounds = [true, false]}
      : memref<8x?xf32, strided<[16, 1]>>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// Write through a dynamic subview, then read the whole buffer. The write updates
// only the elements within the dynamic extent, so the read returns the written
// value for those and the buffer's earlier value for the rest.

// CHECK-LABEL: func.func @write_then_read_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[W:.*]]: vector<8x16xf32>, %[[N:.*]]: index
// CHECK-NOT:     memref.alloca
// CHECK:         %[[MASK:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[W]], %[[V]]
// CHECK:         return %[[SEL]]
func.func @write_then_read_dyn_subview(%v: vector<8x16xf32>, %w: vector<8x16xf32>,
                                       %n: index, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]}
      : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1]
      : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  vector.transfer_write %w, %sv[%c0, %c0] {in_bounds = [true, false]}
      : vector<8x16xf32>, memref<8x?xf32, strided<[16, 1]>>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]}
      : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// A read with a mask through a dynamic subview combines both masks: the subview
// extent (create_mask) AND the transfer's own mask operand.
// CHECK-LABEL: func.func @read_with_mask_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[M:.*]]: vector<8x16xi1>, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK:         %[[CM:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[AND:.*]] = arith.andi %[[CM]], %[[M]]
// CHECK:         %[[SEL:.*]] = arith.select %[[AND]], %[[V]], %{{.*}}
// CHECK:         return %[[SEL]]
func.func @read_with_mask_dyn_subview(%v: vector<8x16xf32>, %n: index, %m: vector<8x16xi1>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %r = vector.transfer_read %sv[%c0, %c0], %pad, %m {in_bounds = [true, false]} : memref<8x?xf32, strided<[16, 1]>>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// A dynamic subview of a static subview: both views apply, and the mask uses
// the shape the subview's sizes are relative to (4x16, not the buffer's 8x16).

// CHECK-LABEL: func.func @dyn_subview_of_static_subview(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<8x16xf32>, %[[N:[a-z0-9_]+]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.subview
// CHECK:         %[[EX:.*]] = vector.extract_strided_slice %[[V]] offsets = [2, 0], sizes = [4, 16]
// CHECK:         %[[CM:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<4x16xi1>
// CHECK:         %[[PS:.*]] = vector.broadcast
// CHECK:         %[[SEL:.*]] = arith.select %[[CM]], %[[EX]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @dyn_subview_of_static_subview(%v: vector<8x16xf32>, %n: index,
                                         %pad: f32) -> vector<4x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv1 = memref.subview %a[2, 0] [4, 16] [1, 1] : memref<8x16xf32> to memref<4x16xf32, strided<[16, 1], offset: 32>>
  %sv2 = memref.subview %sv1[0, 0] [%n, 16] [1, 1] : memref<4x16xf32, strided<[16, 1], offset: 32>> to memref<?x16xf32, strided<[16, 1], offset: 32>>
  %r = vector.transfer_read %sv2[%c0, %c0], %pad {in_bounds = [false, true]} : memref<?x16xf32, strided<[16, 1], offset: 32>>, vector<4x16xf32>
  return %r : vector<4x16xf32>
}

// -----

// The promoted value flows across control flow: a block argument merges the
// masked store on one path with the untouched value on the other.

// CHECK-LABEL: func.func @dyn_subview_across_cfg(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<8x16xf32>, %[[W:[a-z0-9_]+]]: vector<8x16xf32>, %[[N:[a-z0-9_]+]]: index
// CHECK-NOT:     memref.alloca
// CHECK:         cf.cond_br %{{.*}}, ^[[BB1:.*]], ^[[BB2:.*]]
// CHECK:       ^[[BB1]]:
// CHECK:         %[[CM:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<8x16xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[CM]], %[[W]], %[[V]]
// CHECK:         cf.br ^[[BB3:.*]](%[[SEL]] : vector<8x16xf32>)
// CHECK:       ^[[BB2]]:
// CHECK:         cf.br ^[[BB3]](%[[V]] : vector<8x16xf32>)
// CHECK:       ^[[BB3]](%[[PHI:.*]]: vector<8x16xf32>):
// CHECK:         return %[[PHI]]
func.func @dyn_subview_across_cfg(%v: vector<8x16xf32>, %w: vector<8x16xf32>,
                                  %n: index, %pad: f32, %c: i1) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  cf.cond_br %c, ^bb1, ^bb2
^bb1:
  %sv = memref.subview %a[0, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1]>>
  vector.transfer_write %w, %sv[%c0, %c0] {in_bounds = [false, true]} : vector<8x16xf32>, memref<?x16xf32, strided<[16, 1]>>
  cf.br ^bb3
^bb2:
  cf.br ^bb3
^bb3:
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<8x16xf32>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// Negative: a dynamic OFFSET (not just size) subview is not promotable; the
// buffer is left alone.

// CHECK-LABEL: func.func @neg_dynamic_offset(
// CHECK:         memref.alloca
// CHECK:         memref.subview
// CHECK:         vector.transfer_read
func.func @neg_dynamic_offset(%v: vector<8x16xf32>, %off: index, %n: index,
                              %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]}
      : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, %off] [8, %n] [1, 1]
      : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1], offset: ?>>
  %r = vector.transfer_read %sv[%c0, %c0], %pad {in_bounds = [true, false]}
      : memref<8x?xf32, strided<[16, 1], offset: ?>>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// Negative: a dynamic size with a static but non-zero offset. The valid region
// no longer starts at the parent's origin, which `vector.create_mask` cannot
// express, so the buffer is not promoted.
// CHECK-LABEL: func.func @neg_static_offset_dyn_size(
// CHECK:         memref.alloca
// CHECK:         memref.subview
// CHECK:         vector.transfer_read
func.func @neg_static_offset_dyn_size(%v: vector<8x16xf32>, %n: index,
                                      %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[2, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1], offset: 32>>
  %r = vector.transfer_read %sv[%c0, %c0], %pad {in_bounds = [false, true]} : memref<?x16xf32, strided<[16, 1], offset: 32>>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}
