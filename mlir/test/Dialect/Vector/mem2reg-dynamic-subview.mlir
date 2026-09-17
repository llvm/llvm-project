// RUN: mlir-opt %s -mem2reg -split-input-file | FileCheck %s

// A static buffer read through a DYNAMIC subview with an out-of-bounds transfer
// is promoted directly by mem2reg: the written vector is threaded into the read
// and the sliced-away tail is masked in with the transfer's padding value.

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

// Write through a dynamic subview then read back: the store composes onto the
// parent within the dynamic extent (select), so a subsequent whole-buffer read
// sees the masked value.

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

// A masked read through a dynamic subview combines both masks: the subview
// extent (create_mask) AND the transfer's own mask.
// CHECK-LABEL: func.func @masked_read_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[M:.*]]: vector<8x16xi1>, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK:         %[[CM:.*]] = vector.create_mask %{{.*}}, %[[N]] : vector<8x16xi1>
// CHECK:         %[[AND:.*]] = arith.andi %[[CM]], %[[M]]
// CHECK:         %[[SEL:.*]] = arith.select %[[AND]], %[[V]], %{{.*}}
// CHECK:         return %[[SEL]]
func.func @masked_read_dyn_subview(%v: vector<8x16xf32>, %n: index, %m: vector<8x16xi1>, %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [8, %n] [1, 1] : memref<8x16xf32> to memref<8x?xf32, strided<[16, 1]>>
  %r = vector.transfer_read %sv[%c0, %c0], %pad, %m {in_bounds = [true, false]} : memref<8x?xf32, strided<[16, 1]>>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}

// -----

// A dynamic subview reshaped by `memref.collapse_shape`: the reshape keeps the
// whole parent value, so the read is the parent value shape_cast to the collapsed
// shape, masked by the subview extent -- the mask is built in the parent's shape
// and shape_cast along with the value (both describe the same elements).
// CHECK-LABEL: func.func @collapse_shape_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<8x16xf32>, %[[N:.*]]: index, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.collapse_shape
// CHECK:         %[[SC:.*]] = vector.shape_cast %[[V]] : vector<8x16xf32> to vector<128xf32>
// CHECK:         %[[CM:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<8x16xi1>
// CHECK:         %[[MASK:.*]] = vector.shape_cast %[[CM]] : vector<8x16xi1> to vector<128xi1>
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<128xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[SC]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @collapse_shape_dyn_subview(%v: vector<8x16xf32>, %n: index, %pad: f32) -> vector<128xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1]>>
  %flat = memref.collapse_shape %sv [[0, 1]] : memref<?x16xf32, strided<[16, 1]>> into memref<?xf32, strided<[1]>>
  %r = vector.transfer_read %flat[%c0], %pad {in_bounds = [false]} : memref<?xf32, strided<[1]>>, vector<128xf32>
  return %r : vector<128xf32>
}

// -----

// `memref.expand_shape` of a dynamic view: the dynamic result dimension is
// recovered from the parent extent it splits (16 = ? x 2 gives 8), so the alias
// is `vector<8x2xf32>` and the mask is reshaped alongside the value.
// CHECK-LABEL: func.func @expand_shape_dyn_subview(
// CHECK-SAME:      %[[V:.*]]: vector<16xf32>, %[[M:.*]]: index, %[[PAD:.*]]: f32
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.expand_shape
// CHECK:         %[[SC:.*]] = vector.shape_cast %[[V]] : vector<16xf32> to vector<8x2xf32>
// CHECK:         %[[CM:.*]] = vector.create_mask %[[M]] : vector<16xi1>
// CHECK:         %[[MASK:.*]] = vector.shape_cast %[[CM]] : vector<16xi1> to vector<8x2xi1>
// CHECK:         %[[PS:.*]] = vector.broadcast %[[PAD]] : f32 to vector<8x2xf32>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[SC]], %[[PS]]
// CHECK:         return %[[SEL]]
func.func @expand_shape_dyn_subview(%v: vector<16xf32>, %m: index, %pad: f32) -> vector<8x2xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<16xf32>
  vector.transfer_write %v, %a[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
  %sv = memref.subview %a[0] [%m] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %exp = memref.expand_shape %sv [[0, 1]] output_shape [%m, 2] : memref<?xf32, strided<[1]>> into memref<?x2xf32>
  %r = vector.transfer_read %exp[%c0, %c0], %pad {in_bounds = [false, true]} : memref<?x2xf32>, vector<8x2xf32>
  return %r : vector<8x2xf32>
}

// -----

// Negative: two dynamic dimensions in one reassociation group -- the alias shape
// is not determined by the parent extent, so the buffer is not promoted.
// CHECK-LABEL: func.func @neg_two_dynamic_dims_in_group(
// CHECK:         memref.alloca
// CHECK:         memref.expand_shape
// CHECK:         vector.transfer_read
func.func @neg_two_dynamic_dims_in_group(%v: vector<16xf32>, %m: index, %d0: index,
                                         %d1: index, %pad: f32) -> vector<8x2xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<16xf32>
  vector.transfer_write %v, %a[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
  %sv = memref.subview %a[0] [%m] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
  %exp = memref.expand_shape %sv [[0, 1]] output_shape [%d0, %d1] : memref<?xf32, strided<[1]>> into memref<?x?xf32>
  %r = vector.transfer_read %exp[%c0, %c0], %pad : memref<?x?xf32>, vector<8x2xf32>
  return %r : vector<8x2xf32>
}

// -----

// Negative: a non-unit dimension precedes the dynamic one in its group, so the
// view and the alias disagree on that dimension's stride -- the view strides by
// the runtime %m, while an alias shaped from the parent strides by 8 / 2 = 4.
// Promoting would move data rather than merely mask it: for a buffer holding
// 1..8 with %n = 4 (so %m = 2), the view reads
// ((1, 2, pad, pad), (3, 4, pad, pad)) whereas a masked shape_cast of the parent
// value yields ((1, 2, 3, 4), (pad, pad, pad, pad)).
// CHECK-LABEL: func.func @neg_nonunit_dim_before_dynamic(
// CHECK:         memref.alloca
// CHECK:         memref.subview
// CHECK:         memref.expand_shape
// CHECK:         vector.transfer_read
func.func @neg_nonunit_dim_before_dynamic(%v: vector<8xf32>, %n: index, %m: index,
                                          %pad: f32) -> vector<2x4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %v, %a[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  %sv = memref.subview %a[0] [%n] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>
  %exp = memref.expand_shape %sv [[0, 1]] output_shape [2, %m] : memref<?xf32, strided<[1]>> into memref<2x?xf32>
  %r = vector.transfer_read %exp[%c0, %c0], %pad {in_bounds = [true, false]} : memref<2x?xf32>, vector<2x4xf32>
  return %r : vector<2x4xf32>
}

// -----

// The reported bufferization pattern: a padded static buffer, a dynamic subview
// of its real region, expanded to a higher rank (with a dynamic dim), copied out
// to another buffer. The whole chain promotes: the parent value is shape_cast to
// the expanded shape and written masked to the copy's valid region.
// CHECK-LABEL: func.func @expand_shape_dyn_subview_copy_out(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<128x64xf16>, %[[N:[a-z0-9_]+]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.expand_shape
// CHECK-NOT:     memref.copy
// CHECK:         %[[SC:.*]] = vector.shape_cast %[[V]] : vector<128x64xf16> to vector<1x1x128x64xf16>
// CHECK:         %[[CM:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<128x64xi1>
// CHECK:         %[[MASK:.*]] = vector.shape_cast %[[CM]] : vector<128x64xi1> to vector<1x1x128x64xi1>
// CHECK:         vector.transfer_write %[[SC]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], %[[MASK]] {in_bounds = [true, true, false, true]} : vector<1x1x128x64xf16>, memref<1x1x?x64xf16, {{.*}}>
func.func @expand_shape_dyn_subview_copy_out(%v: vector<128x64xf16>, %n: index,
    %dst: memref<?x16x?x64xf16>, %i: index, %j: index, %k: index) {
  %c0 = arith.constant 0 : index
  %buf = memref.alloca() : memref<128x64xf16>
  vector.transfer_write %v, %buf[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf16>, memref<128x64xf16>
  %sv = memref.subview %buf[0, 0] [%n, 64] [1, 1]
      : memref<128x64xf16> to memref<?x64xf16, strided<[64, 1]>>
  %ex = memref.expand_shape %sv [[0, 1, 2], [3]] output_shape [1, 1, %n, 64]
      : memref<?x64xf16, strided<[64, 1]>> into memref<1x1x?x64xf16, strided<[?, ?, 64, 1]>>
  %dsv = memref.subview %dst[%i, %j, %k, 0] [1, 1, %n, 64] [1, 1, 1, 1]
      : memref<?x16x?x64xf16> to memref<1x1x?x64xf16, strided<[?, ?, 64, 1], offset: ?>>
  memref.copy %ex, %dsv : memref<1x1x?x64xf16, strided<[?, ?, 64, 1]>> to memref<1x1x?x64xf16, strided<[?, ?, 64, 1], offset: ?>>
  return
}

// -----

// The same chain with the copy reversed -- loading a tile from the other buffer
// into the staging buffer through the expanded dynamic view. The copy becomes a
// read of the source's slice, reshaped back and composed onto the buffer's value
// over the subview extent. The read carries a padding operand and is NOT in
// bounds on the dynamic dimension, which is what makes reading past the source's
// extent defined; those padded lanes are discarded by the select, so the lanes
// outside the extent keep the buffer's previous value.
// CHECK-LABEL: func.func @expand_shape_dyn_subview_copy_in(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<128x64xf16>, %[[N:[a-z0-9_]+]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.expand_shape
// CHECK-NOT:     memref.copy
// CHECK:         %[[SSV:.*]] = memref.subview %{{.*}} : memref<?x16x?x64xf16> to memref<1x1x?x64xf16, {{.*}}>
// CHECK-DAG:     %[[PAD:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:         %[[RD:.*]] = vector.transfer_read %[[SSV]][%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], %[[PAD]] {in_bounds = [true, true, false, true]} : memref<1x1x?x64xf16, {{.*}}>, vector<1x1x128x64xf16>
// CHECK:         %[[SC:.*]] = vector.shape_cast %[[RD]] : vector<1x1x128x64xf16> to vector<128x64xf16>
// CHECK:         %[[CM:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<128x64xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[CM]], %[[SC]], %[[V]]
// CHECK:         return %[[SEL]]
func.func @expand_shape_dyn_subview_copy_in(%v: vector<128x64xf16>, %n: index,
    %src: memref<?x16x?x64xf16>, %i: index, %j: index, %k: index, %pad: f16) -> vector<128x64xf16> {
  %c0 = arith.constant 0 : index
  %buf = memref.alloca() : memref<128x64xf16>
  vector.transfer_write %v, %buf[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf16>, memref<128x64xf16>
  %sv = memref.subview %buf[0, 0] [%n, 64] [1, 1]
      : memref<128x64xf16> to memref<?x64xf16, strided<[64, 1]>>
  %ex = memref.expand_shape %sv [[0, 1, 2], [3]] output_shape [1, 1, %n, 64]
      : memref<?x64xf16, strided<[64, 1]>> into memref<1x1x?x64xf16, strided<[?, ?, 64, 1]>>
  %ssv = memref.subview %src[%i, %j, %k, 0] [1, 1, %n, 64] [1, 1, 1, 1]
      : memref<?x16x?x64xf16> to memref<1x1x?x64xf16, strided<[?, ?, 64, 1], offset: ?>>
  memref.copy %ssv, %ex : memref<1x1x?x64xf16, strided<[?, ?, 64, 1], offset: ?>> to memref<1x1x?x64xf16, strided<[?, ?, 64, 1]>>
  %r = vector.transfer_read %buf[%c0, %c0], %pad {in_bounds = [true, true]} : memref<128x64xf16>, vector<128x64xf16>
  return %r : vector<128x64xf16>
}

// -----

// A dynamic view nested on a static subview: the two aliases compose, and the
// mask is built in the extent the subview's sizes are relative to -- the static
// subview's shape (4x16), not the buffer's (8x16).

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

// Two reshapes compose over a dynamic view. The value is projected one alias at
// a time, so it picks up a shape_cast per reshape (8x16 -> 128 -> 64x2). The
// mask is not projected: it is built in the parent's shape where it is needed
// and cast once to the final shape, which is equivalent because shape_casts
// preserve row-major order and therefore compose.

// CHECK-LABEL: func.func @collapse_then_expand_dyn_subview(
// CHECK-SAME:      %[[V:[a-z0-9_]+]]: vector<8x16xf32>, %[[N:[a-z0-9_]+]]: index
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.collapse_shape
// CHECK-NOT:     memref.expand_shape
// CHECK:         %[[F:.*]] = vector.shape_cast %[[V]] : vector<8x16xf32> to vector<128xf32>
// CHECK:         %[[E:.*]] = vector.shape_cast %[[F]] : vector<128xf32> to vector<64x2xf32>
// CHECK:         %[[CM:.*]] = vector.create_mask %[[N]], %{{.*}} : vector<8x16xi1>
// CHECK:         %[[MASK:.*]] = vector.shape_cast %[[CM]] : vector<8x16xi1> to vector<64x2xi1>
// CHECK:         %[[SEL:.*]] = arith.select %[[MASK]], %[[E]], %{{.*}}
// CHECK:         return %[[SEL]]
func.func @collapse_then_expand_dyn_subview(%v: vector<8x16xf32>, %n: index,
                                            %d: index, %pad: f32) -> vector<64x2xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1]>>
  %f = memref.collapse_shape %sv [[0, 1]] : memref<?x16xf32, strided<[16, 1]>> into memref<?xf32, strided<[1]>>
  %e = memref.expand_shape %f [[0, 1]] output_shape [%d, 2] : memref<?xf32, strided<[1]>> into memref<?x2xf32>
  %r = vector.transfer_read %e[%c0, %c0], %pad {in_bounds = [false, true]} : memref<?x2xf32>, vector<64x2xf32>
  return %r : vector<64x2xf32>
}

// -----

// The promoted value flows across control flow: the masked store on one path is
// merged with the untouched value on the other by a block argument. The mask is
// built where the view is, so it is only computed on the path that stores.

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

// -----

// Negative: the collapsed group's dynamic extent is not its outermost, so the
// group's stride would have to be that runtime extent. This is the collapse
// counterpart of @neg_nonunit_dim_before_dynamic, rejected one step earlier:
// such a collapse is not guaranteed contiguous, so its result stride is dynamic.
// CHECK-LABEL: func.func @neg_collapse_dyn_inner_dim(
// CHECK:         memref.alloca
// CHECK:         memref.collapse_shape
// CHECK:         vector.transfer_read
func.func @neg_collapse_dyn_inner_dim(%v: vector<4x4xf32>, %n: index,
                                      %pad: f32) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<4x4xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<4x4xf32>
  %sv = memref.subview %a[0, 0] [4, %n] [1, 1] : memref<4x4xf32> to memref<4x?xf32, strided<[4, 1]>>
  %f = memref.collapse_shape %sv [[0, 1]] : memref<4x?xf32, strided<[4, 1]>> into memref<?xf32, strided<[?]>>
  %r = vector.transfer_read %f[%c0], %pad {in_bounds = [false]} : memref<?xf32, strided<[?]>>, vector<16xf32>
  return %r : vector<16xf32>
}

// -----

// Negative, and a crash regression: a reassociation group may hold a zero static
// extent, which the parent extent cannot be divided by. The buffer is not
// promoted; without the guard the division faults.
// CHECK-LABEL: func.func @neg_zero_extent_in_group(
// CHECK:         memref.alloca
// CHECK:         memref.subview
// CHECK:         memref.expand_shape
// CHECK:         vector.transfer_read
func.func @neg_zero_extent_in_group(%v: vector<8x16xf32>, %n: index, %d: index,
                                    %pad: f32) -> vector<8x16xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8x16xf32>
  vector.transfer_write %v, %a[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<8x16xf32>
  %sv = memref.subview %a[0, 0] [%n, 16] [1, 1] : memref<8x16xf32> to memref<?x16xf32, strided<[16, 1]>>
  %e = memref.expand_shape %sv [[0, 1], [2]] output_shape [%d, 0, 16] : memref<?x16xf32, strided<[16, 1]>> into memref<?x0x16xf32, strided<[0, 16, 1]>>
  %r = vector.transfer_read %e[%c0, %c0, %c0], %pad : memref<?x0x16xf32, strided<[0, 16, 1]>>, vector<8x16xf32>
  return %r : vector<8x16xf32>
}
