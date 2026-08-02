// RUN: mlir-opt %s --mem2reg --split-input-file | FileCheck %s

// A memref that is only ever accessed as a whole buffer through
// vector.transfer_read / vector.transfer_write is promoted to a single vector
// SSA value.

// CHECK-LABEL: func.func @whole_buffer_write_read
//   CHECK-SAME:   (%[[PAD:.*]]: f32)
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   vector.transfer_write
//    CHECK-NOT:   vector.transfer_read
//        CHECK:   %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
//        CHECK:   return %[[CST]] : vector<4xf32>
func.func @whole_buffer_write_read(%pad: f32) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<4xf32>
  %a = memref.alloca() : memref<4xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A whole-buffer slot carried across scf.for is threaded as an iter_arg/result.

// CHECK-LABEL: func.func @whole_buffer_in_loop
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   vector.transfer_write
//    CHECK-NOT:   vector.transfer_read
//        CHECK:   %[[RES:.*]] = scf.for {{.*}} iter_args(%[[IT:.*]] = %{{.*}}) -> (vector<4xf32>)
//        CHECK:     %[[NEXT:.*]] = arith.addf %[[IT]], %[[IT]] : vector<4xf32>
//        CHECK:     scf.yield %[[NEXT]] : vector<4xf32>
//        CHECK:   return %[[RES]] : vector<4xf32>
func.func @whole_buffer_in_loop(%pad: f32, %lb: index, %ub: index, %step: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<4xf32>
  %a = memref.alloca() : memref<4xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  scf.for %i = %lb to %ub step %step {
    %v = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
    %n = arith.addf %v, %v : vector<4xf32>
    vector.transfer_write %n, %a[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  }
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A multi-dimensional whole-buffer memref is promoted to a matching vector.

// CHECK-LABEL: func.func @whole_buffer_2d
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   vector.transfer
//        CHECK:   return %{{.*}} : vector<2x4xf32>
func.func @whole_buffer_2d(%pad: f32) -> vector<2x4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<2x4xf32>
  %a = memref.alloca() : memref<2x4xf32>
  vector.transfer_write %cst, %a[%c0, %c0] {in_bounds = [true, true]} : vector<2x4xf32>, memref<2x4xf32>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<2x4xf32>, vector<2x4xf32>
  return %r : vector<2x4xf32>
}

// -----

// Non-zero access offset: the transfer does not cover the whole buffer, so the
// slot must NOT be promoted.

// CHECK-LABEL: func.func @negative_nonzero_index
//        CHECK:   memref.alloca
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
func.func @negative_nonzero_index(%pad: f32) -> vector<4xf32> {
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<1.0> : vector<4xf32>
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %cst, %a[%c1] {in_bounds = [true]} : vector<4xf32>, memref<8xf32>
  %r = vector.transfer_read %a[%c1], %pad {in_bounds = [true]} : memref<8xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A masked transfer only touches part of the buffer: must NOT be promoted.

// CHECK-LABEL: func.func @negative_masked
//        CHECK:   memref.alloca
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
func.func @negative_masked(%pad: f32, %m: vector<4xi1>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<4xf32>
  %a = memref.alloca() : memref<4xf32>
  vector.transfer_write %cst, %a[%c0], %m {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A partial (out-of-bounds) transfer must NOT be promoted.

// CHECK-LABEL: func.func @negative_out_of_bounds
//        CHECK:   memref.alloca
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
func.func @negative_out_of_bounds(%pad: f32) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<8xf32>
  %a = memref.alloca() : memref<4xf32>
  vector.transfer_write %cst, %a[%c0] : vector<8xf32>, memref<4xf32>
  %r = vector.transfer_read %a[%c0], %pad : memref<4xf32>, vector<8xf32>
  return %r : vector<8xf32>
}

// -----

// A non-identity (transposing) permutation map is not a whole-buffer identity
// access: must NOT be promoted.

// CHECK-LABEL: func.func @negative_transpose_map
//        CHECK:   memref.alloca
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
func.func @negative_transpose_map(%pad: f32) -> vector<4x2xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<2x4xf32>
  %a = memref.alloca() : memref<2x4xf32>
  vector.transfer_write %cst, %a[%c0, %c0] {in_bounds = [true, true]} : vector<2x4xf32>, memref<2x4xf32>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2x4xf32>, vector<4x2xf32>
  return %r : vector<4x2xf32>
}

// -----

// An alloca also accessed through a scalar memref.load cannot be promoted to a
// vector: must NOT be promoted.

// CHECK-LABEL: func.func @negative_mixed_scalar_access
//        CHECK:   memref.alloca
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
//        CHECK:   memref.load
func.func @negative_mixed_scalar_access(%pad: f32) -> (vector<4xf32>, f32) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<4xf32>
  %a = memref.alloca() : memref<4xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  %s = memref.load %a[%c0] : memref<4xf32>
  return %r, %s : vector<4xf32>, f32
}

// -----

// A scalable vector never equals the fixed-size slot type: must NOT be
// promoted.

// CHECK-LABEL: func.func @negative_scalable
//        CHECK:   memref.alloca
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
func.func @negative_scalable(%pad: f32) -> vector<[4]xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<[4]xf32>
  %a = memref.alloca() : memref<4xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<[4]xf32>, memref<4xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<[4]xf32>
  return %r : vector<[4]xf32>
}

// -----

// -----

// A dynamic-shape memref never yields a promotable slot (its extents are not
// known statically, so it cannot map to a fixed-size vector): must NOT be
// promoted.

// CHECK-LABEL: func.func @negative_dynamic_shape
//        CHECK:   memref.alloca
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
func.func @negative_dynamic_shape(%pad: f32, %d: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<4xf32>
  %a = memref.alloca(%d) : memref<?xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<?xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A static, same-rank memref.subview is a promotable sub-slice alias: a write
// into the subview becomes vector.insert_strided_slice into the buffer's value,
// and the whole-buffer read returns that composed value.

// CHECK-LABEL: func.func @subview_static_write
//   CHECK-SAME:   (%[[V:.*]]: vector<4xf32>, %[[INIT:.*]]: vector<8xf32>, %[[PAD:.*]]: f32)
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.subview
//    CHECK-NOT:   vector.transfer_write
//    CHECK-NOT:   vector.transfer_read
//        CHECK:   %[[INS:.*]] = vector.insert_strided_slice %[[V]], %[[INIT]] {offsets = [2], strides = [1]}
//        CHECK:   return %[[INS]] : vector<8xf32>
func.func @subview_static_write(%v: vector<4xf32>, %init: vector<8xf32>, %pad: f32) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %init, %a[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  %sv = memref.subview %a[2] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 2>>
  vector.transfer_write %v, %sv[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1], offset: 2>>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
  return %r : vector<8xf32>
}

// -----

// A read of a static subview becomes vector.extract_strided_slice of the value.

// CHECK-LABEL: func.func @subview_static_read
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.subview
//        CHECK:   %[[EXT:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2], sizes = [4], strides = [1]}
//        CHECK:   return %[[EXT]] : vector<4xf32>
func.func @subview_static_read(%init: vector<8xf32>, %pad: f32) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %init, %a[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  %sv = memref.subview %a[2] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 2>>
  %r = vector.transfer_read %sv[%c0], %pad {in_bounds = [true]} : memref<4xf32, strided<[1], offset: 2>>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A buffer accessed *only* through subviews (no whole-buffer transfer) still
// promotes: the slot comes from the alloca, not from any transfer. The memref,
// the subviews, and the transfers are all removed; the written slice is
// inserted into the buffer value and read back out. (canonicalize/cse would
// then fold this to a plain forward of the written vector.)

// CHECK-LABEL: func.func @subview_only_write_read
//   CHECK-SAME:   (%[[V:.*]]: vector<4xf32>, %[[PAD:.*]]: f32)
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.subview
//    CHECK-NOT:   vector.transfer_write
//    CHECK-NOT:   vector.transfer_read
//        CHECK:   vector.insert_strided_slice %[[V]], %{{.*}} {offsets = [0], strides = [1]}
func.func @subview_only_write_read(%v: vector<4xf32>, %pad: f32) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  %svW = memref.subview %a[0] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1]>>
  vector.transfer_write %v, %svW[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1]>>
  %svR = memref.subview %a[0] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1]>>
  %r = vector.transfer_read %svR[%c0], %pad {in_bounds = [true]} : memref<4xf32, strided<[1]>>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// Two disjoint subview writes that together cover the buffer, followed by a
// subview read spanning the boundary between them. The read composes both
// writes through the parent value: it returns the low half of the first write
// and the high half of the second (i.e. insert both slices, then extract).

// CHECK-LABEL: func.func @subview_disjoint_writes_boundary_read
//   CHECK-SAME:   (%[[VA:.*]]: vector<4xf32>, %[[VB:.*]]: vector<4xf32>, %[[PAD:.*]]: f32)
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.subview
//        CHECK:   %[[D0:.*]] = vector.insert_strided_slice %[[VA]], %{{.*}} {offsets = [0], strides = [1]}
//        CHECK:   %[[D1:.*]] = vector.insert_strided_slice %[[VB]], %[[D0]] {offsets = [4], strides = [1]}
//        CHECK:   %[[R:.*]] = vector.extract_strided_slice %[[D1]] {offsets = [2], sizes = [4], strides = [1]}
//        CHECK:   return %[[R]] : vector<4xf32>
func.func @subview_disjoint_writes_boundary_read(%vA: vector<4xf32>, %vB: vector<4xf32>, %pad: f32) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  %s0 = memref.subview %a[0] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1]>>
  %s4 = memref.subview %a[4] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 4>>
  %s2 = memref.subview %a[2] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 2>>
  vector.transfer_write %vA, %s0[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1]>>
  vector.transfer_write %vB, %s4[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1], offset: 4>>
  %r = vector.transfer_read %s2[%c0], %pad {in_bounds = [true]} : memref<4xf32, strided<[1], offset: 2>>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// Reads and writes may occur in any order: a read of the subview before any
// write returns the slot's default value (ub.poison), matching the semantics of
// reading an uninitialized alloca. A later read observes the written value.

// CHECK-LABEL: func.func @subview_read_before_write
//   CHECK-SAME:   (%[[V:.*]]: vector<4xf32>, %[[PAD:.*]]: f32)
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.subview
//        CHECK:   %[[POISON:.*]] = ub.poison : vector<8xf32>
//        CHECK:   %[[R0:.*]] = vector.extract_strided_slice %[[POISON]] {offsets = [0], sizes = [4], strides = [1]}
//        CHECK:   %[[INS:.*]] = vector.insert_strided_slice %[[V]], %[[POISON]] {offsets = [0], strides = [1]}
//        CHECK:   %[[R1:.*]] = vector.extract_strided_slice %[[INS]] {offsets = [0], sizes = [4], strides = [1]}
//        CHECK:   return %[[R0]], %[[R1]] : vector<4xf32>, vector<4xf32>
func.func @subview_read_before_write(%v: vector<4xf32>, %pad: f32) -> (vector<4xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  %s = memref.subview %a[0] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1]>>
  %r0 = vector.transfer_read %s[%c0], %pad {in_bounds = [true]} : memref<4xf32, strided<[1]>>, vector<4xf32>
  vector.transfer_write %v, %s[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1]>>
  %r1 = vector.transfer_read %s[%c0], %pad {in_bounds = [true]} : memref<4xf32, strided<[1]>>, vector<4xf32>
  return %r0, %r1 : vector<4xf32>, vector<4xf32>
}

// -----

// A dynamic subview offset cannot be expressed as a static strided slice: the
// buffer is left untouched.

// CHECK-LABEL: func.func @no_promote_subview_dynamic_offset
//        CHECK:   memref.alloca
//        CHECK:   memref.subview
func.func @no_promote_subview_dynamic_offset(%v: vector<4xf32>, %init: vector<8xf32>, %pad: f32, %off: index) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %init, %a[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  %sv = memref.subview %a[%off] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
  vector.transfer_write %v, %sv[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1], offset: ?>>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
  return %r : vector<8xf32>
}

// -----

// A rank-reducing subview (2d -> 1d) is not promotable: strided-slice projection
// requires equal rank.

// CHECK-LABEL: func.func @no_promote_subview_rank_reducing
//        CHECK:   memref.alloca
//        CHECK:   memref.subview
func.func @no_promote_subview_rank_reducing(%v: vector<4xf32>, %init: vector<2x4xf32>, %pad: f32) -> vector<2x4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<2x4xf32>
  vector.transfer_write %init, %a[%c0, %c0] {in_bounds = [true, true]} : vector<2x4xf32>, memref<2x4xf32>
  %sv = memref.subview %a[1, 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: 4>>
  vector.transfer_write %v, %sv[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1], offset: 4>>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<2x4xf32>, vector<2x4xf32>
  return %r : vector<2x4xf32>
}

// -----

// A masked write into the subview is not a whole-sub-region access: not promoted.

// CHECK-LABEL: func.func @no_promote_subview_masked
//        CHECK:   memref.alloca
//        CHECK:   memref.subview
func.func @no_promote_subview_masked(%v: vector<4xf32>, %init: vector<8xf32>, %pad: f32, %m: vector<4xi1>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %init, %a[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  %sv = memref.subview %a[2] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 2>>
  vector.transfer_write %v, %sv[%c0], %m {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1], offset: 2>>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
  return %r : vector<8xf32>
}
