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

// A vscale*C-sized memref accessed by a whole-buffer vector<[C]> transfer is
// promoted: its runtime length matches the vector exactly. Here the scalable
// slot is carried across scf.for as an iter_arg/result.

// CHECK-LABEL: func.func @scalable_whole_buffer_in_loop
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   vector.transfer_write
//    CHECK-NOT:   vector.transfer_read
//        CHECK:   %[[RES:.*]] = scf.for {{.*}} iter_args(%[[IT:.*]] = %{{.*}}) -> (vector<[4]xf32>)
//        CHECK:     %[[NEXT:.*]] = arith.addf %[[IT]], %[[IT]] : vector<[4]xf32>
//        CHECK:     scf.yield %[[NEXT]] : vector<[4]xf32>
//        CHECK:   return %[[RES]] : vector<[4]xf32>
func.func @scalable_whole_buffer_in_loop(%pad: f32, %lb: index, %ub: index, %step: index) -> vector<[4]xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant dense<1.0> : vector<[4]xf32>
  %vs = vector.vscale
  %sz = arith.muli %vs, %c4 : index
  %a = memref.alloca(%sz) : memref<?xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<[4]xf32>, memref<?xf32>
  scf.for %i = %lb to %ub step %step {
    %v = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<?xf32>, vector<[4]xf32>
    %n = arith.addf %v, %v : vector<[4]xf32>
    vector.transfer_write %n, %a[%c0] {in_bounds = [true]} : vector<[4]xf32>, memref<?xf32>
  }
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<?xf32>, vector<[4]xf32>
  return %r : vector<[4]xf32>
}

// -----

// Size mismatch (vscale*8 buffer, vector<[4]xf32> transfer) is a partial
// access: must NOT be promoted.

// CHECK-LABEL: func.func @negative_scalable
//        CHECK:   memref.alloca
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
func.func @negative_scalable(%pad: f32) -> vector<[4]xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant dense<1.0> : vector<[4]xf32>
  %vs = vector.vscale
  %sz = arith.muli %vs, %c8 : index
  %a = memref.alloca(%sz) : memref<?xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<[4]xf32>, memref<?xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<?xf32>, vector<[4]xf32>
  return %r : vector<[4]xf32>
}

// -----

// An unanalyzable dynamic size (not a vscale multiple) cannot be matched to the
// vector length, even with a scalable transfer: must NOT be promoted.

// CHECK-LABEL: func.func @negative_scalable_plain_dynamic
//        CHECK:   memref.alloca
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
func.func @negative_scalable_plain_dynamic(%pad: f32, %d: index) -> vector<[4]xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<[4]xf32>
  %a = memref.alloca(%d) : memref<?xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<[4]xf32>, memref<?xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<?xf32>, vector<[4]xf32>
  return %r : vector<[4]xf32>
}

// -----

// A dynamic-shape memref with a fixed-size transfer: the runtime extent is
// unknown, so the transfer cannot cover the whole buffer. Must NOT be promoted.

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
//        CHECK:   %[[INS:.*]] = vector.insert_strided_slice %[[V]], %[[INIT]] offsets = [2], strides = [1]
//        CHECK:   return %[[INS]] : vector<8xf32>
func.func @subview_static_write(%v: vector<4xf32>, %init: vector<8xf32>, %pad: f32) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %init, %a[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  // Write via a subview.
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
//        CHECK:   %[[EXT:.*]] = vector.extract_strided_slice %{{.*}} offsets = [2], sizes = [4], strides = [1]
//        CHECK:   return %[[EXT]] : vector<4xf32>
func.func @subview_static_read(%init: vector<8xf32>, %pad: f32) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %init, %a[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  // Read via a subview.
  %sv = memref.subview %a[2] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 2>>
  %r = vector.transfer_read %sv[%c0], %pad {in_bounds = [true]} : memref<4xf32, strided<[1], offset: 2>>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A buffer accessed *only* through subviews, with no whole-buffer transfer to
// seed the promoted value: the reaching definition is the allocator's default
// value (ub.poison), and the written slice is inserted into it and read back
// out.

// CHECK-LABEL: func.func @subview_only_write_read
//   CHECK-SAME:   (%[[V:.*]]: vector<4xf32>, %[[PAD:.*]]: f32)
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.subview
//    CHECK-NOT:   vector.transfer_write
//    CHECK-NOT:   vector.transfer_read
//        CHECK:   %[[POISON:.*]] = ub.poison : vector<8xf32>
//        CHECK:   vector.insert_strided_slice %[[V]], %[[POISON]] offsets = [0], strides = [1]
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
//        CHECK:   %[[D0:.*]] = vector.insert_strided_slice %[[VA]], %{{.*}} offsets = [0], strides = [1]
//        CHECK:   %[[D1:.*]] = vector.insert_strided_slice %[[VB]], %[[D0]] offsets = [4], strides = [1]
//        CHECK:   %[[R:.*]] = vector.extract_strided_slice %[[D1]] offsets = [2], sizes = [4], strides = [1]
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

// Two overlapping subview writes: %vA covers [0, 4) and %vB covers [2, 6),
// overlapping on [2, 4). Each write composes into the parent value in program
// order, so the buffer holds %vA at [0, 2) and %vB at [2, 6). A subview read
// spanning [1, 5) then extracts across the overlap, reading one lane of %vA
// and three lanes of %vB from the composed value.

// CHECK-LABEL: func.func @subview_overlapping_writes_overlap_read
//   CHECK-SAME:   (%[[VA:.*]]: vector<4xf32>, %[[VB:.*]]: vector<4xf32>, %[[PAD:.*]]: f32)
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.subview
//        CHECK:   %[[D0:.*]] = vector.insert_strided_slice %[[VA]], %{{.*}} offsets = [0], strides = [1]
//        CHECK:   %[[D1:.*]] = vector.insert_strided_slice %[[VB]], %[[D0]] offsets = [2], strides = [1]
//        CHECK:   %[[R:.*]] = vector.extract_strided_slice %[[D1]] offsets = [1], sizes = [4], strides = [1]
//        CHECK:   return %[[R]] : vector<4xf32>
func.func @subview_overlapping_writes_overlap_read(%vA: vector<4xf32>, %vB: vector<4xf32>, %pad: f32) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  %s0 = memref.subview %a[0] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1]>>
  %s2 = memref.subview %a[2] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 2>>
  %s1 = memref.subview %a[1] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 1>>
  vector.transfer_write %vA, %s0[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1]>>
  vector.transfer_write %vB, %s2[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1], offset: 2>>
  %r = vector.transfer_read %s1[%c0], %pad {in_bounds = [true]} : memref<4xf32, strided<[1], offset: 1>>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A buffer allocated before an scf.for and accessed inside the loop body only
// through a subview, with a cross-iteration dependence (each iteration reads the
// value the previous iteration wrote). The subview aliaser composes with region
// promotion: the whole buffer is carried across iterations as a vector iter_arg,
// the in-body read/write become extract/insert_strided_slice on that value.

// CHECK-LABEL: func.func @subview_in_scf_for_cross_iter
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   memref.subview
//    CHECK-NOT:   vector.transfer_write
//    CHECK-NOT:   vector.transfer_read
//        CHECK:   %[[R:.*]] = scf.for %{{.*}} iter_args(%[[IT:.*]] = %{{.*}}) -> (vector<8xf32>)
//        CHECK:     %[[V:.*]] = vector.extract_strided_slice %[[IT]] offsets = [0], sizes = [4], strides = [1]
//        CHECK:     %[[N:.*]] = arith.addf %[[V]], %[[V]]
//        CHECK:     %[[INS:.*]] = vector.insert_strided_slice %[[N]], %[[IT]] offsets = [0], strides = [1]
//        CHECK:     scf.yield %[[INS]] : vector<8xf32>
//        CHECK:   vector.extract_strided_slice %[[R]] offsets = [0], sizes = [4], strides = [1]
func.func @subview_in_scf_for_cross_iter(%lb: index, %ub: index, %step: index, %init: vector<8xf32>, %pad: f32) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %init, %a[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  scf.for %i = %lb to %ub step %step {
    %sv = memref.subview %a[0] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1]>>
    %v = vector.transfer_read %sv[%c0], %pad {in_bounds = [true]} : memref<4xf32, strided<[1]>>, vector<4xf32>
    %n = arith.addf %v, %v : vector<4xf32>
    vector.transfer_write %n, %sv[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1]>>
  }
  %svr = memref.subview %a[0] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1]>>
  %r = vector.transfer_read %svr[%c0], %pad {in_bounds = [true]} : memref<4xf32, strided<[1]>>, vector<4xf32>
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
//        CHECK:   %[[R0:.*]] = vector.extract_strided_slice %[[POISON]] offsets = [0], sizes = [4], strides = [1]
//        CHECK:   %[[INS:.*]] = vector.insert_strided_slice %[[V]], %[[POISON]] offsets = [0], strides = [1]
//        CHECK:   %[[R1:.*]] = vector.extract_strided_slice %[[INS]] offsets = [0], sizes = [4], strides = [1]
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

// CHECK-LABEL: func.func @negative_subview_dynamic_offset
//        CHECK:   memref.alloca
//        CHECK:   memref.subview
func.func @negative_subview_dynamic_offset(%v: vector<4xf32>, %init: vector<8xf32>, %pad: f32, %off: index) -> vector<8xf32> {
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

// CHECK-LABEL: func.func @negative_subview_rank_reducing
//        CHECK:   memref.alloca
//        CHECK:   memref.subview
func.func @negative_subview_rank_reducing(%v: vector<4xf32>, %init: vector<2x4xf32>, %pad: f32) -> vector<2x4xf32> {
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

// CHECK-LABEL: func.func @negative_subview_masked
//        CHECK:   memref.alloca
//        CHECK:   memref.subview
func.func @negative_subview_masked(%v: vector<4xf32>, %init: vector<8xf32>, %pad: f32, %m: vector<4xi1>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %init, %a[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  %sv = memref.subview %a[2] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 2>>
  vector.transfer_write %v, %sv[%c0], %m {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1], offset: 2>>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
  return %r : vector<8xf32>
}

// -----

// A prefetching software-pipelined loop. The stage buffer double-buffers the
// prefetched tile: the prefetch of the next tile is skipped on the last iteration.

// CHECK-LABEL: func.func @pipelined_prefetch
//    CHECK-NOT:   memref.alloca
//        CHECK:   %[[P0:.*]] = vector.transfer_read %{{.*}} : memref<64xf32>, vector<8xf32>
//    CHECK-NOT:   vector.transfer_write
//        CHECK:   scf.for %[[I:.*]] = {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}, %[[STAGE:.*]] = %[[P0]]) -> (vector<8xf32>, vector<8xf32>)
//        CHECK:     %[[ACCN:.*]] = arith.addf %[[ACC]], %[[STAGE]] : vector<8xf32>
//        CHECK:     %[[INEXT:.*]] = arith.addi %[[I]], %{{.*}}
//        CHECK:     %[[G:.*]] = arith.cmpi slt, %[[INEXT]], %{{.*}}
//        CHECK:     %[[NEXT:.*]] = scf.if %[[G]] -> (vector<8xf32>) {
//        CHECK:       %[[CUR:.*]] = vector.transfer_read %{{.*}}[%[[INEXT]]]
//        CHECK:       scf.yield %[[CUR]] : vector<8xf32>
//        CHECK:     } else {
//        CHECK:       scf.yield %[[STAGE]] : vector<8xf32>
//        CHECK:     }
//        CHECK:     scf.yield %[[ACCN]], %[[NEXT]] : vector<8xf32>, vector<8xf32>
func.func @pipelined_prefetch(%lb: index, %ub: index, %step: index, %in: memref<64xf32>, %pad: f32) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<0.0> : vector<8xf32>
  %stage = memref.alloca() : memref<8xf32>
  // Prologue: prefetch tile[0] into the stage buffer.
  %p0 = vector.transfer_read %in[%c0], %pad {in_bounds = [true]} : memref<64xf32>, vector<8xf32>
  vector.transfer_write %p0, %stage[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (vector<8xf32>) {
    // Consume the tile prefetched by the previous iteration (unconditional).
    %tile = vector.transfer_read %stage[%c0], %pad {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
    %accn = arith.addf %acc, %tile : vector<8xf32>
    // Prefetch the next tile into the stage buffer (skip on the last iteration).
    %inext = arith.addi %i, %c1 : index
    %g = arith.cmpi slt, %inext, %ub : index
    scf.if %g {
      %next = vector.transfer_read %in[%inext], %pad {in_bounds = [true]} : memref<64xf32>, vector<8xf32>
      vector.transfer_write %next, %stage[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
    }
    scf.yield %accn : vector<8xf32>
  }
  return %r : vector<8xf32>
}

// -----

// A whole-buffer accumulator updated conditionally inside an scf.if (a masked
// running reduction).

// CHECK-LABEL: func.func @cond_accumulate
//    CHECK-NOT:   memref.alloca
//    CHECK-NOT:   vector.transfer_write
//    CHECK-NOT:   vector.transfer_read
//        CHECK:   %[[R:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (vector<4xf32>)
//        CHECK:     %[[M:.*]] = memref.load
//        CHECK:     %[[NEW:.*]] = scf.if %[[M]] -> (vector<4xf32>) {
//        CHECK:       %[[D:.*]] = arith.addf %[[ACC]], %[[ACC]] : vector<4xf32>
//        CHECK:       scf.yield %[[D]] : vector<4xf32>
//        CHECK:     } else {
//        CHECK:       scf.yield %[[ACC]] : vector<4xf32>
//        CHECK:     }
//        CHECK:     scf.yield %[[NEW]] : vector<4xf32>
//        CHECK:   return %[[R]] : vector<4xf32>
func.func @cond_accumulate(%lb: index, %ub: index, %step: index, %init: vector<4xf32>, %pad: f32, %mask: memref<?xi1>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %a = memref.alloca() : memref<4xf32>
  vector.transfer_write %init, %a[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  scf.for %i = %lb to %ub step %step {
    %m = memref.load %mask[%i] : memref<?xi1>
    %v = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
    scf.if %m {
      %d = arith.addf %v, %v : vector<4xf32>
      vector.transfer_write %d, %a[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
    }
    scf.yield
  }
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A tiled GEMM with a dynamic K early-exit. A local accumulator tile D sums the
// K-tiles via vector.contract only while k < dyn_k. 

// CHECK-LABEL: func.func @gemm_k_early_exit
//   CHECK-SAME:   (%[[A:.*]]: memref<4x16xf32>, %[[B:.*]]: memref<16x4xf32>, %[[C:.*]]: memref<4x4xf32>, %[[DYNK:.*]]: index, %[[PAD:.*]]: f32)
//    CHECK-NOT:   memref.alloca
//        CHECK:   %[[R:.*]] = scf.for %[[K:.*]] = {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (vector<4x4xf32>)
//        CHECK:     %[[INRANGE:.*]] = arith.cmpi slt, %[[K]], %[[DYNK]]
//        CHECK:     %[[NEW:.*]] = scf.if %[[INRANGE]] -> (vector<4x4xf32>) {
//        CHECK:       %[[ATILE:.*]] = vector.transfer_read %[[A]]
//        CHECK:       %[[BTILE:.*]] = vector.transfer_read %[[B]]
//        CHECK:       %[[MM:.*]] = vector.contract {{.*}} %[[ATILE]], %[[BTILE]], %[[ACC]]
//        CHECK:       scf.yield %[[MM]] : vector<4x4xf32>
//        CHECK:     } else {
//        CHECK:       scf.yield %[[ACC]] : vector<4x4xf32>
//        CHECK:     }
//        CHECK:     scf.yield %[[NEW]] : vector<4x4xf32>
//        CHECK:   %[[CVAL:.*]] = vector.transfer_read %[[C]]
//        CHECK:   %[[SUM:.*]] = arith.addf %[[CVAL]], %[[R]] : vector<4x4xf32>
//        CHECK:   vector.transfer_write %[[SUM]], %[[C]]
func.func @gemm_k_early_exit(%A: memref<4x16xf32>, %B: memref<16x4xf32>,
                             %C: memref<4x4xf32>, %dyn_k: index, %pad: f32) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant dense<0.0> : vector<4x4xf32>
  // Local accumulator tile D, zero-initialized.
  %d = memref.alloca() : memref<4x4xf32>
  vector.transfer_write %cst, %d[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<4x4xf32>
  scf.for %k = %c0 to %c16 step %c4 {
    // Early exit: only accumulate K tiles whose offset is below dyn_k.
    %inrange = arith.cmpi slt, %k, %dyn_k : index
    scf.if %inrange {
      %atile = vector.transfer_read %A[%c0, %k], %pad {in_bounds = [true, true]} : memref<4x16xf32>, vector<4x4xf32>
      %btile = vector.transfer_read %B[%k, %c0], %pad {in_bounds = [true, true]} : memref<16x4xf32>, vector<4x4xf32>
      %acc = vector.transfer_read %d[%c0, %c0], %pad {in_bounds = [true, true]} : memref<4x4xf32>, vector<4x4xf32>
      %mm = vector.contract {indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %atile, %btile, %acc : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
      vector.transfer_write %mm, %d[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<4x4xf32>
    }
    scf.yield
  }
  // Accumulate the local tile D into the C parameter: C = C + D.
  %cval = vector.transfer_read %C[%c0, %c0], %pad {in_bounds = [true, true]} : memref<4x4xf32>, vector<4x4xf32>
  %dval = vector.transfer_read %d[%c0, %c0], %pad {in_bounds = [true, true]} : memref<4x4xf32>, vector<4x4xf32>
  %sum = arith.addf %cval, %dval : vector<4x4xf32>
  vector.transfer_write %sum, %C[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, memref<4x4xf32>
  return
}
