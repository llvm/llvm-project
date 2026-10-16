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
