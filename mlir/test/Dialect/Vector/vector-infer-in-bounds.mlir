// RUN: mlir-opt %s -vector-infer-in-bounds -split-input-file | FileCheck %s

// The index is an affine.for induction variable rather than a constant. The
// largest value it takes is 384, and 384 + 128 == 512 == the memref dim, so the
// transfer is exactly in bounds.

// CHECK-LABEL: func @fold_transfer_in_bounds_from_loop_iv
//       CHECK:   vector.transfer_read %{{.*}} {in_bounds = [true]} : memref<512xf32>, vector<128xf32>
func.func @fold_transfer_in_bounds_from_loop_iv(%m: memref<512xf32>, %p: f32) -> vector<128xf32> {
  %acc = arith.constant dense<0.0> : vector<128xf32>
  %r = affine.for %i = 0 to 385 step 128 iter_args(%a = %acc) -> (vector<128xf32>) {
    %v = vector.transfer_read %m[%i], %p : memref<512xf32>, vector<128xf32>
    %s = arith.addf %a, %v : vector<128xf32>
    affine.yield %s : vector<128xf32>
  }
  return %r : vector<128xf32>
}

// -----

// Same loop, but the memref is one element shorter: 384 + 128 == 512 > 511.
// The attribute must not be added -- doing so would be a silent out-of-bounds
// read.

// CHECK-LABEL: func @no_fold_transfer_in_bounds_off_by_one
//       CHECK:   vector.transfer_read
//   CHECK-NOT:   in_bounds
//       CHECK:   : memref<511xf32>, vector<128xf32>
func.func @no_fold_transfer_in_bounds_off_by_one(%m: memref<511xf32>, %p: f32) -> vector<128xf32> {
  %acc = arith.constant dense<0.0> : vector<128xf32>
  %r = affine.for %i = 0 to 385 step 128 iter_args(%a = %acc) -> (vector<128xf32>) {
    %v = vector.transfer_read %m[%i], %p : memref<511xf32>, vector<128xf32>
    %s = arith.addf %a, %v : vector<128xf32>
    affine.yield %s : vector<128xf32>
  }
  return %r : vector<128xf32>
}

// -----

// The induction variable is always a multiple of the step away from the lower
// bound, so for `0 to 300 step 128` it only ever takes {0, 128, 256} and
// 256 + 128 == 384 <= 390. A bound of `ub - 1` = 299 would give 427 > 390 and
// miss this.

// CHECK-LABEL: func @fold_transfer_in_bounds_step_alignment
//       CHECK:   vector.transfer_read %{{.*}} {in_bounds = [true]} : memref<390xf32>, vector<128xf32>
func.func @fold_transfer_in_bounds_step_alignment(%m: memref<390xf32>, %p: f32) -> vector<128xf32> {
  %acc = arith.constant dense<0.0> : vector<128xf32>
  %r = affine.for %i = 0 to 300 step 128 iter_args(%a = %acc) -> (vector<128xf32>) {
    %v = vector.transfer_read %m[%i], %p : memref<390xf32>, vector<128xf32>
    %s = arith.addf %a, %v : vector<128xf32>
    affine.yield %s : vector<128xf32>
  }
  return %r : vector<128xf32>
}

// -----

// The index is an affine.apply over an induction variable rather than the
// induction variable itself: 384 + 64 + 64 == 512.

// CHECK-LABEL: func @fold_transfer_in_bounds_affine_apply_index
//       CHECK:   vector.transfer_read %{{.*}} {in_bounds = [true]} : memref<512xf32>, vector<64xf32>
func.func @fold_transfer_in_bounds_affine_apply_index(%m: memref<512xf32>, %p: f32) -> vector<64xf32> {
  %acc = arith.constant dense<0.0> : vector<64xf32>
  %r = affine.for %i = 0 to 385 step 128 iter_args(%a = %acc) -> (vector<64xf32>) {
    %idx = affine.apply affine_map<(d0) -> (d0 + 64)>(%i)
    %v = vector.transfer_read %m[%idx], %p : memref<512xf32>, vector<64xf32>
    %s = arith.addf %a, %v : vector<64xf32>
    affine.yield %s : vector<64xf32>
  }
  return %r : vector<64xf32>
}

// -----

// `in_bounds` promises that the starting point is in bounds too, so an index
// that is provably below the memref must not be folded even though it leaves
// room for a full vector at the top end (-52 + 128 <= 512).

// CHECK-LABEL: func @no_fold_transfer_in_bounds_negative_constant_index
//       CHECK:   vector.transfer_read
//   CHECK-NOT:   in_bounds
//       CHECK:   : memref<512xf32>, vector<128xf32>
func.func @no_fold_transfer_in_bounds_negative_constant_index(%m: memref<512xf32>, %p: f32) -> vector<128xf32> {
  %c = arith.constant -50 : index
  %v = vector.transfer_read %m[%c], %p : memref<512xf32>, vector<128xf32>
  return %v : vector<128xf32>
}

// -----

// CHECK-LABEL: func @no_fold_transfer_in_bounds_negative_loop_iv
//       CHECK:   vector.transfer_read
//   CHECK-NOT:   in_bounds
//       CHECK:   : memref<512xf32>, vector<128xf32>
func.func @no_fold_transfer_in_bounds_negative_loop_iv(%m: memref<512xf32>, %p: f32) -> vector<128xf32> {
  %acc = arith.constant dense<0.0> : vector<128xf32>
  %r = affine.for %i = -100 to -50 step 16 iter_args(%a = %acc) -> (vector<128xf32>) {
    %v = vector.transfer_read %m[%i], %p : memref<512xf32>, vector<128xf32>
    %s = arith.addf %a, %v : vector<128xf32>
    affine.yield %s : vector<128xf32>
  }
  return %r : vector<128xf32>
}

// -----

// Dynamic loop bound: no constant bound can be derived for the induction
// variable, so the transfer must stay potentially out-of-bounds.

// CHECK-LABEL: func @no_fold_transfer_in_bounds_dynamic_loop_bound
//       CHECK:   vector.transfer_read
//   CHECK-NOT:   in_bounds
//       CHECK:   : memref<512xf32>, vector<128xf32>
func.func @no_fold_transfer_in_bounds_dynamic_loop_bound(%m: memref<512xf32>, %p: f32, %n: index) -> vector<128xf32> {
  %acc = arith.constant dense<0.0> : vector<128xf32>
  %r = affine.for %i = 0 to %n step 128 iter_args(%a = %acc) -> (vector<128xf32>) {
    %v = vector.transfer_read %m[%i], %p : memref<512xf32>, vector<128xf32>
    %s = arith.addf %a, %v : vector<128xf32>
    affine.yield %s : vector<128xf32>
  }
  return %r : vector<128xf32>
}

// -----

// The index is bounded, but the source dimension is dynamic, so there is no
// static size to compare against and nothing can be proved.

// CHECK-LABEL: func @no_fold_transfer_in_bounds_dynamic_source_dim
//       CHECK:   vector.transfer_read
//   CHECK-NOT:   in_bounds
//       CHECK:   : memref<?xf32>, vector<128xf32>
func.func @no_fold_transfer_in_bounds_dynamic_source_dim(%m: memref<?xf32>, %p: f32) -> vector<128xf32> {
  %acc = arith.constant dense<0.0> : vector<128xf32>
  %r = affine.for %i = 0 to 385 step 128 iter_args(%a = %acc) -> (vector<128xf32>) {
    %v = vector.transfer_read %m[%i], %p : memref<?xf32>, vector<128xf32>
    %s = arith.addf %a, %v : vector<128xf32>
    affine.yield %s : vector<128xf32>
  }
  return %r : vector<128xf32>
}

// -----

// A scalable vector dimension must not be folded even when a constant bound is
// available for the index: `vector<[4]xf32>` reads `4 * vscale` elements.

// CHECK-LABEL: func @no_fold_transfer_in_bounds_scalable_loop_iv
//       CHECK:   vector.transfer_read
//   CHECK-NOT:   in_bounds
//       CHECK:   : memref<4xf32>, vector<[4]xf32>
func.func @no_fold_transfer_in_bounds_scalable_loop_iv(%m: memref<4xf32>, %p: f32) -> vector<[4]xf32> {
  %acc = arith.constant dense<0.0> : vector<[4]xf32>
  %r = affine.for %i = 0 to 1 iter_args(%a = %acc) -> (vector<[4]xf32>) {
    %v = vector.transfer_read %m[%i], %p : memref<4xf32>, vector<[4]xf32>
    %s = arith.addf %a, %v : vector<[4]xf32>
    affine.yield %s : vector<[4]xf32>
  }
  return %r : vector<[4]xf32>
}

// -----

// The write path uses the same bound computation, where an unsound fold is an
// out-of-bounds store.

// CHECK-LABEL: func @fold_transfer_write_in_bounds_from_loop_iv
//       CHECK:   vector.transfer_write %{{.*}} {in_bounds = [true]} : vector<128xf32>, memref<512xf32>
func.func @fold_transfer_write_in_bounds_from_loop_iv(%m: memref<512xf32>, %v: vector<128xf32>) {
  affine.for %i = 0 to 385 step 128 {
    vector.transfer_write %v, %m[%i] : vector<128xf32>, memref<512xf32>
  }
  return
}

// -----

// CHECK-LABEL: func @no_fold_transfer_write_in_bounds_off_by_one
//       CHECK:   vector.transfer_write
//   CHECK-NOT:   in_bounds
//       CHECK:   : vector<128xf32>, memref<511xf32>
func.func @no_fold_transfer_write_in_bounds_off_by_one(%m: memref<511xf32>, %v: vector<128xf32>) {
  affine.for %i = 0 to 385 step 128 {
    vector.transfer_write %v, %m[%i] : vector<128xf32>, memref<511xf32>
  }
  return
}

// -----

// Both dims driven by induction variables of a 2-D nest.

// CHECK-LABEL: func @fold_transfer_in_bounds_2d_nest
//       CHECK:   vector.transfer_read %{{.*}} {in_bounds = [true, true]} : memref<64x512xf32>, vector<4x128xf32>
func.func @fold_transfer_in_bounds_2d_nest(%m: memref<64x512xf32>, %p: f32) -> vector<4x128xf32> {
  %acc = arith.constant dense<0.0> : vector<4x128xf32>
  %r = affine.for %i = 0 to 61 step 4 iter_args(%a = %acc) -> (vector<4x128xf32>) {
    %r2 = affine.for %j = 0 to 385 step 128 iter_args(%b = %a) -> (vector<4x128xf32>) {
      %v = vector.transfer_read %m[%i, %j], %p : memref<64x512xf32>, vector<4x128xf32>
      %s = arith.addf %b, %v : vector<4x128xf32>
      affine.yield %s : vector<4x128xf32>
    }
    affine.yield %r2 : vector<4x128xf32>
  }
  return %r : vector<4x128xf32>
}

// -----

// A tensor source goes through the same path as a memref.

// CHECK-LABEL: func @fold_transfer_in_bounds_tensor_source
//       CHECK:   vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<512xf32>, vector<128xf32>
func.func @fold_transfer_in_bounds_tensor_source(%t: tensor<512xf32>, %p: f32) -> vector<128xf32> {
  %acc = arith.constant dense<0.0> : vector<128xf32>
  %r = affine.for %i = 0 to 385 step 128 iter_args(%a = %acc) -> (vector<128xf32>) {
    %v = vector.transfer_read %t[%i], %p : tensor<512xf32>, vector<128xf32>
    %s = arith.addf %a, %v : vector<128xf32>
    affine.yield %s : vector<128xf32>
  }
  return %r : vector<128xf32>
}

// -----

// scf.for already implements ValueBoundsOpInterface, so it benefits from the
// bound query without any affine-specific support.

// CHECK-LABEL: func @fold_transfer_in_bounds_scf_for
//       CHECK:   vector.transfer_read %{{.*}} {in_bounds = [true]} : memref<512xf32>, vector<128xf32>
func.func @fold_transfer_in_bounds_scf_for(%m: memref<512xf32>, %p: f32) -> vector<128xf32> {
  %c0 = arith.constant 0 : index
  %c385 = arith.constant 385 : index
  %c128 = arith.constant 128 : index
  %acc = arith.constant dense<0.0> : vector<128xf32>
  %r = scf.for %i = %c0 to %c385 step %c128 iter_args(%a = %acc) -> (vector<128xf32>) {
    %v = vector.transfer_read %m[%i], %p : memref<512xf32>, vector<128xf32>
    %s = arith.addf %a, %v : vector<128xf32>
    scf.yield %s : vector<128xf32>
  }
  return %r : vector<128xf32>
}

// -----

// CHECK-LABEL: func @no_fold_transfer_in_bounds_scf_for_off_by_one
//       CHECK:   vector.transfer_read
//   CHECK-NOT:   in_bounds
//       CHECK:   : memref<511xf32>, vector<128xf32>
func.func @no_fold_transfer_in_bounds_scf_for_off_by_one(%m: memref<511xf32>, %p: f32) -> vector<128xf32> {
  %c0 = arith.constant 0 : index
  %c385 = arith.constant 385 : index
  %c128 = arith.constant 128 : index
  %acc = arith.constant dense<0.0> : vector<128xf32>
  %r = scf.for %i = %c0 to %c385 step %c128 iter_args(%a = %acc) -> (vector<128xf32>) {
    %v = vector.transfer_read %m[%i], %p : memref<511xf32>, vector<128xf32>
    %s = arith.addf %a, %v : vector<128xf32>
    scf.yield %s : vector<128xf32>
  }
  return %r : vector<128xf32>
}

// -----

// A transposing permutation map: vector dim 0 comes from memref dim 1 and vice
// versa. Each vector dim must be checked against the memref dim it actually
// maps to. Here iv max is 60: dim 1 (512) has room for 4, dim 0 (64) does not
// have room for 8 (60 + 8 = 68 > 64), so exactly one dim is in bounds.

// CHECK-LABEL: func @fold_transfer_in_bounds_transposed_partial
//       CHECK:   vector.transfer_read %{{.*}} {in_bounds = [true, false]
func.func @fold_transfer_in_bounds_transposed_partial(%m: memref<64x512xf32>, %p: f32) -> vector<4x8xf32> {
  %acc = arith.constant dense<0.0> : vector<4x8xf32>
  %r = affine.for %i = 0 to 61 step 4 iter_args(%a = %acc) -> (vector<4x8xf32>) {
    %v = vector.transfer_read %m[%i, %i], %p {permutation_map = affine_map<(d0, d1) -> (d1, d0)>}
       : memref<64x512xf32>, vector<4x8xf32>
    %s = arith.addf %a, %v : vector<4x8xf32>
    affine.yield %s : vector<4x8xf32>
  }
  return %r : vector<4x8xf32>
}

// -----

// Same map, iv max 56: 56 + 4 = 60 <= 512 and 56 + 8 = 64 <= 64, so both.

// CHECK-LABEL: func @fold_transfer_in_bounds_transposed_full
//       CHECK:   vector.transfer_read %{{.*}} {in_bounds = [true, true]
func.func @fold_transfer_in_bounds_transposed_full(%m: memref<64x512xf32>, %p: f32) -> vector<4x8xf32> {
  %acc = arith.constant dense<0.0> : vector<4x8xf32>
  %r = affine.for %i = 0 to 57 step 8 iter_args(%a = %acc) -> (vector<4x8xf32>) {
    %v = vector.transfer_read %m[%i, %i], %p {permutation_map = affine_map<(d0, d1) -> (d1, d0)>}
       : memref<64x512xf32>, vector<4x8xf32>
    %s = arith.addf %a, %v : vector<4x8xf32>
    affine.yield %s : vector<4x8xf32>
  }
  return %r : vector<4x8xf32>
}

// -----

// A broadcast vector dim is only in bounds once every non-broadcast dim is.
// Here the real dim (128 wide, iv max 384, memref dim 512) fits exactly.

// CHECK-LABEL: func @fold_transfer_in_bounds_broadcast_dim
//       CHECK:   vector.transfer_read %{{.*}} {in_bounds = [true, true]
func.func @fold_transfer_in_bounds_broadcast_dim(%m: memref<64x512xf32>, %p: f32) -> vector<4x128xf32> {
  %acc = arith.constant dense<0.0> : vector<4x128xf32>
  %r = affine.for %i = 0 to 64 iter_args(%o = %acc) -> (vector<4x128xf32>) {
    %r2 = affine.for %j = 0 to 385 step 128 iter_args(%a = %o) -> (vector<4x128xf32>) {
      %v = vector.transfer_read %m[%i, %j], %p {permutation_map = affine_map<(d0, d1) -> (0, d1)>}
         : memref<64x512xf32>, vector<4x128xf32>
      %s = arith.addf %a, %v : vector<4x128xf32>
      affine.yield %s : vector<4x128xf32>
    }
    affine.yield %r2 : vector<4x128xf32>
  }
  return %r : vector<4x128xf32>
}

// -----

// One element shorter: the real dim no longer fits, so the broadcast dim must
// not be claimed either.

// CHECK-LABEL: func @no_fold_transfer_in_bounds_broadcast_dim_oob
//       CHECK:   vector.transfer_read
//   CHECK-NOT:   in_bounds
//       CHECK:   : memref<64x511xf32>, vector<4x128xf32>
func.func @no_fold_transfer_in_bounds_broadcast_dim_oob(%m: memref<64x511xf32>, %p: f32) -> vector<4x128xf32> {
  %acc = arith.constant dense<0.0> : vector<4x128xf32>
  %r = affine.for %i = 0 to 64 iter_args(%o = %acc) -> (vector<4x128xf32>) {
    %r2 = affine.for %j = 0 to 385 step 128 iter_args(%a = %o) -> (vector<4x128xf32>) {
      %v = vector.transfer_read %m[%i, %j], %p {permutation_map = affine_map<(d0, d1) -> (0, d1)>}
         : memref<64x511xf32>, vector<4x128xf32>
      %s = arith.addf %a, %v : vector<4x128xf32>
      affine.yield %s : vector<4x128xf32>
    }
    affine.yield %r2 : vector<4x128xf32>
  }
  return %r : vector<4x128xf32>
}
