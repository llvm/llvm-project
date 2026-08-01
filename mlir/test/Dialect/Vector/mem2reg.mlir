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

// The buffer has a use (subview) that mem2reg does not recognize as a
// removable load or store, so the slot is left untouched: must NOT be promoted.

// CHECK-LABEL: func.func @negative_subview
//        CHECK:   memref.alloca
//        CHECK:   vector.transfer_write
//        CHECK:   memref.subview
//        CHECK:   vector.transfer_read
func.func @negative_subview(%pad: f32) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<8xf32>
  %a = memref.alloca() : memref<8xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  %sv = memref.subview %a[0] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1]>>
  %r = vector.transfer_read %sv[%c0], %pad {in_bounds = [true]} : memref<4xf32, strided<[1]>>, vector<4xf32>
  return %r : vector<4xf32>
}

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
