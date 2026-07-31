// RUN: mlir-opt %s --test-xegpu-whole-buffer-promotion --split-input-file | FileCheck %s
// RUN: mlir-opt %s --test-xegpu-whole-buffer-promotion="max-promoted-buffer-bytes=8" --split-input-file | FileCheck %s --check-prefix=CAP

// A memref.alloc that is only ever accessed as a whole buffer through
// vector.transfer_read / vector.transfer_write is promoted to a single vector
// SSA value. This is the VectorToXeGPU-local, Mem2Reg-style promotion used as a
// pre-lowering step; it is exercised here in isolation.

// CHECK-LABEL: func.func @whole_buffer_write_read
//   CHECK-SAME:   (%[[PAD:.*]]: f32)
//    CHECK-NOT:   memref.alloc
//    CHECK-NOT:   vector.transfer_write
//    CHECK-NOT:   vector.transfer_read
//        CHECK:   %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
//        CHECK:   return %[[CST]] : vector<4xf32>
func.func @whole_buffer_write_read(%pad: f32) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<4xf32>
  %a = memref.alloc() : memref<4xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A whole-buffer slot carried across scf.for is threaded as an iter_arg/result.

// CHECK-LABEL: func.func @whole_buffer_in_loop
//    CHECK-NOT:   memref.alloc
//    CHECK-NOT:   vector.transfer_write
//    CHECK-NOT:   vector.transfer_read
//        CHECK:   %[[RES:.*]] = scf.for {{.*}} iter_args(%[[IT:.*]] = %{{.*}}) -> (vector<4xf32>)
//        CHECK:     %[[NEXT:.*]] = arith.addf %[[IT]], %[[IT]] : vector<4xf32>
//        CHECK:     scf.yield %[[NEXT]] : vector<4xf32>
//        CHECK:   return %[[RES]] : vector<4xf32>
func.func @whole_buffer_in_loop(%pad: f32, %lb: index, %ub: index, %step: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<4xf32>
  %a = memref.alloc() : memref<4xf32>
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
//    CHECK-NOT:   memref.alloc
//    CHECK-NOT:   vector.transfer
//        CHECK:   return %{{.*}} : vector<2x4xf32>
func.func @whole_buffer_2d(%pad: f32) -> vector<2x4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<2x4xf32>
  %a = memref.alloc() : memref<2x4xf32>
  vector.transfer_write %cst, %a[%c0, %c0] {in_bounds = [true, true]} : vector<2x4xf32>, memref<2x4xf32>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true]} : memref<2x4xf32>, vector<2x4xf32>
  return %r : vector<2x4xf32>
}

// -----

// The byte cap gates promotion: a 16-byte buffer is promoted under the default
// 4096-byte cap (CHECK) but left as memory under an 8-byte cap (CAP).

//  CHECK-LABEL: func.func @size_cap
//    CHECK-NOT:   memref.alloc
//        CHECK:   return %{{.*}} : vector<4xf32>

//    CAP-LABEL: func.func @size_cap
//        CAP:   memref.alloc
//        CAP:   vector.transfer_write
//        CAP:   vector.transfer_read
func.func @size_cap(%pad: f32) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<4xf32>
  %a = memref.alloc() : memref<4xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// Non-zero access offset: the transfer does not cover the whole buffer, so the
// slot must NOT be promoted.

// CHECK-LABEL: func.func @negative_nonzero_index
//        CHECK:   memref.alloc
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
func.func @negative_nonzero_index(%pad: f32) -> vector<4xf32> {
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<1.0> : vector<4xf32>
  %a = memref.alloc() : memref<8xf32>
  vector.transfer_write %cst, %a[%c1] {in_bounds = [true]} : vector<4xf32>, memref<8xf32>
  %r = vector.transfer_read %a[%c1], %pad {in_bounds = [true]} : memref<8xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A masked transfer only touches part of the buffer: must NOT be promoted.

// CHECK-LABEL: func.func @negative_masked
//        CHECK:   memref.alloc
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
func.func @negative_masked(%pad: f32, %m: vector<4xi1>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<4xf32>
  %a = memref.alloc() : memref<4xf32>
  vector.transfer_write %cst, %a[%c0], %m {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// A partial (out-of-bounds) transfer must NOT be promoted.

// CHECK-LABEL: func.func @negative_out_of_bounds
//        CHECK:   memref.alloc
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
func.func @negative_out_of_bounds(%pad: f32) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<8xf32>
  %a = memref.alloc() : memref<4xf32>
  vector.transfer_write %cst, %a[%c0] : vector<8xf32>, memref<4xf32>
  %r = vector.transfer_read %a[%c0], %pad : memref<4xf32>, vector<8xf32>
  return %r : vector<8xf32>
}

// -----

// A non-identity (transposing) permutation map is not a whole-buffer identity
// access: must NOT be promoted.

// CHECK-LABEL: func.func @negative_transpose_map
//        CHECK:   memref.alloc
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
func.func @negative_transpose_map(%pad: f32) -> vector<4x2xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<2x4xf32>
  %a = memref.alloc() : memref<2x4xf32>
  vector.transfer_write %cst, %a[%c0, %c0] {in_bounds = [true, true]} : vector<2x4xf32>, memref<2x4xf32>
  %r = vector.transfer_read %a[%c0, %c0], %pad {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2x4xf32>, vector<4x2xf32>
  return %r : vector<4x2xf32>
}

// -----

// An alloc also accessed through a scalar memref.load cannot be promoted to a
// vector: must NOT be promoted.

// CHECK-LABEL: func.func @negative_mixed_scalar_access
//        CHECK:   memref.alloc
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
//        CHECK:   memref.load
func.func @negative_mixed_scalar_access(%pad: f32) -> (vector<4xf32>, f32) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<4xf32>
  %a = memref.alloc() : memref<4xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  %s = memref.load %a[%c0] : memref<4xf32>
  return %r, %s : vector<4xf32>, f32
}

// -----

// A scalable vector never equals the fixed-size slot type: must NOT be
// promoted.

// CHECK-LABEL: func.func @negative_scalable
//        CHECK:   memref.alloc
//        CHECK:   vector.transfer_write
//        CHECK:   vector.transfer_read
func.func @negative_scalable(%pad: f32) -> vector<[4]xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.0> : vector<[4]xf32>
  %a = memref.alloc() : memref<4xf32>
  vector.transfer_write %cst, %a[%c0] {in_bounds = [true]} : vector<[4]xf32>, memref<4xf32>
  %r = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : memref<4xf32>, vector<[4]xf32>
  return %r : vector<[4]xf32>
}
