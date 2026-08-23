// RUN: tr-opt %s --convert-tr-row-sum-to-gpu | FileCheck %s

// Milestone 19: the canonical source uses truncating `arith.divui`.
// The GPU kernel uses `arith.ceildivui` and treats out-of-bounds columns
// as zero. The same dynamic kernel covers
//
//   K in {1, 31, 32, 33, 127, 128, 129, 255, 256, 257}
//
//   K=1     -> 1 tile, 127 masked zeros / row
//   K=31    -> 1 tile
//   K=32    -> 1 tile, one live element in the first lane stripe
//   K=33    -> 1 tile
//   K=127   -> 1 tile
//   K=128   -> 1 tile, mask never taken
//   K=129   -> 2 tiles, second tile has 1 live column
//   K=255   -> 2 tiles
//   K=256   -> 2 tiles, mask never taken
//   K=257   -> 3 tiles
//
// Floating-point assumption: the lane-then-warp reduction tree
// reassociates the K-sum relative to a sequential left fold. TileReducer
// treats row-sum as reassociative (same contract as the Linalg generic).
// Masked zeros are identities for add.

// CHECK-LABEL: func.func @row_sum
func.func @row_sum(%in: !tr.buffer<MxKxf32>, %out: !tr.buffer<Mxf32>) {
  %row_blk     = tr.program_id 0 : index
  %c128        = arith.constant 128 : index
  %k           = tr.dim %in, 1 : !tr.buffer<MxKxf32>, index
  // Source: truncating division. The pass must not leave this as the
  // trip count of the physical K loop.
  %num_k_tiles = arith.divui %k, %c128 : index
  %zero = tr.constant 0.0 : !tr.tile<128xf32>
  %result = tr.for %kt = 0 to %num_k_tiles step 1
      iter_args(%acc = %zero) -> !tr.tile<128xf32> {
    %t       = tr.load %in[%row_blk, %kt]
        : !tr.buffer<MxKxf32>, !tr.tile<128x128xf32>
    %partial = tr.reduce_sum %t, axis = 1
        : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
    %acc2    = tr.add %acc, %partial : !tr.tile<128xf32>
    tr.yield %acc2 : !tr.tile<128xf32>
  }
  tr.store %out[%row_blk], %result : !tr.buffer<Mxf32>, !tr.tile<128xf32>
  return
}

// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[Z:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: memref.dim
// CHECK: %[[K:.*]] = memref.dim
// CHECK-NOT: = arith.divui
// CHECK: arith.ceildivui %[[K]], %[[C128]]
// CHECK: arith.cmpi ult
// CHECK: scf.if
// CHECK: memref.load
// CHECK: else
// CHECK: scf.yield %[[Z]] : f32
