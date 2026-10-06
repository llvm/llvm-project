// RUN: tr-opt %s --convert-tr-row-sum-to-gpu | FileCheck %s

// Milestone 18: tr.load -> tr.reduce_sum(axis=1) fuses to coalesced
// global loads, per-lane register accumulation, and a subgroup reduce.
// The logical 128x128 tile is never a physical temporary. No shared
// memory for the baseline row sum.

// CHECK-LABEL: func.func @row_sum
func.func @row_sum(%in: !tr.buffer<MxKxf32>, %out: !tr.buffer<Mxf32>) {
  %row_blk     = tr.program_id 0 : index
  %c128        = arith.constant 128 : index
  %k           = tr.dim %in, 1 : !tr.buffer<MxKxf32>, index
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

// No 128x128 temporary, no tile alloca, no shared memory.
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.alloca
// CHECK-NOT: gpu.barrier
// CHECK-NOT: gpu.alloc
// CHECK-NOT: #gpu.address_space<workgroup>
// CHECK-NOT: tr.load
// CHECK-NOT: tr.reduce_sum
// CHECK-NOT: tr.for
// CHECK-NOT: linalg.generic

// Fused physical form.
// CHECK: memref.load
// CHECK: arith.addf
// CHECK: gpu.subgroup_reduce add
// CHECK: memref.store
