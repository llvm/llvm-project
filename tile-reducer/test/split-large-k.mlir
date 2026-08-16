// RUN: tr-opt %s --tr-emit-gpu-kernels=k-splits=8 | FileCheck %s

// Milestone 24: M=1, K=1e8 has too little parallelism as one block.
// One logical program (gpu.block_id x) is refined into many physical
// blocks (gpu.block_id y) that write partials, then a second stage
// reduces them. logical program_id != necessarily one GPU block.

// CHECK: gpu.func @row_sum_splitk_stage1
// CHECK: gpu.block_id x
// CHECK: gpu.block_id y
// CHECK: gpu.grid_dim y
// CHECK: gpu.func @row_sum_splitk_stage2
// CHECK: func.func @row_sum
// CHECK: gpu.launch_func @tr_kernels::@row_sum_splitk_stage1
// CHECK-SAME: blocks in
// CHECK: gpu.launch_func @tr_kernels::@row_sum_splitk_stage2

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
