// RUN: tr-opt %s --tr-emit-gpu-kernels | FileCheck %s

// Milestone 20: gpu.module @tr_kernels, SymbolTable insert/lookup, and a
// nested SymbolRefAttr on gpu.launch_func (@tr_kernels::@row_sum_kernel).
// The host launches by symbol, not by a string name.

// CHECK: module attributes {gpu.container_module}
// CHECK: gpu.module @tr_kernels
// CHECK: gpu.func @row_sum_kernel
// CHECK-SAME: kernel
// CHECK: gpu.block_id x
// CHECK: gpu.lane_id
// CHECK: gpu.subgroup_id
// CHECK: gpu.subgroup_reduce add
// CHECK: gpu.return

// CHECK: func.func @row_sum
// CHECK: gpu.launch_func @tr_kernels::@row_sum_kernel
// CHECK-SAME: blocks in
// CHECK-SAME: threads in

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
