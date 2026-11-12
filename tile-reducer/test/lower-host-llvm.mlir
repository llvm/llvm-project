// RUN: tr-opt %s --tr-emit-gpu-kernels --tr-split-host-device --tr-lower-device-to-nvvm --tr-lower-host-to-llvm | FileCheck %s

// Milestone 29 host path: launch/runtime ops toward LLVM dialect.
// Device must be NVVM/LLVM first so launch_func operand packing matches
// the lowered kernel ABI. Uses the upstream GPU runtime ABI, not a
// TileReducer-specific calling convention.

// CHECK: gpu.module @tr_kernels
// CHECK: llvm.func @row_sum_kernel
// CHECK-SAME: nvvm.kernel
// CHECK: llvm.func @full_sum_stage1
// CHECK: llvm.func @full_sum_stage2
// CHECK: llvm.func @column_sum_kernel

// CHECK: llvm.func @row_sum
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: tr.role = "host"
// CHECK: gpu.launch_func @tr_kernels::@row_sum_kernel
// CHECK-SAME: args(%{{.*}} : !llvm.ptr

// CHECK: llvm.func @full_sum
// CHECK-SAME: tr.role = "host"
// CHECK: gpu.launch_func @tr_kernels::@full_sum_stage1
// CHECK: gpu.launch_func @tr_kernels::@full_sum_stage2

// CHECK: llvm.func @column_sum
// CHECK-SAME: tr.role = "host"
// CHECK: gpu.launch_func @tr_kernels::@column_sum_kernel

// CHECK-NOT: func.func @
// CHECK-NOT: tr.runtime

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

func.func @full_sum(%in: !tr.buffer<MxKxf32>, %out: !tr.buffer<1xf32>) {
  %blk  = tr.program_id 0 : index
  %c128 = arith.constant 128 : index
  %c0   = arith.constant 0 : index
  %k    = tr.dim %in, 1 : !tr.buffer<MxKxf32>, index
  %num  = arith.divui %k, %c128 : index
  %zero = tr.constant 0.0 : !tr.tile<f32>
  %result = tr.for %kt = 0 to %num step 1
      iter_args(%acc = %zero) -> !tr.tile<f32> {
    %t    = tr.load %in[%blk, %kt]
        : !tr.buffer<MxKxf32>, !tr.tile<128x128xf32>
    %row  = tr.reduce_sum %t, axis = 1
        : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
    %s    = tr.reduce_sum %row, axis = 0
        : !tr.tile<128xf32> -> !tr.tile<f32>
    %acc2 = tr.add %acc, %s : !tr.tile<f32>
    tr.yield %acc2 : !tr.tile<f32>
  }
  tr.store %out[%c0], %result : !tr.buffer<1xf32>, !tr.tile<f32>
  return
}

func.func @column_sum(%in: !tr.buffer<MxKxf32>, %out: !tr.buffer<Kxf32>) {
  %col_blk = tr.program_id 0 : index
  %c128    = arith.constant 128 : index
  %m       = tr.dim %in, 0 : !tr.buffer<MxKxf32>, index
  %num     = arith.divui %m, %c128 : index
  %zero    = tr.constant 0.0 : !tr.tile<128xf32>
  %result = tr.for %mt = 0 to %num step 1
      iter_args(%acc = %zero) -> !tr.tile<128xf32> {
    %t       = tr.load %in[%mt, %col_blk]
        : !tr.buffer<MxKxf32>, !tr.tile<128x128xf32>
    %partial = tr.reduce_sum %t, axis = 0
        : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
    %acc2    = tr.add %acc, %partial : !tr.tile<128xf32>
    tr.yield %acc2 : !tr.tile<128xf32>
  }
  tr.store %out[%col_blk], %result : !tr.buffer<Kxf32>, !tr.tile<128xf32>
  return
}
