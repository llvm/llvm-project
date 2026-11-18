// RUN: tr-opt %s --tr-emit-gpu-kernels | FileCheck %s --check-prefix=BEFORE
// RUN: tr-opt %s --tr-emit-gpu-kernels --tr-split-host-device | FileCheck %s --check-prefix=AFTER

// Milestone 28: a GPU program is not one linear function.
// BEFORE is GPU IR after outlining (host launch + device module, no target).
// AFTER records the split: #nvvm.target on the device module, tr.role on
// host launch ops vs device kernels.

// BEFORE-NOT: nvvm.target
// BEFORE-NOT: tr.role
// BEFORE-NOT: tr.host_device_split
// BEFORE: gpu.module @tr_kernels
// BEFORE: gpu.func @row_sum_kernel
// BEFORE-SAME: kernel
// BEFORE: gpu.func @full_sum_stage1
// BEFORE: gpu.func @full_sum_stage2
// BEFORE: gpu.func @column_sum_kernel
// BEFORE: func.func @row_sum
// BEFORE: gpu.launch_func @tr_kernels::@row_sum_kernel
// BEFORE: func.func @full_sum
// BEFORE: gpu.launch_func @tr_kernels::@full_sum_stage1
// BEFORE: gpu.launch_func @tr_kernels::@full_sum_stage2
// BEFORE: func.func @column_sum
// BEFORE: gpu.launch_func @tr_kernels::@column_sum_kernel

// AFTER: module attributes
// AFTER-SAME: gpu.container_module
// AFTER-SAME: tr.host_device_split
// AFTER: gpu.module @tr_kernels
// AFTER-SAME: #nvvm.target
// AFTER-SAME: chip = "sm_80"

// AFTER: gpu.func @row_sum_kernel
// AFTER-SAME: kernel
// AFTER-SAME: tr.role = "device"
// AFTER: gpu.block_id x
// AFTER: gpu.lane_id
// AFTER: gpu.subgroup_reduce add
// AFTER: gpu.return

// AFTER: gpu.func @full_sum_stage1
// AFTER-SAME: workgroup(%{{.*}}: memref<8xf32, #gpu.address_space<workgroup>>)
// AFTER-SAME: kernel
// AFTER-SAME: tr.role = "device"
// AFTER: gpu.barrier
// AFTER: gpu.func @full_sum_stage2
// AFTER-SAME: kernel
// AFTER-SAME: tr.role = "device"

// AFTER: gpu.func @column_sum_kernel
// AFTER-SAME: workgroup(%{{.*}}: memref<128x128xf32, #gpu.address_space<workgroup>>)
// AFTER-SAME: kernel
// AFTER-SAME: tr.role = "device"
// AFTER: gpu.barrier

// AFTER: func.func @row_sum
// AFTER-SAME: tr.role = "host"
// AFTER: gpu.launch_func @tr_kernels::@row_sum_kernel
// AFTER-SAME: blocks in
// AFTER-SAME: threads in

// AFTER: func.func @full_sum
// AFTER-SAME: tr.role = "host"
// AFTER: gpu.launch_func @tr_kernels::@full_sum_stage1
// AFTER: gpu.launch_func @tr_kernels::@full_sum_stage2

// AFTER: func.func @column_sum
// AFTER-SAME: tr.role = "host"
// AFTER: gpu.launch_func @tr_kernels::@column_sum_kernel
// AFTER-NOT: gpu.launch {{.*}}blocks(

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
