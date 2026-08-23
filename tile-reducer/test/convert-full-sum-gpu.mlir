// RUN: tr-opt %s --tr-emit-gpu-kernels | FileCheck %s

// Milestone 21: MxK -> scalar via two kernels inside @tr_kernels.
//   thread local sum -> warp reduce -> smem[warp] -> barrier -> block reduce
//   @full_sum_stage1 writes one partial per block
//   @full_sum_stage2 reduces the partials
// Prefer two-stage reduction over unordered FP atomics.

// CHECK: gpu.module @tr_kernels
// CHECK: gpu.func @full_sum_stage1
// CHECK-SAME: workgroup(%{{.*}}: memref<8xf32, #gpu.address_space<workgroup>>)
// CHECK-SAME: kernel
// CHECK: gpu.subgroup_reduce add
// CHECK: gpu.barrier
// CHECK: gpu.func @full_sum_stage2
// CHECK-SAME: kernel
// CHECK: gpu.launch_func @tr_kernels::@full_sum_stage1
// CHECK: gpu.launch_func @tr_kernels::@full_sum_stage2
// CHECK-NOT: gpu.atomic

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
