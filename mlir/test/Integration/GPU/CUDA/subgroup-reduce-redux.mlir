// RUN: mlir-opt %s \
// RUN:  | mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 cubin-format=isa" \
// RUN:  | FileCheck %s

// A subgroup reduction the source states is executed by the whole subgroup
// becomes one redux.sync instruction, once the chip the pipeline serializes
// for is one that has it. Below sm_80 the pipeline still cannot lower the
// operation at all, since it runs no shuffle expansion either.

// CHECK: redux.sync.add.s32

gpu.module @kernels {
  gpu.func @reduce(%out: memref<32xi32>) kernel {
    %tid = gpu.thread_id x
    %v = arith.index_cast %tid : index to i32
    %r = gpu.subgroup_reduce add %v uniform : (i32) -> (i32)
    memref.store %r, %out[%tid] : memref<32xi32>
    gpu.return
  }
}
