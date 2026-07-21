
// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(gpu.func(acc-cg-to-gpu)))" | FileCheck %s

// Verify that a seq routine uses stack allocation (memref.alloca) for
// privatized variables instead of workgroup shared memory + barriers.
// This is a regression test for barriers in a sequential (single-thread)
// kernel: workgroup barriers there are unnecessary and can miscompile.
//
// After ParallelDimAssignment, a seq kernel keeps seqDim as the sole
// launch arg (internal scf.parallel loops get sequential par_dims).

module attributes {gpu.container_module} {

  acc.routine @routine_seq func(@seq_routine) seq

  gpu.module @cuda_device_mod {

    // CHECK-LABEL: gpu.func @seq_routine
    // CHECK-NOT: workgroup
    // CHECK-NOT: nvvm.barrier
    // CHECK-NOT: gpu.barrier
    gpu.func @seq_routine()
        attributes {acc.specialized_routine = #acc.specialized_routine<@routine_seq, <seq>, "seq_routine">} {
      %c1 = arith.constant 1 : index
      %0 = acc.par_width %c1 {par_dim = #acc.par_dim<sequential>}
      // CHECK: memref.alloca() : memref<10xi32>
      // CHECK-NOT: memref.store {{.*}} #gpu.address_space<workgroup>
      %1 = acc.privatize : () -> !acc.private_type<memref<10xi32>>
      acc.compute_region launch(%arg0 = %0) ins(%arg10 = %1) : (!acc.private_type<memref<10xi32>>) {
        %2 = acc.private_local %arg10 : (!acc.private_type<memref<10xi32>>) -> memref<10xi32>
        acc.yield
      } {origin = "acc.routine"}
      gpu.return
    }

  }
}
