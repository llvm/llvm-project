// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(gpu.func(acc-cg-to-gpu)))" | FileCheck %s

// CHECK-LABEL: gpu.func @test()
// CHECK: %[[PRIVATIZED_MEMREF:[0-9]+]] = memref.load {{.*}} : memref<memref<4xi8>, #gpu.address_space<workgroup>>

module attributes {gpu.container_module} {
  acc.routine @test_routine func(@test) worker

  gpu.module @cuda_device_mod {
    gpu.func @test()
        attributes {acc.specialized_routine = #acc.specialized_routine<@test_routine, <worker>, "test">} {
      %c1 = arith.constant 1 : index
      %1 = acc.par_width %c1 {par_dim = #acc.par_dim<thread_x>}
      %2 = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
      %4 = acc.privatize : () -> !acc.private_type<memref<4xi8>>
      acc.compute_region launch(%arg3 = %1, %arg4 = %2) ins(%arg10 = %4) : (!acc.private_type<memref<4xi8>>) {
        %8 = acc.private_local %arg10 : (!acc.private_type<memref<4xi8>>) -> memref<f32>
        acc.yield
      } {origin = "acc.routine"}
      gpu.return
    }
  }
}
