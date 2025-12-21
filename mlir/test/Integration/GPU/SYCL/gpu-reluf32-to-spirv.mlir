// RUN: mlir-opt %s -pass-pipeline='builtin.module(spirv-attach-target{ver=v1.0 caps=Addresses,Int64,Kernel},func.func(gpu-async-region),convert-gpu-to-spirv{use-64bit-index=true},gpu.module(spirv.module(spirv-lower-abi-attrs,spirv-update-vce)),func.func(llvm-request-c-wrappers),convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,convert-func-to-llvm,gpu-to-llvm{use-bare-pointers-for-kernels=true},gpu-module-to-binary,expand-strided-metadata,lower-affine,finalize-memref-to-llvm,reconcile-unrealized-casts)' \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_sycl_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module @relu attributes {gpu.container_module} {
  memref.global "private" constant @__constant_4x5xf32 : memref<4x5xf32> = dense<[
    [-1.000000e-01, -2.000000e-01, -3.000000e-01, 4.000000e-01, 5.000000e-01],
    [1.000000e-01, -2.000000e-01, 3.000000e-01, -4.000000e-01, 5.000000e-01],
    [1.000000e-01, 2.000000e-01, 3.000000e-01, -4.000000e-01, -5.000000e-01],
    [1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01]
  ]>

  func.func @main() {
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_4x5xf32 : memref<4x5xf32>

    scf.for %arg0 = %c0 to %c100 step %c1 {
      %1 = func.call @test(%0) : (memref<4x5xf32>) -> memref<4x5xf32>
      %cast = memref.cast %1 : memref<4x5xf32> to memref<*xf32>
      func.call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
      // CHECK: [0, 0, 0, 0.4, 0.5],
      // CHECK: [0.1, 0, 0.3, 0, 0.5],
      // CHECK: [0.1, 0.2, 0.3, 0, 0],
      // CHECK: [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    return
  }

  func.func private @printMemrefF32(memref<*xf32>)
  func.func @test(%arg0: memref<4x5xf32>) -> memref<4x5xf32> {
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %host_result = memref.alloc() : memref<4x5xf32>
    %gpu_input = gpu.alloc() : memref<4x5xf32>
    gpu.memcpy %gpu_input, %arg0 : memref<4x5xf32>, memref<4x5xf32>
    %gpu_result = gpu.alloc() : memref<4x5xf32>
    gpu.launch_func @test_kernel::@test_relu blocks in (%c4, %c5, %c1) threads in (%c1, %c1, %c1) args(%gpu_input : memref<4x5xf32>,  %gpu_result : memref<4x5xf32>)
    gpu.memcpy %host_result, %gpu_result : memref<4x5xf32>, memref<4x5xf32>
    gpu.dealloc %gpu_input : memref<4x5xf32>
    gpu.dealloc %gpu_result : memref<4x5xf32>
    return %host_result : memref<4x5xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Int8, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_relu(%arg0: memref<4x5xf32>, %arg1: memref<4x5xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 5, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %zero = arith.constant 0.000000e+00 : f32
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<4x5xf32>
      %3 = arith.cmpf ogt, %2, %zero : f32
      %4 = arith.select %3, %2, %zero : f32
      memref.store %4, %arg1[%0, %1] : memref<4x5xf32>
      gpu.return
    }
  }
}
