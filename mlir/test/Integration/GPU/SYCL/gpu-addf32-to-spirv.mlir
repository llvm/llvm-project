// RUN: mlir-opt %s -pass-pipeline='builtin.module(spirv-attach-target{ver=v1.0 caps=Addresses,Int64,Kernel},convert-gpu-to-spirv{use-64bit-index=true},gpu.module(spirv.module(spirv-lower-abi-attrs,spirv-update-vce)),func.func(llvm-request-c-wrappers),convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,convert-func-to-llvm,gpu-to-llvm{use-bare-pointers-for-kernels=true},gpu-module-to-binary,expand-strided-metadata,lower-affine,finalize-memref-to-llvm,reconcile-unrealized-casts)' \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_sycl_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module @add attributes {gpu.container_module} {
  memref.global "private" constant @__constant_2x2x2xf32_0 : memref<2x2x2xf32> = dense<[[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8 ]]]>
  memref.global "private" constant @__constant_2x2x2xf32 : memref<2x2x2xf32> = dense<[[[1.2, 2.3], [4.5, 5.8]], [[7.2, 8.3], [10.5, 11.8]]]>
  func.func @main() {
    %0 = memref.get_global @__constant_2x2x2xf32 : memref<2x2x2xf32>
    %1 = memref.get_global @__constant_2x2x2xf32_0 : memref<2x2x2xf32>
    %2 = call @test(%0, %1) : (memref<2x2x2xf32>, memref<2x2x2xf32>) -> memref<2x2x2xf32>
    %cast = memref.cast %2 : memref<2x2x2xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @test(%arg0: memref<2x2x2xf32>, %arg1: memref<2x2x2xf32>) -> memref<2x2x2xf32> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %mem = gpu.alloc host_shared () : memref<2x2x2xf32>
    memref.copy %arg1, %mem : memref<2x2x2xf32> to memref<2x2x2xf32>
    %memref_0 = gpu.alloc host_shared () : memref<2x2x2xf32>
    memref.copy %arg0, %memref_0 : memref<2x2x2xf32> to memref<2x2x2xf32>
    %memref_2 = gpu.alloc host_shared () : memref<2x2x2xf32>
    %2 = gpu.wait async
    %3 = gpu.launch_func async [%2] @test_kernel::@test_kernel blocks in (%c2, %c2, %c2) threads in (%c1, %c1, %c1) args(%memref_0 : memref<2x2x2xf32>, %mem : memref<2x2x2xf32>, %memref_2 : memref<2x2x2xf32>)
    gpu.wait [%3]
    %alloc = memref.alloc() : memref<2x2x2xf32>
    memref.copy %memref_2, %alloc : memref<2x2x2xf32> to memref<2x2x2xf32>
    %4 = gpu.wait async
    %5 = gpu.dealloc async [%4] %memref_2 : memref<2x2x2xf32>
    %6 = gpu.dealloc async [%5] %memref_0 : memref<2x2x2xf32>
    %7 = gpu.dealloc async [%6] %mem : memref<2x2x2xf32>
    gpu.wait [%7]
    return %alloc : memref<2x2x2xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<2x2x2xf32>, %arg1: memref<2x2x2xf32>, %arg2: memref<2x2x2xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 2, 2, 2>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = memref.load %arg0[%0, %1, %2] : memref<2x2x2xf32>
      %4 = memref.load %arg1[%0, %1, %2] : memref<2x2x2xf32>
      %5 = arith.addf %3, %4 : f32
      memref.store %5, %arg2[%0, %1, %2] : memref<2x2x2xf32>
      gpu.return
    }
  }
  // CHECK: [2.3, 4.5]
  // CHECK: [7.8, 10.2]
  // CHECK: [12.7, 14.9]
  // CHECK: [18.2, 20.6]
}
