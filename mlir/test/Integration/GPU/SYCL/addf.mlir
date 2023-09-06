// RUN: mlir-opt %s -pass-pipeline='builtin.module(convert-gpu-to-spirv{use-64bit-index=true use-opencl=true},spirv.module(spirv-lower-abi-attrs,spirv-update-vce),func.func(llvm-request-c-wrappers),gpu-serialize-to-spirv,convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,convert-func-to-llvm,gpu-to-llvm{use-bare-pointers-for-kernels=true},expand-strided-metadata,lower-affine,finalize-memref-to-llvm,reconcile-unrealized-casts)' \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_sycl_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module @add attributes {
  gpu.container_module
} {
  memref.global "private" constant @__constant_9xf32_0 : memref<9xf32> = dense<[1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]>
  memref.global "private" constant @__constant_9xf32 : memref<9xf32> = dense<[2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2]>
  func.func @main() {
    %0 = memref.get_global @__constant_9xf32 : memref<9xf32>
    %1 = memref.get_global @__constant_9xf32_0 : memref<9xf32>
    %2 = call @test(%0, %1) : (memref<9xf32>, memref<9xf32>) -> memref<9xf32>
    %cast = memref.cast %2 : memref<9xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @test(%arg0: memref<9xf32>, %arg1: memref<9xf32>) -> memref<9xf32> {
  %c9 = arith.constant 9 : index
  %c1 = arith.constant 1 : index
  %0 = gpu.wait async
  %memref, %asyncToken = gpu.alloc async [%0] host_shared (): memref<9xf32>
  gpu.wait [%asyncToken]
  memref.copy %arg1, %memref : memref<9xf32> to memref<9xf32>
  %1 = gpu.wait async
  %memref_0, %asyncToken_1 = gpu.alloc async [%1] host_shared () : memref<9xf32>
  gpu.wait [%asyncToken_1]
  memref.copy %arg0, %memref_0 : memref<9xf32> to memref<9xf32>
  %2 = gpu.wait async
  %memref_2, %asyncToken_3 = gpu.alloc async [%2] host_shared () : memref<9xf32>
  %3 = gpu.launch_func async [%asyncToken_3] @test_kernel::@test_kernel blocks in (%c9, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<9xf32>, %memref : memref<9xf32>, %memref_2 : memref<9xf32>)
  gpu.wait [%3]
  %alloc = memref.alloc() : memref<9xf32>
  memref.copy %memref_2, %alloc : memref<9xf32> to memref<9xf32>
  %4 = gpu.wait async
  %5 = gpu.dealloc async [%4] %memref_2 : memref<9xf32>
  %6 = gpu.dealloc async [%5] %memref_0 : memref<9xf32>
  %7 = gpu.dealloc async [%6] %memref : memref<9xf32>
  gpu.wait [%7]
  return %alloc : memref<9xf32>
  }
  gpu.module @test_kernel attributes {
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>
  }{
    gpu.func @test_kernel(%arg0: memref<9xf32>, %arg1: memref<9xf32>, %arg2: memref<9xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 9, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      //%0 = gpu.block_id  x
      //%2 = memref.load %arg0[%0] : memref<9xf32>
      //%3 = memref.load %arg1[%0] : memref<9xf32>
      //%4 = arith.addf %2, %3 : f32
      //memref.store %4, %arg2[%0] : memref<9xf32>
      gpu.return
    }
  }
  // CHECK: [3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3]
}
