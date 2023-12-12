// RUN: mlir-opt %s -pass-pipeline='builtin.module(spirv-attach-target{ver=v1.0 caps=Addresses,Int64,Kernel},convert-gpu-to-spirv{use-64bit-index=true},gpu.module(spirv.module(spirv-lower-abi-attrs,spirv-update-vce)),func.func(llvm-request-c-wrappers),convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,convert-func-to-llvm,gpu-to-llvm{use-bare-pointers-for-kernels=true},gpu-module-to-binary,expand-strided-metadata,lower-affine,finalize-memref-to-llvm,reconcile-unrealized-casts)' \
// RUN: | mlir-cpu-runner \
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
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc host_shared () : memref<4x5xf32>
    memref.copy %arg0, %memref : memref<4x5xf32> to memref<4x5xf32>
    %memref_0 = gpu.alloc host_shared () : memref<4x5xi1>
    %2 = gpu.wait async
    %3 = gpu.launch_func async [%2]  @test_kernel::@test_kernel blocks in (%c4, %c5, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<4x5xf32>, %cst : f32, %memref_0 : memref<4x5xi1>)
    gpu.wait [%3]
    %memref_1 = gpu.alloc host_shared () : memref<4x5xf32>
    %4 = gpu.wait async
    %5 = gpu.launch_func async [%4]  @test_kernel_0::@test_kernel blocks in (%c4, %c5, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<4x5xi1>, %memref : memref<4x5xf32>, %cst : f32, %memref_1 : memref<4x5xf32>)
    gpu.wait [%5]
    %alloc = memref.alloc() : memref<4x5xf32>
    memref.copy %memref_1, %alloc : memref<4x5xf32> to memref<4x5xf32>
    %6 = gpu.wait async
    %7 = gpu.dealloc async [%6] %memref_1 : memref<4x5xf32>
    %8 = gpu.dealloc async [%7] %memref_0 : memref<4x5xi1>
    %9 = gpu.dealloc async [%8] %memref : memref<4x5xf32>
    return %alloc : memref<4x5xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Int8, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<4x5xf32>, %arg1: f32, %arg2: memref<4x5xi1>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 5, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<4x5xf32>
      %3 = arith.cmpf olt, %2, %arg1 : f32
      memref.store %3, %arg2[%0, %1] : memref<4x5xi1>
      gpu.return
    }
  }
  gpu.module @test_kernel_0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Int8, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<4x5xi1>, %arg1: memref<4x5xf32>, %arg2: f32, %arg3: memref<4x5xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 5, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<4x5xi1>
      %3 = memref.load %arg1[%0, %1] : memref<4x5xf32>
      %4 = arith.select %2, %arg2, %3 : f32
      memref.store %4, %arg3[%0, %1] : memref<4x5xf32>
      gpu.return
    }
  }
}
