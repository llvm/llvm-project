// RUN: mlir-opt %s -pass-pipeline='builtin.module(spirv-attach-target{ver=v1.0 caps=Addresses,Int64,Kernel},convert-gpu-to-spirv{use-64bit-index=true},gpu.module(spirv.module(spirv-lower-abi-attrs,spirv-update-vce)),func.func(llvm-request-c-wrappers),convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,convert-func-to-llvm,gpu-to-llvm{use-bare-pointers-for-kernels=true},gpu-module-to-binary,expand-strided-metadata,lower-affine,finalize-memref-to-llvm,reconcile-unrealized-casts)' \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_sycl_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module @add attributes {gpu.container_module} {
  memref.global "private" constant @__constant_3x3xi64_0 : memref<3x3xi64> = dense<[[1, 4098, 3], [16777220, 5, 4294967302], [7, 1099511627784, 9]]>
  memref.global "private" constant @__constant_3x3xi64 : memref<3x3xi64> = dense<[[1, 2, 3], [4, 5, 4102], [16777223, 4294967304, 1099511627785]]>
  func.func @main() {
    %0 = memref.get_global @__constant_3x3xi64 : memref<3x3xi64>
    %1 = memref.get_global @__constant_3x3xi64_0 : memref<3x3xi64>
    %2 = call @test(%0, %1) : (memref<3x3xi64>, memref<3x3xi64>) -> memref<3x3xi64>
    %cast = memref.cast %2 : memref<3x3xi64> to memref<*xi64>
    call @printMemrefI64(%cast) : (memref<*xi64>) -> ()
    return
  }
  func.func private @printMemrefI64(memref<*xi64>)
  func.func @test(%arg0: memref<3x3xi64>, %arg1: memref<3x3xi64>) -> memref<3x3xi64> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %mem = gpu.alloc host_shared () : memref<3x3xi64>
    memref.copy %arg1, %mem : memref<3x3xi64> to memref<3x3xi64>
    %memref_0 = gpu.alloc host_shared () : memref<3x3xi64>
    memref.copy %arg0, %memref_0 : memref<3x3xi64> to memref<3x3xi64>
    %memref_2 = gpu.alloc host_shared () : memref<3x3xi64>
    %2 = gpu.wait async
    %3 = gpu.launch_func async [%2] @test_kernel::@test_kernel blocks in (%c3, %c3, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<3x3xi64>, %mem : memref<3x3xi64>, %memref_2 : memref<3x3xi64>)
    gpu.wait [%3]
    %alloc = memref.alloc() : memref<3x3xi64>
    memref.copy %memref_2, %alloc : memref<3x3xi64> to memref<3x3xi64>
    %4 = gpu.wait async
    %5 = gpu.dealloc async [%4] %memref_2 : memref<3x3xi64>
    %6 = gpu.dealloc async [%5] %memref_0 : memref<3x3xi64>
    %7 = gpu.dealloc async [%6] %mem : memref<3x3xi64>
    gpu.wait [%7]
    return %alloc : memref<3x3xi64>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Int64, Kernel], []>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<3x3xi64>, %arg1: memref<3x3xi64>, %arg2: memref<3x3xi64>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 3, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<3x3xi64>
      %3 = memref.load %arg1[%0, %1] : memref<3x3xi64>
      %4 = arith.addi %2, %3 : i64
      memref.store %4, %arg2[%0, %1] : memref<3x3xi64>
      gpu.return
    }
  }
  // CHECK: [2,   4100,   6],
  // CHECK: [16777224,   10,   4294971404],
  // CHECK: [16777230,   1103806595088,   1099511627794]
}
