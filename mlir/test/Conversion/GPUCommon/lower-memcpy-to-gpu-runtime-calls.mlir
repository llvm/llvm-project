// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module attributes {gpu.container_module} {

  // CHECK: func @foo
  func.func @foo(%dst : memref<7xf32, 1>, %src : memref<7xf32>) {
    // CHECK: %[[t0:.*]] = llvm.call @mgpuStreamCreate
    %t0 = gpu.wait async
    // CHECK: %[[size_bytes:.*]] = llvm.mul
    // CHECK-NOT: llvm.addrspacecast
    // CHECK: %[[addr_cast:.*]] = llvm.addrspacecast
    // CHECK: llvm.call @mgpuMemcpy(%[[addr_cast]], %{{.*}}, %[[size_bytes]], %[[t0]])
    %t1 = gpu.memcpy async [%t0] %dst, %src : memref<7xf32, 1>, memref<7xf32>
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[t0]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[t0]])
    gpu.wait [%t1]
    return
  }

  // CHECK: func @test_copy_memref_with_offset
  func.func @test_copy_memref_with_offset(%dst : memref<10xf32, strided<[1], offset: 8>>, %src : memref<10xf32, strided<[1], offset: 3>>) {
    // CHECK: %[[stream:.*]] = llvm.call @mgpuStreamCreate
    %t0 = gpu.wait async
    // CHECK: %[[cst3:.*]] = llvm.mlir.constant(3 : index)
    // CHECK: %[[src:.*]] = llvm.getelementptr %{{.*}}[%[[cst3]]]
    // CHECK: %[[cst8:.*]] = llvm.mlir.constant(8 : index) 
    // CHECK: %[[dst:.*]] = llvm.getelementptr %{{.*}}[%[[cst8]]]
    // CHECK: %[[size_bytes:.*]] = llvm.mul
    // CHECK: llvm.call @mgpuMemcpy(%[[dst]], %[[src]], %[[size_bytes]], %[[stream]])
    %t1 = gpu.memcpy async [%t0] %dst, %src : memref<10xf32, strided<[1], offset: 8>>, memref<10xf32, strided<[1], offset: 3>>
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[stream]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[stream]])
    gpu.wait [%t1]
    return
  }
}
