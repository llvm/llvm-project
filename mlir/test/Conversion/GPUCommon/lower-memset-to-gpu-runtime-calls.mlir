// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module attributes {gpu.container_module} {

  // CHECK: func @foo
  func.func @foo(%dst : memref<7xf32, 1>, %value : f32) {
    // CHECK: %[[t0:.*]] = llvm.call @mgpuStreamCreate
    %t0 = gpu.wait async
    // CHECK: %[[size_bytes:.*]] = llvm.mlir.constant
    // CHECK: %[[addr_cast:.*]] = llvm.addrspacecast
    // CHECK: llvm.call @mgpuMemset32(%[[addr_cast]], %{{.*}}, %[[size_bytes]], %[[t0]])
    %t1 = gpu.memset async [%t0] %dst, %value : memref<7xf32, 1>, f32
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[t0]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[t0]])
    gpu.wait [%t1]
    return
  }
}

module attributes {gpu.container_module} {

  // CHECK: func @foo16
  func.func @foo16(%dst : memref<7xf16, 1>, %value : f16) {
    // CHECK: %[[t0:.*]] = llvm.call @mgpuStreamCreate
    %t0 = gpu.wait async
    // CHECK: %[[size_bytes:.*]] = llvm.mlir.constant
    // CHECK: %[[addr_cast:.*]] = llvm.addrspacecast
    // CHECK: llvm.call @mgpuMemset16(%[[addr_cast]], %{{.*}}, %[[size_bytes]], %[[t0]])
    %t1 = gpu.memset async [%t0] %dst, %value : memref<7xf16, 1>, f16
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[t0]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[t0]])
    gpu.wait [%t1]
    return
  }
}
