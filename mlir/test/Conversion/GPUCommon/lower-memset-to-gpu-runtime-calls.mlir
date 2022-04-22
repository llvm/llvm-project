// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module attributes {gpu.container_module} {

  // CHECK: func @foo
  func.func @foo(%dst : memref<7xf32, 1>, %value : f32) {
    // CHECK: %[[t0:.*]] = llvm.call @mgpuStreamCreate
    %t0 = gpu.wait async
    // CHECK: %[[size_bytes:.*]] = llvm.mlir.constant
    // CHECK: %[[value:.*]] = llvm.bitcast
    // CHECK: %[[dst:.*]] = llvm.bitcast
    // CHECK: llvm.call @mgpuMemset32(%[[dst]], %[[value]], %[[size_bytes]], %[[t0]])
    %t1 = gpu.memset async [%t0] %dst, %value : memref<7xf32, 1>, f32
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[t0]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[t0]])
    gpu.wait [%t1]
    return
  }
}
