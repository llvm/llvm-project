// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module attributes {gpu.container_module} {

  func @foo() {
    // CHECK: %[[t0:.*]] = llvm.call @mgpuStreamCreate
    // CHECK: %[[e0:.*]] = llvm.call @mgpuEventCreate
    // CHECK: llvm.call @mgpuEventRecord(%[[e0]], %[[t0]])
    %t0 = gpu.wait async
    // CHECK: %[[t1:.*]] = llvm.call @mgpuStreamCreate
    // CHECK: llvm.call @mgpuStreamWaitEvent(%[[t1]], %[[e0]])
    // CHECK: llvm.call @mgpuEventDestroy(%[[e0]])
    %t1 = gpu.wait async [%t0]
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[t0]])
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[t1]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[t0]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[t1]])
    gpu.wait [%t0, %t1]
    return
  }
}
