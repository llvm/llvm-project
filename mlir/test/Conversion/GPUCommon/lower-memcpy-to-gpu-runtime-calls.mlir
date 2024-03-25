// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module attributes {gpu.container_module} {

  // CHECK: func @foo
  func.func @foo(%dst : memref<7xf32, 1>, %src : memref<7xf32>) {
    // CHECK: %[[t0:.*]] = llvm.call @mgpuStreamCreate
    %t0 = gpu.wait async
    // CHECK: %[[size_bytes:.*]] = llvm.ptrtoint
    // CHECK-NOT: llvm.addrspacecast
    // CHECK: %[[addr_cast:.*]] = llvm.addrspacecast
    // CHECK: llvm.call @mgpuMemcpy(%[[addr_cast]], %{{.*}}, %[[size_bytes]], %[[t0]])
    %t1 = gpu.memcpy async [%t0] %dst, %src : memref<7xf32, 1>, memref<7xf32>
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[t0]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[t0]])
    gpu.wait [%t1]
    return
  }
}
