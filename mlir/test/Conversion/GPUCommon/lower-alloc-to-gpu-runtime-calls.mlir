// RUN: mlir-opt -allow-unregistered-dialect %s --gpu-to-llvm | FileCheck %s

module attributes {gpu.container_module} {
  func @main() {
    // CHECK: %[[stream:.*]] = llvm.call @mgpuStreamCreate()
    %0 = gpu.wait async
    // CHECK: %[[size_bytes:.*]] = llvm.ptrtoint
    // CHECK: llvm.call @mgpuMemAlloc(%[[size_bytes]], %[[stream]])
    %1, %2 = gpu.alloc async [%0] () : memref<13xf32>
    // CHECK: %[[float_ptr:.*]] = llvm.extractvalue {{.*}}[0]
    // CHECK: %[[void_ptr:.*]] = llvm.bitcast %[[float_ptr]]
    // CHECK: llvm.call @mgpuMemFree(%[[void_ptr]], %[[stream]])
    %3 = gpu.dealloc async [%2] %1 : memref<13xf32>
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[stream]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[stream]])
    gpu.wait [%3]
    return
  }
}
