// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL: llvm.func @main
  // CHECK-SAME: %[[size:.*]]: i64
  func.func @main(%size : index) {
    // CHECK: %[[stream:.*]] = llvm.call @mgpuStreamCreate()
    %0 = gpu.wait async
    // CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}}[%[[size]]]
    // CHECK: %[[size_bytes:.*]] = ptr.ptrtoint %[[gep]]
    // CHECK: %[[isHostShared:.*]] = llvm.mlir.constant 
    // CHECK: llvm.call @mgpuMemAlloc(%[[size_bytes]], %[[stream]], %[[isHostShared]])
    %1, %2 = gpu.alloc async [%0] (%size) : memref<?xf32>
    // CHECK: %[[float_ptr:.*]] = llvm.extractvalue {{.*}}[0]
    // CHECK: llvm.call @mgpuMemFree(%[[float_ptr]], %[[stream]])
    %3 = gpu.dealloc async [%2] %1 : memref<?xf32>
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[stream]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[stream]])
    gpu.wait [%3]
    return
  }

  // CHECK-LABEL: llvm.func @alloc_sync
  // CHECK-SAME: %[[size:.*]]: i64
  func.func @alloc_sync(%size : index) {
    // CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}}[%[[size]]]
    // CHECK: %[[size_bytes:.*]] = ptr.ptrtoint %[[gep]]
    // CHECK: %[[nullptr:.*]] = llvm.mlir.zero
    // CHECK: %[[isHostShared:.*]] = llvm.mlir.constant 
    // CHECK: llvm.call @mgpuMemAlloc(%[[size_bytes]], %[[nullptr]], %[[isHostShared]])
    %0 = gpu.alloc host_shared (%size) : memref<?xf32>
    // CHECK: %[[stream:.*]] = llvm.call @mgpuStreamCreate()
    %1 = gpu.wait async
    %2 = gpu.dealloc async [%1] %0 : memref<?xf32>
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[stream]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[stream]])
    gpu.wait [%2]
    return
  }
}
