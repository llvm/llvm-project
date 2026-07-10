// RUN: mlir-opt %s --gpu-to-llvm -split-input-file | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL: llvm.func @main
  // CHECK-SAME: %[[size:.*]]: i64
  func.func @main(%size : index) {
    // CHECK: %[[stream:.*]] = llvm.call @mgpuStreamCreate()
    %0 = gpu.wait async
    // CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}}[%[[size]]]
    // CHECK: %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]]
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

  // CHECK-LABEL: llvm.func @alloc_dealloc_sync
  // CHECK-SAME: %[[size:.*]]: i64
  func.func @alloc_dealloc_sync(%size : index) {
    // CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}}[%[[size]]]
    // CHECK: %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]]
    // CHECK: %[[nullptr:.*]] = llvm.mlir.zero
    // CHECK: %[[isHostShared:.*]] = llvm.mlir.constant
    // CHECK: llvm.call @mgpuMemAlloc(%[[size_bytes]], %[[nullptr]], %[[isHostShared]])
    %0 = gpu.alloc (%size) : memref<?xf32>
    // CHECK: %[[float_ptr:.*]] = llvm.extractvalue {{.*}}[0]
    // CHECK: %[[nullptr2:.*]] = llvm.mlir.zero
    // CHECK: llvm.call @mgpuMemFree(%[[float_ptr]], %[[nullptr2]])
    gpu.dealloc %0 : memref<?xf32>
    return
  }

  // CHECK-LABEL: llvm.func @alloc_sync
  // CHECK-SAME: %[[size:.*]]: i64
  func.func @alloc_sync(%size : index) {
    // CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}}[%[[size]]]
    // CHECK: %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]]
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

// -----

// More than one async dependency is not supported; the alloc and dealloc
// should be left unconverted.
module attributes {gpu.container_module} {
  // CHECK-LABEL: func @multi_dep_unsupported
  func.func @multi_dep_unsupported(%size : index) {
    %t1 = gpu.wait async
    %t2 = gpu.wait async
    // CHECK: gpu.alloc async [{{.*}}, {{.*}}]
    // CHECK-NOT: mgpuMemAlloc
    %buf, %t3 = gpu.alloc async [%t1, %t2] (%size) : memref<?xf32>
    // CHECK: gpu.dealloc async [{{.*}}, {{.*}}]
    // CHECK-NOT: mgpuMemFree
    %t4 = gpu.dealloc async [%t3, %t1] %buf : memref<?xf32>
    gpu.wait [%t4]
    return
  }
}
