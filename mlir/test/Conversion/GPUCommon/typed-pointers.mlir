// RUN: mlir-opt %s --gpu-to-llvm='use-opaque-pointers=0' --split-input-file | FileCheck %s

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
    // CHECK: %[[void_ptr:.*]] = llvm.bitcast %[[float_ptr]]
    // CHECK: llvm.call @mgpuMemFree(%[[void_ptr]], %[[stream]])
    %3 = gpu.dealloc async [%2] %1 : memref<?xf32>
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[stream]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[stream]])
    gpu.wait [%3]
    return
  }

  // CHECK: func @foo
  func.func @foo(%dst : memref<7xf32, 1>, %src : memref<7xf32>) {
    // CHECK: %[[t0:.*]] = llvm.call @mgpuStreamCreate
    %t0 = gpu.wait async
    // CHECK: %[[size_bytes:.*]] = llvm.ptrtoint
    // CHECK-NOT: llvm.addrspacecast
    // CHECK: %[[src:.*]] = llvm.bitcast
    // CHECK: %[[addr_cast:.*]] = llvm.addrspacecast
    // CHECK: %[[dst:.*]] = llvm.bitcast %[[addr_cast]]
    // CHECK: llvm.call @mgpuMemcpy(%[[dst]], %[[src]], %[[size_bytes]], %[[t0]])
    %t1 = gpu.memcpy async [%t0] %dst, %src : memref<7xf32, 1>, memref<7xf32>
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[t0]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[t0]])
    gpu.wait [%t1]
    return
  }
}

// -----

module attributes {gpu.container_module} {

  // CHECK: func @memset_f32
  func.func @memset_f32(%dst : memref<7xf32, 1>, %value : f32) {
    // CHECK: %[[t0:.*]] = llvm.call @mgpuStreamCreate
    %t0 = gpu.wait async
    // CHECK: %[[size_bytes:.*]] = llvm.mlir.constant
    // CHECK: %[[value:.*]] = llvm.bitcast
    // CHECK: %[[addr_cast:.*]] = llvm.addrspacecast
    // CHECK: %[[dst:.*]] = llvm.bitcast %[[addr_cast]]
    // CHECK: llvm.call @mgpuMemset32(%[[dst]], %[[value]], %[[size_bytes]], %[[t0]])
    %t1 = gpu.memset async [%t0] %dst, %value : memref<7xf32, 1>, f32
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[t0]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[t0]])
    gpu.wait [%t1]
    return
  }
}

// -----

module attributes {gpu.container_module} {

  // CHECK: func @memset_f16
  func.func @memset_f16(%dst : memref<7xf16, 1>, %value : f16) {
    // CHECK: %[[t0:.*]] = llvm.call @mgpuStreamCreate
    %t0 = gpu.wait async
    // CHECK: %[[size_bytes:.*]] = llvm.mlir.constant
    // CHECK: %[[value:.*]] = llvm.bitcast
    // CHECK: %[[addr_cast:.*]] = llvm.addrspacecast
    // CHECK: %[[dst:.*]] = llvm.bitcast %[[addr_cast]]
    // CHECK: llvm.call @mgpuMemset16(%[[dst]], %[[value]], %[[size_bytes]], %[[t0]])
    %t1 = gpu.memset async [%t0] %dst, %value : memref<7xf16, 1>, f16
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[t0]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[t0]])
    gpu.wait [%t1]
    return
  }
}
