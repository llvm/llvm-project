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

// -----

module attributes {gpu.container_module} {

  // CHECK: func @dynamic
  func.func @dynamic(%dst : memref<?x?xf32, 1>, %src : memref<?x?xf32>) {
    // CHECK: %[[T0:.*]] = llvm.call @mgpuStreamCreate
    %t0 = gpu.wait async
    %t1 = gpu.memcpy async [%t0] %dst, %src : memref<?x?xf32, 1>, memref<?x?xf32>
    // CHECK: %[[DIM_SIZE_0:.*]] = llvm.extractvalue %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK-NEXT: %[[DIM_SIZE_1:.*]] = llvm.extractvalue %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[NUM_ELEMENTS:.*]] = llvm.mul %[[DIM_SIZE_0]], %[[DIM_SIZE_1]]  : i64
    // CHECK: %[[SIZE_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[NUM_ELEMENTS]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK-NEXT: %[[SIZE_INT:.*]] = llvm.ptrtoint %[[SIZE_PTR]] : !llvm.ptr to i64
    // CHECK: %[[ADDR_CAST:.*]] = llvm.addrspacecast
    // CHECK: llvm.call @mgpuMemcpy(%[[ADDR_CAST]], %{{.*}}, %[[SIZE_INT]], %[[T0]])
    return
  }
}
