// RUN: mlir-opt %s --gpu-to-llvm -split-input-file | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK: gpu.module
  gpu.module @kernel_module [#nvvm.target] {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }

  func.func @foo(%buffer: memref<?xf32>) {
  // CHECK: [[C8:%.*]] = llvm.mlir.constant(8 : index) : i64
  // CHECK: [[C32:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: [[C256:%.*]] = llvm.mlir.constant(256 : i32) : i32
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : i32
    %c256 = arith.constant 256 : i32

  // CHECK: gpu.launch_func @kernel_module::@kernel
  // CHECK: blocks in ([[C8]], [[C8]], [[C8]]) threads in ([[C8]], [[C8]], [[C8]]) : i64
  // CHECK: dynamic_shared_memory_size [[C256]]
  // CHECK: args([[C32]] : i32, %{{.*}} : !llvm.ptr, %{{.*}} : !llvm.ptr, %{{.*}} : i64, %{{.*}} : i64, %{{.*}} : i64)
    gpu.launch_func @kernel_module::@kernel
        blocks in (%c8, %c8, %c8)
        threads in (%c8, %c8, %c8)
        dynamic_shared_memory_size %c256
        args(%c32 : i32, %buffer : memref<?xf32>)
    return
  }
}


// -----

module attributes {gpu.container_module} {
  // CHECK: gpu.module
  gpu.module @kernel_module [#nvvm.target] {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }

  func.func @foo(%buffer: memref<?xf32>) {
  // CHECK: [[C8:%.*]] = llvm.mlir.constant(8 : index) : i64
  // CHECK: [[C32:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: [[C256:%.*]] = llvm.mlir.constant(256 : i32) : i32
  // CHECK: [[C2:%.*]] = llvm.mlir.constant(2 : index) : i64
    %c8 = arith.constant 8 : index    
    %c32 = arith.constant 32 : i32
    %c256 = arith.constant 256 : i32
    %c2 = arith.constant 2 : index

  // CHECK: gpu.launch_func @kernel_module::@kernel
  // CHECK: clusters in ([[C2]], [[C2]], [[C2]])
  // CHECK: blocks in ([[C8]], [[C8]], [[C8]]) threads in ([[C8]], [[C8]], [[C8]]) : i64
  // CHECK: dynamic_shared_memory_size [[C256]]
  // CHECK: args([[C32]] : i32, %{{.*}} : !llvm.ptr, %{{.*}} : !llvm.ptr, %{{.*}} : i64, %{{.*}} : i64, %{{.*}} : i64)
    gpu.launch_func @kernel_module::@kernel
        clusters in (%c2, %c2, %c2)
        blocks in (%c8, %c8, %c8)
        threads in (%c8, %c8, %c8)
        dynamic_shared_memory_size %c256
        args(%c32 : i32, %buffer : memref<?xf32>)
    return
  }
}

// -----

module attributes {gpu.container_module} {
  // CHECK: gpu.binary
  gpu.binary @kernel_module [#gpu.object<#rocdl.target, "blob">]

  func.func @foo(%buffer: memref<?xf32>) {
  // CHECK: [[C8:%.*]] = llvm.mlir.constant(8 : index) : i64
  // CHECK: [[C32:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: [[C256:%.*]] = llvm.mlir.constant(256 : i32) : i32
  // CHECK: [[C2:%.*]] = llvm.mlir.constant(2 : index) : i64
    %c8 = arith.constant 8 : index    
    %c32 = arith.constant 32 : i32
    %c256 = arith.constant 256 : i32
    %c2 = arith.constant 2 : index

  // CHECK: gpu.launch_func @kernel_module::@kernel
  // CHECK: clusters in ([[C2]], [[C2]], [[C2]])
  // CHECK: blocks in ([[C8]], [[C8]], [[C8]]) threads in ([[C8]], [[C8]], [[C8]]) : i64
  // CHECK: dynamic_shared_memory_size [[C256]]
  // CHECK: args([[C32]] : i32, %{{.*}} : !llvm.ptr, %{{.*}} : !llvm.ptr, %{{.*}} : i64, %{{.*}} : i64, %{{.*}} : i64)
    gpu.launch_func @kernel_module::@kernel
        clusters in (%c2, %c2, %c2)
        blocks in (%c8, %c8, %c8)
        threads in (%c8, %c8, %c8)
        dynamic_shared_memory_size %c256
        args(%c32 : i32, %buffer : memref<?xf32>)
    return
  }
}
