// RUN: mlir-opt %s --gpu-to-llvm="gpu-binary-annotation=nvvm.cubin" -split-input-file | FileCheck %s
// RUN: mlir-opt %s --gpu-to-llvm="gpu-binary-annotation=rocdl.hsaco" -split-input-file | FileCheck %s --check-prefix=ROCDL

module attributes {gpu.container_module} {

  // CHECK: llvm.mlir.global internal constant @[[KERNEL_NAME:.*]]("kernel\00")
  // CHECK: llvm.mlir.global internal constant @[[GLOBAL:.*]]("CUBIN")
  // ROCDL: llvm.mlir.global internal constant @[[GLOBAL:.*]]("HSACO")

  gpu.module @kernel_module attributes {
      nvvm.cubin = "CUBIN", rocdl.hsaco = "HSACO"
  } {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }

  func.func @foo(%buffer: memref<?xf32>) {
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : i32
    %c256 = arith.constant 256 : i32
    gpu.launch_func @kernel_module::@kernel
        blocks in (%c8, %c8, %c8)
        threads in (%c8, %c8, %c8)
        dynamic_shared_memory_size %c256
        args(%c32 : i32, %buffer : memref<?xf32>)
    return
  }

  // CHECK-DAG: [[C256:%.*]] = llvm.mlir.constant(256 : i32) : i32
  // CHECK-DAG: [[C8:%.*]] = llvm.mlir.constant(8 : index) : i64
  // CHECK: [[ADDRESSOF:%.*]] = llvm.mlir.addressof @[[GLOBAL]]
  // CHECK: [[BINARY:%.*]] = llvm.getelementptr [[ADDRESSOF]]{{\[}}0, 0]
  // CHECK-SAME: -> !llvm.ptr
  // CHECK: [[BINARYSIZE:%.*]] = llvm.mlir.constant
  // CHECK: [[MODULE:%.*]] = llvm.call @mgpuModuleLoad([[BINARY]], [[BINARYSIZE]])
  // CHECK: [[PARAMSCOUNT:%.*]] = llvm.mlir.constant
  // CHECK: [[FUNC:%.*]] = llvm.call @mgpuModuleGetFunction([[MODULE]], {{.*}})

  // CHECK: [[STREAM:%.*]] = llvm.call @mgpuStreamCreate

  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
  // CHECK: %[[MEMREF:.*]] = llvm.alloca %[[ONE]] x !llvm.struct[[STRUCT_BODY:<.*>]]
  // CHECK: [[NUM_PARAMS:%.*]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK-NEXT: [[PARAMS:%.*]] = llvm.alloca [[NUM_PARAMS]] x !llvm.ptr

  // CHECK: llvm.getelementptr %[[MEMREF]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct[[STRUCT_BODY:<.*>]]
  // CHECK: llvm.getelementptr %[[MEMREF]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct[[STRUCT_BODY:<.*>]]
  // CHECK: llvm.getelementptr %[[MEMREF]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct[[STRUCT_BODY:<.*>]]
  // CHECK: llvm.getelementptr %[[MEMREF]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct[[STRUCT_BODY:<.*>]]
  // CHECK: llvm.getelementptr %[[MEMREF]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct[[STRUCT_BODY:<.*>]]
  // CHECK: llvm.getelementptr %[[MEMREF]][0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct[[STRUCT_BODY:<.*>]]

  // CHECK: [[EXTRA_PARAMS:%.*]] = llvm.mlir.zero : !llvm.ptr

  // CHECK: llvm.call @mgpuLaunchKernel([[FUNC]], [[C8]], [[C8]], [[C8]],
  // CHECK-SAME: [[C8]], [[C8]], [[C8]], [[C256]], [[STREAM]],
  // CHECK-SAME: [[PARAMS]], [[EXTRA_PARAMS]], [[PARAMSCOUNT]])
  // CHECK: llvm.call @mgpuStreamSynchronize
  // CHECK: llvm.call @mgpuStreamDestroy
  // CHECK: llvm.call @mgpuModuleUnload
}

// -----

module attributes {gpu.container_module} {
  // CHECK: gpu.module
  // ROCDL: gpu.module
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
