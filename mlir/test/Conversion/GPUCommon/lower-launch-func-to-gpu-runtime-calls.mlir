// RUN: mlir-opt %s --gpu-to-llvm="gpu-binary-annotation=nvvm.cubin" | FileCheck %s
// RUN: mlir-opt %s --gpu-to-llvm="gpu-binary-annotation=rocdl.hsaco" | FileCheck %s --check-prefix=ROCDL

module attributes {gpu.container_module} {

  // CHECK: llvm.mlir.global internal constant @[[KERNEL_NAME:.*]]("kernel\00")
  // CHECK: llvm.mlir.global internal constant @[[GLOBAL:.*]]("CUBIN")
  // ROCDL: llvm.mlir.global internal constant @[[GLOBAL:.*]]("HSACO")

  gpu.module @kernel_module attributes {
      nvvm.cubin = "CUBIN", rocdl.hsaco = "HSACO"
  } {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr<f32>,
        %arg2: !llvm.ptr<f32>, %arg3: i64, %arg4: i64,
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
  // CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : index)
  // CHECK: [[BINARY:%.*]] = llvm.getelementptr [[ADDRESSOF]]{{\[}}[[C0]], [[C0]]]
  // CHECK-SAME: -> !llvm.ptr<i8>

  // CHECK: [[MODULE:%.*]] = llvm.call @mgpuModuleLoad([[BINARY]])
  // CHECK: [[FUNC:%.*]] = llvm.call @mgpuModuleGetFunction([[MODULE]], {{.*}})

  // CHECK: [[STREAM:%.*]] = llvm.call @mgpuStreamCreate

  // CHECK: [[NUM_PARAMS:%.*]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK-NEXT: [[PARAMS:%.*]] = llvm.alloca [[NUM_PARAMS]] x !llvm.ptr<i8>

  // CHECK: [[EXTRA_PARAMS:%.*]] = llvm.mlir.null : !llvm.ptr<ptr<i8>>

  // CHECK: llvm.call @mgpuLaunchKernel([[FUNC]], [[C8]], [[C8]], [[C8]],
  // CHECK-SAME: [[C8]], [[C8]], [[C8]], [[C256]], [[STREAM]],
  // CHECK-SAME: [[PARAMS]], [[EXTRA_PARAMS]])
  // CHECK: llvm.call @mgpuStreamSynchronize
  // CHECK: llvm.call @mgpuStreamDestroy
  // CHECK: llvm.call @mgpuModuleUnload
}
