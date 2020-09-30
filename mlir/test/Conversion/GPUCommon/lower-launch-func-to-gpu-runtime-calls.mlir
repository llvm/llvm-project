// RUN: mlir-opt -allow-unregistered-dialect %s --gpu-to-llvm="gpu-binary-annotation=nvvm.cubin" | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s --gpu-to-llvm="gpu-binary-annotation=rocdl.hsaco" | FileCheck %s --check-prefix=ROCDL

module attributes {gpu.container_module} {

  // CHECK: llvm.mlir.global internal constant @[[KERNEL_NAME:.*]]("kernel\00")
  // CHECK: llvm.mlir.global internal constant @[[GLOBAL:.*]]("CUBIN")
  // ROCDL: llvm.mlir.global internal constant @[[GLOBAL:.*]]("HSACO")

  gpu.module @kernel_module attributes {
      nvvm.cubin = "CUBIN", rocdl.hsaco = "HSACO"
  } {
    llvm.func @kernel(%arg0: !llvm.i32, %arg1: !llvm.ptr<float>,
        %arg2: !llvm.ptr<float>, %arg3: !llvm.i64, %arg4: !llvm.i64,
        %arg5: !llvm.i64) attributes {gpu.kernel} {
      llvm.return
    }
  }

  func @foo(%buffer: memref<?xf32>) {
    %c8 = constant 8 : index
    %c32 = constant 32 : i32
    "gpu.launch_func"(%c8, %c8, %c8, %c8, %c8, %c8, %c32, %buffer) {
      kernel = @kernel_module::@kernel
    } : (index, index, index, index, index, index, i32, memref<?xf32>) -> ()
    return
  }

  // CHECK: [[C8:%.*]] = llvm.mlir.constant(8 : index) : !llvm.i64   
  // CHECK: [[ADDRESSOF:%.*]] = llvm.mlir.addressof @[[GLOBAL]]
  // CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : index)
  // CHECK: [[BINARY:%.*]] = llvm.getelementptr [[ADDRESSOF]]{{\[}}[[C0]], [[C0]]]
  // CHECK-SAME: -> !llvm.ptr<i8>

  // CHECK: [[MODULE:%.*]] = llvm.call @mgpuModuleLoad([[BINARY]])
  // CHECK: [[FUNC:%.*]] = llvm.call @mgpuModuleGetFunction([[MODULE]], {{.*}})

  // CHECK: [[C0_I32:%.*]] = llvm.mlir.constant(0 : i32)
  // CHECK: [[STREAM:%.*]] = llvm.call @mgpuStreamCreate

  // CHECK: [[NUM_PARAMS:%.*]] = llvm.mlir.constant(6 : i32) : !llvm.i32
  // CHECK-NEXT: [[PARAMS:%.*]] = llvm.alloca [[NUM_PARAMS]] x !llvm.ptr<i8>

  // CHECK: [[EXTRA_PARAMS:%.*]] = llvm.mlir.null : !llvm.ptr<ptr<i8>>

  // CHECK: llvm.call @mgpuLaunchKernel([[FUNC]], [[C8]], [[C8]], [[C8]],
  // CHECK-SAME: [[C8]], [[C8]], [[C8]], [[C0_I32]], [[STREAM]],
  // CHECK-SAME: [[PARAMS]], [[EXTRA_PARAMS]])
  // CHECK: llvm.call @mgpuStreamSynchronize
}
