// RUN: mlir-opt -allow-unregistered-dialect %s --launch-func-to-gpu-runtime="gpu-binary-annotation=nvvm.cubin" | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s --launch-func-to-gpu-runtime="gpu-binary-annotation=rocdl.hsaco" | FileCheck %s --check-prefix=ROCDL

module attributes {gpu.container_module} {

  // CHECK: llvm.mlir.global internal constant @[[kernel_name:.*]]("kernel\00")
  // CHECK: llvm.mlir.global internal constant @[[global:.*]]("CUBIN")
  // ROCDL: llvm.mlir.global internal constant @[[global:.*]]("HSACO")

  gpu.module @kernel_module attributes {nvvm.cubin = "CUBIN", rocdl.hsaco = "HSACO"} {
    llvm.func @kernel(%arg0: !llvm.float, %arg1: !llvm<"float*">) attributes {gpu.kernel} {
      llvm.return
    }
  }

  llvm.func @foo() {
    %0 = "op"() : () -> (!llvm.float)
    %1 = "op"() : () -> (!llvm<"float*">)
    %cst = llvm.mlir.constant(8 : index) : !llvm.i64

    // CHECK: %[[addressof:.*]] = llvm.mlir.addressof @[[global]]
    // CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : index)
    // CHECK: %[[binary:.*]] = llvm.getelementptr %[[addressof]][%[[c0]], %[[c0]]]
    // CHECK-SAME: -> !llvm<"i8*">
    // CHECK: %[[module:.*]] = llvm.call @mgpuModuleLoad(%[[binary]]) : (!llvm<"i8*">) -> !llvm<"i8*">
    // CHECK: %[[func:.*]] = llvm.call @mgpuModuleGetFunction(%[[module]], {{.*}}) : (!llvm<"i8*">, !llvm<"i8*">) -> !llvm<"i8*">
    // CHECK: llvm.call @mgpuStreamCreate
    // CHECK: llvm.call @mgpuLaunchKernel
    // CHECK: llvm.call @mgpuStreamSynchronize
    "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1) { kernel = @kernel_module::@kernel }
        : (!llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.float, !llvm<"float*">) -> ()

    llvm.return
  }

}
