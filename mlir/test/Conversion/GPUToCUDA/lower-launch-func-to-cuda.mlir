// RUN: mlir-opt -allow-unregistered-dialect %s --launch-func-to-cuda | FileCheck %s

module attributes {gpu.container_module} {

  // CHECK: llvm.mlir.global internal constant @[[kernel_name:.*]]("kernel\00")
  // CHECK: llvm.mlir.global internal constant @[[global:.*]]("CUBIN")

  gpu.module @kernel_module attributes {nvvm.cubin = "CUBIN"} {
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
    // CHECK: %[[cubin_ptr:.*]] = llvm.getelementptr %[[addressof]][%[[c0]], %[[c0]]]
    // CHECK-SAME: -> !llvm<"i8*">
    // CHECK: %[[module_ptr:.*]] = llvm.alloca {{.*}} x !llvm<"i8*"> : (!llvm.i32) -> !llvm<"i8**">
    // CHECK: llvm.call @mcuModuleLoad(%[[module_ptr]], %[[cubin_ptr]]) : (!llvm<"i8**">, !llvm<"i8*">) -> !llvm.i32
    // CHECK: %[[func_ptr:.*]] = llvm.alloca {{.*}} x !llvm<"i8*"> : (!llvm.i32) -> !llvm<"i8**">
    // CHECK: llvm.call @mcuModuleGetFunction(%[[func_ptr]], {{.*}}, {{.*}}) : (!llvm<"i8**">, !llvm<"i8*">, !llvm<"i8*">) -> !llvm.i32
    // CHECK: llvm.call @mcuGetStreamHelper
    // CHECK: llvm.call @mcuLaunchKernel
    // CHECK: llvm.call @mcuStreamSynchronize
    "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1) { kernel = @kernel_module::@kernel }
        : (!llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.float, !llvm<"float*">) -> ()

    llvm.return
  }

}
