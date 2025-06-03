// RUN: mlir-opt %s --nvvm-attach-target="" | FileCheck %s
// RUN: mlir-opt %s --nvvm-attach-target="ptxas-cmd-options=--register-usage-level=8" | FileCheck %s -check-prefix=CHECK-OPTIONS

module attributes {gpu.container_module} {
    // CHECK-LABEL:gpu.module @kernel_module1
    // CHECK: [#nvvm.target]
    // CHECK-OPTIONS: [#nvvm.target<flags = {"ptxas-cmd-options" = ["--register-usage-level=8"]}>]
    gpu.module @kernel_module1 {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
            llvm.return
        }
    }
}
