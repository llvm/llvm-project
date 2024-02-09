// RUN: mlir-opt %s --gpu-module-to-binary --verify-diagnostics

module attributes {gpu.container_module} {
  // expected-error @below {{the module has no target attributes}}
  gpu.module @kernel_module1 {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }
}
