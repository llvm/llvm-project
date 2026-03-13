// RUN: mlir-opt %s --gpu-module-to-binary="format=%gpu_compilation_format opts=--verbose" \
// RUN:   | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL: gpu.binary @kernel_module
  // CHECK: properties = {{{.*}}ISACompilerLog = {{.*}}
  gpu.module @kernel_module [#nvvm.target<chip = "sm_70", flags = {"collect-compiler-diagnostics"}>] {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }
}
