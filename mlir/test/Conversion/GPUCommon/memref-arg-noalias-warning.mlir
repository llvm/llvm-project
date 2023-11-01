// RUN: mlir-opt %s -split-input-file -convert-gpu-to-rocdl='use-bare-ptr-memref-call-conv=0' -verify-diagnostics

gpu.module @kernel {
// expected-warning @+1 {{Cannot copy noalias with non-bare pointers.}}
  gpu.func @func_warning_for_not_bare_pointer(%arg0 : memref<f32> {llvm.noalias} ) {
    gpu.return
  }
}
