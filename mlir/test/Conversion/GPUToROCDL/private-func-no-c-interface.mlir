// RUN: mlir-opt %s -convert-gpu-to-rocdl -split-input-file | FileCheck %s

// A private device function cannot be called from outside its module, so it
// must not be given a C interface wrapper; requesting one would only anchor
// the function against dead-code elimination.
gpu.module @kernel_private {
  // CHECK-LABEL: gpu.module @kernel_private
  // CHECK-NOT: emit_c_interface
  // CHECK-NOT: _mlir_ciface_
  func.func private @helper(%arg0: i64) -> i64 {
    return %arg0 : i64
  }
}

// -----

// A publicly visible device function keeps its C interface wrapper.
gpu.module @kernel_public {
  // CHECK-LABEL: gpu.module @kernel_public
  // CHECK: llvm.emit_c_interface
  // CHECK: @_mlir_ciface_entry
  func.func @entry(%arg0: i64) -> i64 {
    return %arg0 : i64
  }
}
