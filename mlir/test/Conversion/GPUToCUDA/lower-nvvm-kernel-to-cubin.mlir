// RUN: mlir-opt %s --test-kernel-to-cubin -split-input-file | FileCheck %s

// CHECK: attributes {nvvm.cubin = "CUBIN"}
gpu.module @foo {
  llvm.func @kernel(%arg0 : f32, %arg1 : !llvm.ptr<f32>)
    // CHECK: attributes  {gpu.kernel}
    attributes  { gpu.kernel } {
    llvm.return
  }
}

// -----

gpu.module @bar {
  // CHECK: func @kernel_a
  llvm.func @kernel_a()
    attributes  { gpu.kernel } {
    llvm.return
  }

  // CHECK: func @kernel_b
  llvm.func @kernel_b()
    attributes  { gpu.kernel } {
    llvm.return
  }
}
