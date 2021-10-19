// RUN: mlir-opt %s --test-gpu-to-hsaco | FileCheck %s

// CHECK: gpu.module @foo attributes {gpu.binary = "HSACO"}
gpu.module @foo {
  llvm.func @kernel(%arg0 : f32, %arg1 : !llvm.ptr<f32>)
    // CHECK: attributes  {gpu.kernel}
    attributes  { gpu.kernel } {
    llvm.return
  }
}

// CHECK: gpu.module @bar attributes {gpu.binary = "HSACO"}
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
