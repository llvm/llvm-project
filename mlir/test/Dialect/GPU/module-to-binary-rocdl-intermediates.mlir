// REQUIRES: host-supports-amdgpu
// RUN: rm -rf %t
// RUN: mlir-opt %s --gpu-module-to-binary='format=isa dump-intermediates=%t' | FileCheck %s
// RUN: test -f %t/kernel_module.initial.ll
// RUN: test -f %t/kernel_module.linked.ll
// RUN: test -f %t/kernel_module.opt.ll
// RUN: test -f %t/kernel.isa

module attributes {gpu.container_module} {
  // CHECK-LABEL: gpu.binary @kernel_module

  gpu.module @kernel_module [#rocdl.target<chip = "gfx942">] {
    llvm.func @kernel(%arg0: f32) {
      llvm.return
    }
  }
}
