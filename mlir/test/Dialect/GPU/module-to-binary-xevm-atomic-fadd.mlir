// REQUIRES: host-supports-spirv
// RUN: mlir-opt %s --gpu-module-to-binary="format=isa" | FileCheck %s

// A floating-point atomic add lowers to OpAtomicFAddEXT, which needs
// SPV_EXT_shader_atomic_float_add: it is among XeVM's default SPIR-V
// extensions, so the module serializes and declares the extension.

module attributes {gpu.container_module} {
  // CHECK: gpu.binary @kernels
  // CHECK-SAME: SPV_EXT_shader_atomic_float_add
  gpu.module @kernels [#xevm.target] {
    llvm.func spir_kernelcc @atomic_fadd(%ptr: !llvm.ptr<1>, %val: f32) attributes {gpu.kernel} {
      %0 = llvm.atomicrmw fadd %ptr, %val seq_cst : !llvm.ptr<1>, f32
      llvm.return
    }
  }
}
