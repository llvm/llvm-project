// REQUIRES: asserts, host-supports-spirv
// RUN: mlir-opt %s --gpu-module-to-binary="format=llvm" \
// RUN:   -debug-only=serialize-to-llvm -o /dev/null 2>&1 | FileCheck %s

// Every kernel of the module gets the required sub-group size, not only the
// first one. Other functions get none.

module attributes {gpu.container_module} {
  // CHECK-LABEL: LLVM IR for module: "kernels"
  gpu.module @kernels [#xevm.target] {
    // CHECK: define spir_kernel void @kernel1() !intel_reqd_sub_group_size ![[SG:[0-9]+]] {
    llvm.func spir_kernelcc @kernel1() attributes {gpu.kernel} {
      llvm.return
    }
    // CHECK: define spir_kernel void @kernel2() !intel_reqd_sub_group_size ![[SG]] {
    llvm.func spir_kernelcc @kernel2() attributes {gpu.kernel} {
      llvm.return
    }
    // CHECK: define spir_func void @helper() {
    llvm.func spir_funccc @helper() {
      llvm.return
    }
  }
  // CHECK: ![[SG]] = !{i32 16}
}
