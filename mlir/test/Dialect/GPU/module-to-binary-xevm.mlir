// REQUIRES: asserts, host-supports-spirv
// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(convert-xevm-to-llvm),gpu-module-to-binary{format=llvm})" \
// RUN:   -debug-only=serialize-to-llvm -o /dev/null 2>&1 | FileCheck %s
// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(convert-xevm-to-llvm),gpu-module-to-binary{format=isa})" \
// RUN:   | FileCheck %s --check-prefix=ISA

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
    // Block reads and writes need the SPV_INTEL_subgroups extension. The
    // assembly attribute holds the SPIR-V binary, where OpExtension stores the
    // extension name as a string.
    // CHECK: define spir_kernel void @block_load_store(ptr addrspace(1) %0, ptr addrspace(1) %1) !intel_reqd_sub_group_size ![[SG]] {
    // ISA-LABEL: gpu.binary @kernels
    // ISA-SAME: assembly = "{{[^"]*}}SPV_INTEL_subgroups\00
    llvm.func spir_kernelcc @block_load_store(%src: !llvm.ptr<1>, %dst: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = xevm.blockload %src : (!llvm.ptr<1>) -> vector<2xi32>
      xevm.blockstore %dst, %v : (!llvm.ptr<1>, vector<2xi32>)
      llvm.return
    }
  }
  // CHECK: ![[SG]] = !{i32 16}
}
