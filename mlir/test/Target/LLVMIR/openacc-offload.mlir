// RUN: mlir-translate -mlir-to-llvmir -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: @device_module_binary = internal unnamed_addr constant [{{[0-9]+}} x i8] c"\10\FF\10\AD
// CHECK-SAME: section ".llvm.offloading"
// CHECK: @llvm.used = appending global {{.*}} @device_module_binary

module attributes {gpu.container_module} {
  gpu.binary @device_module <#acc.offload_packager> [
    #gpu.object<#nvvm.target<chip = "sm_80">, offload = "BC">]
}

// -----

// CHECK: @nvidia_object_module_binary = internal unnamed_addr constant
// CHECK: @amd_module_binary = internal unnamed_addr constant {{.*}}amdgcn-amd-amdhsa{{.*}}gfx90a
// CHECK: @intel_module_binary = internal unnamed_addr constant {{.*}}spirv64-unknown-unknown{{.*}}bmg

module attributes {gpu.container_module} {
  gpu.binary @nvidia_object_module <#acc.offload_packager> [
    #gpu.object<#nvvm.target<chip = "sm_80">, bin = "\177ELF">]
  gpu.binary @amd_module <#acc.offload_packager> [
    #gpu.object<#rocdl.target<chip = "gfx90a">, offload = "BC">]
  gpu.binary @intel_module <#acc.offload_packager> [
    #gpu.object<#xevm.target, offload = "BC">]
}

// -----

module attributes {gpu.container_module} {
  // expected-error @below {{cannot link the assembly or fatbin formats}}
  // expected-error @below {{LLVM Translation failed for operation: gpu.binary}}
  gpu.binary @ptx <#acc.offload_packager> [
    #gpu.object<#nvvm.target<chip = "sm_80">, assembly = ".version 8.0">]
}

// -----

module attributes {gpu.container_module} {
  // expected-error @below {{cannot link the assembly or fatbin formats}}
  // expected-error @below {{LLVM Translation failed for operation: gpu.binary}}
  gpu.binary @fatbin <#acc.offload_packager> [
    #gpu.object<#nvvm.target<chip = "sm_80">, fatbin = "">]
}
