// RUN: mlir-translate -mlir-to-llvmir -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: @device_module_binary = internal unnamed_addr constant [{{[0-9]+}} x i8] c"{{.*}} section ".llvm.offloading"
// CHECK: @llvm.used = appending global {{.*}} @device_module_binary
// CHECK: @.offloading.entry_name = internal unnamed_addr constant [9 x i8] c"kernel_0\00", section ".llvm.rodata.offloading"
// CHECK: @.offloading.entry.kernel_0 = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 16, i32 0, ptr @kernel_0, ptr @.offloading.entry_name, i64 0, i64 0, ptr null }, section "llvm_offload_entries"

module attributes {gpu.container_module} {
  llvm.func @kernel_0() attributes {acc.offload_entry} {
    llvm.return
  }
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

// -----

// CHECK: @.offloading.entry.var = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 16, i32 0, ptr @var, ptr @.offloading.entry_name, i64 4, i64 0, ptr null }, section "llvm_offload_entries"

module {
  llvm.mlir.global external @var(0 : i32) {acc.offload_entry} : i32
}

// -----

module {
  llvm.func @f() {
    // expected-error @below {{unsupported operation for 'acc.offload_entry'}}
    %c = llvm.mlir.constant(1 : i32) {acc.offload_entry} : i32
    llvm.return
  }
}
