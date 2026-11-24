// REQUIRES: has_llc_binary
// RUN: mlir-translate -mlir-to-llvmir %s | llc -filetype=obj -o %t.o
// RUN: llvm-objdump --offloading %t.o | FileCheck %s

// Check that the images embedded by the offload packager are read back as
// offloading binaries from the object file of the host module.

// CHECK: OFFLOADING IMAGE [0]:
// CHECK-NEXT: kind            llvm ir
// CHECK-NEXT: arch            sm_80
// CHECK-NEXT: triple          nvptx64-nvidia-cuda
// CHECK-NEXT: producer        openacc
// CHECK: OFFLOADING IMAGE [1]:
// CHECK-NEXT: kind            elf
// CHECK-NEXT: arch            gfx90a
// CHECK-NEXT: triple          amdgcn-amd-amdhsa
// CHECK-NEXT: producer        openacc

module attributes {gpu.container_module} {
  gpu.binary @device_module <#acc.offload_packager> [
    #gpu.object<#nvvm.target<chip = "sm_80">, offload = "BC">,
    #gpu.object<#rocdl.target<chip = "gfx90a">, bin =  "\177ELF">]
}
