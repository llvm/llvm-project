// Hardwired-on XNACK is implied by the processor, not an ELF mode selection.
// RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s

// CHECK: Flags [ (0x449)

s_endpgm
