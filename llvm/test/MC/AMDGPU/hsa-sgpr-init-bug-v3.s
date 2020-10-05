// RUN: llvm-mc -mattr=+code-object-v3 -triple amdgcn-amd-amdhsa -mcpu=gfx802 -filetype=obj < %s > %t
// RUN: llvm-objdump -s -j .rodata %t | FileCheck --check-prefix=OBJDUMP %s

// Check that SGPR init bug on gfx803 is corrected by the assembler, setting
// GRANULATED_WAVEFRONT_SGPR_COUNT to 11.

// OBJDUMP: Contents of section .rodata
// OBJDUMP-NEXT: 0000 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0010 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0020 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0030 c002ac00 80000000 00000000 00000000

.text

.amdgcn_target "amdgcn-amd-amdhsa--gfx802"

.p2align 8
minimal:
  s_endpgm

.rodata

.amdhsa_kernel minimal
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel
