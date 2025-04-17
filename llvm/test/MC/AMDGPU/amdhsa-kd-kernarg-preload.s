// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj < %s -o - | llvm-objdump -s -j .rodata - | FileCheck --check-prefix=OBJDUMP %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx942"

.rodata

// Account for preload kernarg SGPRs in KD field GRANULATED_WAVEFRONT_SGPR_COUNT.

// OBJDUMP:      Contents of section .rodata:
// OBJDUMP-NEXT: 0000 00000000 00000000 00000000 00000000  ................
// OBJDUMP-NEXT: 0010 00000000 00000000 00000000 00000000  ................
// OBJDUMP-NEXT: 0020 00000000 00000000 00000000 00000000  ................
// OBJDUMP-NOT:  0030 0000ac00 92000000 00000900 00000000  ................
// OBJDUMP-NEXT: 0030 4000ac00 92000000 00000900 00000000  @...............

.amdhsa_kernel amdhsa_kd_kernarg
  .amdhsa_user_sgpr_kernarg_preload_length 9
  .amdhsa_next_free_sgpr 0
  .amdhsa_next_free_vgpr 0
  .amdhsa_accum_offset 4
.end_amdhsa_kernel
