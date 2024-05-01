// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx90a -filetype=obj < %s > %t
// RUN: llvm-objdump -s -j .rodata %t | FileCheck --check-prefix=OBJDUMP %s

// OBJDUMP:       0000 00000000 0f000000 00000000 00000000

.text

.p2align 8
.type caller,@function
caller:
  s_endpgm

.rodata

.p2align 6
.amdhsa_kernel caller
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
  .amdhsa_private_segment_fixed_size max(7, callee1.private_seg_size, callee2.private_seg_size)
.end_amdhsa_kernel

.set callee1.private_seg_size, 4
.set callee2.private_seg_size, 15

// ASM: .amdhsa_private_segment_fixed_size max(7, callee1.private_seg_size, callee2.private_seg_size)
