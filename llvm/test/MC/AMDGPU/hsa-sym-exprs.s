// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx90a -filetype=obj < %s > %t
// RUN: llvm-objdump -s -j .rodata %t | FileCheck --check-prefix=OBJDUMP %s

// When going from asm -> asm, the expressions should remain the same (i.e., symbolic).
// When going from asm -> obj, the expressions should get resolved (through fixups),

// OBJDUMP: Contents of section .rodata
// expr_defined_later
// OBJDUMP-NEXT: 0000 2b000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0010 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0020 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0030 0000ac00 80000000 00000000 00000000
// expr_defined
// OBJDUMP-NEXT: 0040 2d000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0050 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0060 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0070 0000ac00 80000000 00000000 00000000

.text
// ASM: .text

.amdhsa_code_object_version 4
// ASM: .amdhsa_code_object_version 4

.p2align 8
.type expr_defined_later,@function
expr_defined_later:
  s_endpgm

.p2align 8
.type expr_defined,@function
expr_defined:
  s_endpgm

.rodata
// ASM: .rodata

.p2align 6
.amdhsa_kernel expr_defined_later
  .amdhsa_group_segment_fixed_size defined_value+2
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
.end_amdhsa_kernel

.set defined_value, 41

.p2align 6
.amdhsa_kernel expr_defined
  .amdhsa_group_segment_fixed_size defined_value+4
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
.end_amdhsa_kernel



// ASM: .amdhsa_kernel expr_defined_later
// ASM: .amdhsa_group_segment_fixed_size defined_value+2
// ASM: .end_amdhsa_kernel

// ASM:       .set defined_value, 41
// ASM-NEXT:  .no_dead_strip defined_value

// ASM: .amdhsa_kernel expr_defined
// ASM: .amdhsa_group_segment_fixed_size 45
// ASM: .end_amdhsa_kernel
