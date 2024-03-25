// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx904 -mattr=+xnack < %s | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx904 -mattr=+xnack -filetype=obj < %s > %t
// RUN: llvm-readelf -S -r -s %t | FileCheck --check-prefix=READOBJ %s
// RUN: llvm-objdump -s -j .rodata %t | FileCheck --check-prefix=OBJDUMP %s

// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx904 --amdhsa-code-object-version=6 -mattr=+xnack < %s | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx904 --amdhsa-code-object-version=6 -mattr=+xnack -filetype=obj < %s > %t
// RUN: llvm-readelf -S -r -s %t | FileCheck --check-prefix=READOBJ %s
// RUN: llvm-objdump -s -j .rodata %t | FileCheck --check-prefix=OBJDUMP %s

// READOBJ: Section Headers
// READOBJ: .text   PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9]+}} AX {{[0-9]+}} {{[0-9]+}} 256
// READOBJ: .rodata PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}}        000100 {{[0-9]+}}  A {{[0-9]+}} {{[0-9]+}} 64

// READOBJ: Relocation section '.rela.rodata' at offset
// READOBJ: 0000000000000010 {{[0-9a-f]+}}00000005 R_AMDGPU_REL64 0000000000000000 .text + 10
// READOBJ: 0000000000000050 {{[0-9a-f]+}}00000005 R_AMDGPU_REL64 0000000000000000 .text + 110
// READOBJ: 0000000000000090 {{[0-9a-f]+}}00000005 R_AMDGPU_REL64 0000000000000000 .text + 210
// READOBJ: 00000000000000d0 {{[0-9a-f]+}}00000005 R_AMDGPU_REL64 0000000000000000 .text + 310

// READOBJ: Symbol table '.symtab' contains {{[0-9]+}} entries:
// READOBJ:      0000000000000000  0 FUNC    LOCAL  PROTECTED 2 minimal
// READOBJ-NEXT: 0000000000000100  0 FUNC    LOCAL  PROTECTED 2 complete
// READOBJ-NEXT: 0000000000000200  0 FUNC    LOCAL  PROTECTED 2 special_sgpr
// READOBJ-NEXT: 0000000000000300  0 FUNC    LOCAL  PROTECTED 2 disabled_user_sgpr
// READOBJ-NEXT: 0000000000000000 64 OBJECT  LOCAL  DEFAULT   3 minimal.kd
// READOBJ-NEXT: 0000000000000040 64 OBJECT  LOCAL  DEFAULT   3 complete.kd
// READOBJ-NEXT: 0000000000000080 64 OBJECT  LOCAL  DEFAULT   3 special_sgpr.kd
// READOBJ-NEXT: 00000000000000c0 64 OBJECT  LOCAL  DEFAULT   3 disabled_user_sgpr.kd

// OBJDUMP: Contents of section .rodata
// Note, relocation for KERNEL_CODE_ENTRY_BYTE_OFFSET is not resolved here.
// minimal
// OBJDUMP-NEXT: 0000 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0010 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0020 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0030 0000ac00 80000000 00000000 00000000
// complete
// OBJDUMP-NEXT: 0040 01000000 01000000 08000000 00000000
// OBJDUMP-NEXT: 0050 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0060 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0070 c2500104 1f0f007f 7f080000 00000000
// special_sgpr
// OBJDUMP-NEXT: 0080 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0090 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 00a0 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 00b0 00010000 80000000 00000000 00000000
// disabled_user_sgpr
// OBJDUMP-NEXT: 00c0 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 00d0 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 00e0 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 00f0 0000ac00 80000000 00000000 00000000

.text
// ASM: .text

.amdgcn_target "amdgcn-amd-amdhsa--gfx904:xnack+"
// ASM: .amdgcn_target "amdgcn-amd-amdhsa--gfx904:xnack+"

.amdhsa_code_object_version 5
// ASM: .amdhsa_code_object_version 5

.p2align 8
.type minimal,@function
minimal:
  s_endpgm

.p2align 8
.type complete,@function
complete:
  s_endpgm

.p2align 8
.type special_sgpr,@function
special_sgpr:
  s_endpgm

.p2align 8
.type disabled_user_sgpr,@function
disabled_user_sgpr:
  s_endpgm

.rodata
// ASM: .rodata

// Test that only specifying required directives is allowed, and that defaulted
// values are omitted.
.p2align 6
.amdhsa_kernel minimal
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

// ASM: .amdhsa_kernel minimal
// ASM: .amdhsa_next_free_vgpr 0
// ASM-NEXT: .amdhsa_next_free_sgpr 0
// ASM: .end_amdhsa_kernel

// Test that we can specify all available directives with non-default values.
.p2align 6
.amdhsa_kernel complete
  .amdhsa_group_segment_fixed_size 1
  .amdhsa_private_segment_fixed_size 1
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_count 15
  .amdhsa_user_sgpr_private_segment_buffer 1
  .amdhsa_user_sgpr_dispatch_ptr 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_dispatch_id 1
  .amdhsa_user_sgpr_flat_scratch_init 1
  .amdhsa_user_sgpr_private_segment_size 1
  .amdhsa_uses_dynamic_stack 1
  .amdhsa_system_sgpr_private_segment_wavefront_offset 1
  .amdhsa_system_sgpr_workgroup_id_x 0
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_sgpr_workgroup_info 1
  .amdhsa_system_vgpr_workitem_id 1
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 27
  .amdhsa_reserve_vcc 0
  .amdhsa_reserve_flat_scratch 0
  .amdhsa_reserve_xnack_mask 1
  .amdhsa_float_round_mode_32 1
  .amdhsa_float_round_mode_16_64 1
  .amdhsa_float_denorm_mode_32 1
  .amdhsa_float_denorm_mode_16_64 0
  .amdhsa_dx10_clamp 0
  .amdhsa_ieee_mode 0
  .amdhsa_fp16_overflow 1
  .amdhsa_exception_fp_ieee_invalid_op 1
  .amdhsa_exception_fp_denorm_src 1
  .amdhsa_exception_fp_ieee_div_zero 1
  .amdhsa_exception_fp_ieee_overflow 1
  .amdhsa_exception_fp_ieee_underflow 1
  .amdhsa_exception_fp_ieee_inexact 1
  .amdhsa_exception_int_div_zero 1
.end_amdhsa_kernel

// ASM: .amdhsa_kernel complete
// ASM-NEXT: .amdhsa_group_segment_fixed_size 1
// ASM-NEXT: .amdhsa_private_segment_fixed_size 1
// ASM-NEXT: .amdhsa_kernarg_size 8
// ASM-NEXT: .amdhsa_user_sgpr_count 15
// ASM-NEXT: .amdhsa_user_sgpr_private_segment_buffer 1
// ASM-NEXT: .amdhsa_user_sgpr_dispatch_ptr 1
// ASM-NEXT: .amdhsa_user_sgpr_queue_ptr 1
// ASM-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 1
// ASM-NEXT: .amdhsa_user_sgpr_dispatch_id 1
// ASM-NEXT: .amdhsa_user_sgpr_flat_scratch_init 1
// ASM-NEXT: .amdhsa_user_sgpr_private_segment_size 1
// ASM-NEXT: .amdhsa_uses_dynamic_stack 1
// ASM-NEXT: .amdhsa_system_sgpr_private_segment_wavefront_offset 1
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_x 0
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_y 1
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_z 1
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_info 1
// ASM-NEXT: .amdhsa_system_vgpr_workitem_id 1
// ASM-NEXT: .amdhsa_next_free_vgpr 9
// ASM-NEXT: .amdhsa_next_free_sgpr 27
// ASM-NEXT: .amdhsa_reserve_vcc 0
// ASM-NEXT: .amdhsa_reserve_flat_scratch 0
// ASM-NEXT: .amdhsa_reserve_xnack_mask 1
// ASM-NEXT: .amdhsa_float_round_mode_32 1
// ASM-NEXT: .amdhsa_float_round_mode_16_64 1
// ASM-NEXT: .amdhsa_float_denorm_mode_32 1
// ASM-NEXT: .amdhsa_float_denorm_mode_16_64 0
// ASM-NEXT: .amdhsa_dx10_clamp 0
// ASM-NEXT: .amdhsa_ieee_mode 0
// ASM-NEXT: .amdhsa_fp16_overflow 1
// ASM-NEXT: .amdhsa_exception_fp_ieee_invalid_op 1
// ASM-NEXT: .amdhsa_exception_fp_denorm_src 1
// ASM-NEXT: .amdhsa_exception_fp_ieee_div_zero 1
// ASM-NEXT: .amdhsa_exception_fp_ieee_overflow 1
// ASM-NEXT: .amdhsa_exception_fp_ieee_underflow 1
// ASM-NEXT: .amdhsa_exception_fp_ieee_inexact 1
// ASM-NEXT: .amdhsa_exception_int_div_zero 1
// ASM-NEXT: .end_amdhsa_kernel

// Test that we are including special SGPR usage in the granulated count.
.p2align 6
.amdhsa_kernel special_sgpr
  // Same next_free_sgpr as "complete", but...
  .amdhsa_next_free_sgpr 27
  // ...on GFX9 this should require an additional 6 SGPRs, pushing us from
  // 3 granules to 4
  .amdhsa_reserve_flat_scratch 1

  .amdhsa_reserve_vcc 0
  .amdhsa_reserve_xnack_mask 1

  .amdhsa_float_denorm_mode_16_64 0
  .amdhsa_dx10_clamp 0
  .amdhsa_ieee_mode 0
  .amdhsa_next_free_vgpr 0
.end_amdhsa_kernel

// ASM: .amdhsa_kernel special_sgpr
// ASM: .amdhsa_next_free_vgpr 0
// ASM-NEXT: .amdhsa_next_free_sgpr 27
// ASM-NEXT: .amdhsa_reserve_vcc 0
// ASM-NEXT: .amdhsa_reserve_xnack_mask 1
// ASM: .amdhsa_float_denorm_mode_16_64 0
// ASM-NEXT: .amdhsa_dx10_clamp 0
// ASM-NEXT: .amdhsa_ieee_mode 0
// ASM: .end_amdhsa_kernel

// Test that explicitly disabling user_sgpr's does not affect the user_sgpr
// count, i.e. this should produce the same descriptor as minimal.
.p2align 6
.amdhsa_kernel disabled_user_sgpr
  .amdhsa_user_sgpr_private_segment_buffer 0
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

// ASM: .amdhsa_kernel disabled_user_sgpr
// ASM: .amdhsa_next_free_vgpr 0
// ASM-NEXT: .amdhsa_next_free_sgpr 0
// ASM: .end_amdhsa_kernel

.section .foo

.byte .amdgcn.gfx_generation_number
// ASM: .byte 9

.byte .amdgcn.gfx_generation_minor
// ASM: .byte 0

.byte .amdgcn.gfx_generation_stepping
// ASM: .byte 4

.byte .amdgcn.next_free_vgpr
// ASM: .byte 0
.byte .amdgcn.next_free_sgpr
// ASM: .byte 0

v_mov_b32_e32 v7, s10

.byte .amdgcn.next_free_vgpr
// ASM: .byte 8
.byte .amdgcn.next_free_sgpr
// ASM: .byte 11

.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0

.byte .amdgcn.next_free_vgpr
// ASM: .byte 0
.byte .amdgcn.next_free_sgpr
// ASM: .byte 0

v_mov_b32_e32 v16, s3

.byte .amdgcn.next_free_vgpr
// ASM: .byte 17
.byte .amdgcn.next_free_sgpr
// ASM: .byte 4

// Metadata

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name:       amd_kernel_code_t_test_all
      .symbol: amd_kernel_code_t_test_all@kd
      .kernarg_segment_size: 8
      .group_segment_fixed_size: 16
      .private_segment_fixed_size: 32
      .uses_dynamic_stack: true
      .kernarg_segment_align: 64
      .wavefront_size: 128
      .sgpr_count: 14
      .vgpr_count: 40
      .max_flat_workgroup_size: 256
    - .name:       amd_kernel_code_t_minimal
      .symbol: amd_kernel_code_t_minimal@kd
      .kernarg_segment_size: 8
      .group_segment_fixed_size: 16
      .private_segment_fixed_size: 32
      .uses_dynamic_stack: true
      .kernarg_segment_align: 64
      .wavefront_size: 128
      .sgpr_count: 14
      .vgpr_count: 40
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata

// ASM:      	.amdgpu_metadata
// ASM:      amdhsa.kernels:
// ASM:        - .group_segment_fixed_size: 16
// ASM:          .kernarg_segment_align: 64
// ASM:          .kernarg_segment_size: 8
// ASM:          .max_flat_workgroup_size: 256
// ASM:          .name:           amd_kernel_code_t_test_all
// ASM:          .private_segment_fixed_size: 32
// ASM:          .sgpr_count:     14
// ASM:          .symbol:         'amd_kernel_code_t_test_all@kd'
// ASM:          .uses_dynamic_stack: true
// ASM:          .vgpr_count:     40
// ASM:          .wavefront_size: 128
// ASM:        - .group_segment_fixed_size: 16
// ASM:          .kernarg_segment_align: 64
// ASM:          .kernarg_segment_size: 8
// ASM:          .max_flat_workgroup_size: 256
// ASM:          .name:           amd_kernel_code_t_minimal
// ASM:          .private_segment_fixed_size: 32
// ASM:          .sgpr_count:     14
// ASM:          .symbol:         'amd_kernel_code_t_minimal@kd'
// ASM:          .uses_dynamic_stack: true
// ASM:          .vgpr_count:     40
// ASM:          .wavefront_size: 128
// ASM:      amdhsa.version:
// ASM-NEXT:   - 3
// ASM-NEXT:   - 0
// ASM:      	.end_amdgpu_metadata
