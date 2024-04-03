// RUN: llvm-mc -triple amdgcn--amdhsa -mcpu=kaveri -show-encoding %s | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=kaveri -show-encoding %s | llvm-readobj --symbols -S --sd - | FileCheck %s --check-prefix=ELF

// ELF: Section {
// ELF: Name: .text
// ELF: Type: SHT_PROGBITS (0x1)
// ELF: Flags [ (0x6)
// ELF: SHF_ALLOC (0x2)
// ELF: SHF_EXECINSTR (0x4)

// ELF: Symbol {
// ELF: Name: minimal
// ELF: Section: .text
// ELF: }

.text
// ASM: .text

.amdgcn_target "amdgcn-unknown-amdhsa--gfx700"
// ASM: .amdgcn_target "amdgcn-unknown-amdhsa--gfx700"

.amdhsa_code_object_version 4
// ASM: .amdhsa_code_object_version 4

.set my_is_ptr64, 1

.if my_is_ptr64 == 0
.set my_next_free_vgpr, 4
.else
.set my_next_free_vgpr, 8
.endif

.set my_sgpr, 6

minimal:
.amdhsa_kernel minimal
  .amdhsa_next_free_vgpr 1+(my_next_free_vgpr-1)
  // Make sure a blank line won't break anything:

  .amdhsa_next_free_sgpr my_sgpr/2+3
.end_amdhsa_kernel

; ASM-LABEL: minimal:
; ASM: .amdhsa_kernel minimal
; ASM:         .amdhsa_group_segment_fixed_size 0
; ASM:         .amdhsa_private_segment_fixed_size 0
; ASM:         .amdhsa_kernarg_size 0
; ASM:         .amdhsa_user_sgpr_count 0
; ASM:         .amdhsa_user_sgpr_private_segment_buffer 0
; ASM:         .amdhsa_user_sgpr_dispatch_ptr 0
; ASM:         .amdhsa_user_sgpr_queue_ptr 0
; ASM:         .amdhsa_user_sgpr_kernarg_segment_ptr 0
; ASM:         .amdhsa_user_sgpr_dispatch_id 0
; ASM:         .amdhsa_user_sgpr_flat_scratch_init 0
; ASM:         .amdhsa_user_sgpr_private_segment_size 0
; ASM:         .amdhsa_system_sgpr_private_segment_wavefront_offset 0
; ASM:         .amdhsa_system_sgpr_workgroup_id_x 1
; ASM:         .amdhsa_system_sgpr_workgroup_id_y 0
; ASM:         .amdhsa_system_sgpr_workgroup_id_z 0
; ASM:         .amdhsa_system_sgpr_workgroup_info 0
; ASM:         .amdhsa_system_vgpr_workitem_id 0
; ASM:         .amdhsa_next_free_vgpr 8
; ASM:         .amdhsa_next_free_sgpr 6
; ASM:         .amdhsa_float_round_mode_32 0
; ASM:         .amdhsa_float_round_mode_16_64 0
; ASM:         .amdhsa_float_denorm_mode_32 0
; ASM:         .amdhsa_float_denorm_mode_16_64 3
; ASM:         .amdhsa_dx10_clamp 1
; ASM:         .amdhsa_ieee_mode 1
; ASM:         .amdhsa_exception_fp_ieee_invalid_op 0
; ASM:         .amdhsa_exception_fp_denorm_src 0
; ASM:         .amdhsa_exception_fp_ieee_div_zero 0
; ASM:         .amdhsa_exception_fp_ieee_overflow 0
; ASM:         .amdhsa_exception_fp_ieee_underflow 0
; ASM:         .amdhsa_exception_fp_ieee_inexact 0
; ASM:         .amdhsa_exception_int_div_zero 0
; ASM: .end_amdhsa_kernel
