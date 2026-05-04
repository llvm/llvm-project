// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx90a -mattr=+xnack,+tgsplit < %s | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx90a -mattr=+xnack,+tgsplit -filetype=obj < %s > %t
// RUN: llvm-objdump -s -j .rodata %t | FileCheck --check-prefix=OBJDUMP %s

// OBJDUMP: Contents of section .rodata
// OBJDUMP-NEXT: 0000 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0010 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0020 00000000 00000000 00000000 00000100
// OBJDUMP-NEXT: 0030 0000ac00 80000000 00000000 00000000

.amdgcn_target "amdgcn-amd-amdhsa--gfx90a:xnack+"
// ASM: .amdgcn_target "amdgcn-amd-amdhsa--gfx90a:xnack+"

.amdhsa_code_object_version 4
// ASM: .amdhsa_code_object_version 4

.p2align 8
.type minimal,@function
minimal:
  s_endpgm

.rodata
// ASM: .rodata

.p2align 6
.amdhsa_kernel minimal
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
.end_amdhsa_kernel

// ASM: .amdhsa_kernel minimal
// ASM-NEXT: .amdhsa_group_segment_fixed_size 0
// ASM-NEXT: .amdhsa_private_segment_fixed_size 0
// ASM-NEXT: .amdhsa_kernarg_size 0
// ASM-NEXT: .amdhsa_user_sgpr_count 0
// ASM-NEXT: .amdhsa_user_sgpr_private_segment_buffer 0
// ASM-NEXT: .amdhsa_user_sgpr_dispatch_ptr 0
// ASM-NEXT: .amdhsa_user_sgpr_queue_ptr 0
// ASM-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
// ASM-NEXT: .amdhsa_user_sgpr_dispatch_id 0
// ASM-NEXT: .amdhsa_user_sgpr_flat_scratch_init 0
// ASM-NEXT: .amdhsa_user_sgpr_kernarg_preload_length 0
// ASM-NEXT: .amdhsa_user_sgpr_kernarg_preload_offset 0
// ASM-NEXT: .amdhsa_user_sgpr_private_segment_size 0
// ASM-NEXT: .amdhsa_system_sgpr_private_segment_wavefront_offset 0
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_x 1
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_y 0
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_z 0
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_info 0
// ASM-NEXT: .amdhsa_system_vgpr_workitem_id 0
// ASM-NEXT: .amdhsa_next_free_vgpr 0
// ASM-NEXT: .amdhsa_next_free_sgpr 0
// ASM-NEXT: .amdhsa_accum_offset 4
// ASM-NEXT: .amdhsa_reserve_vcc 1
// ASM-NEXT: .amdhsa_reserve_flat_scratch 1
// ASM-NEXT: .amdhsa_reserve_xnack_mask 1
// ASM-NEXT: .amdhsa_float_round_mode_32 0
// ASM-NEXT: .amdhsa_float_round_mode_16_64 0
// ASM-NEXT: .amdhsa_float_denorm_mode_32 0
// ASM-NEXT: .amdhsa_float_denorm_mode_16_64 3
// ASM-NEXT: .amdhsa_dx10_clamp 1
// ASM-NEXT: .amdhsa_ieee_mode 1
// ASM-NEXT: .amdhsa_fp16_overflow 0
// ASM-NEXT: .amdhsa_tg_split 1
// ASM-NEXT: .amdhsa_exception_fp_ieee_invalid_op 0
// ASM-NEXT: .amdhsa_exception_fp_denorm_src 0
// ASM-NEXT: .amdhsa_exception_fp_ieee_div_zero 0
// ASM-NEXT: .amdhsa_exception_fp_ieee_overflow 0
// ASM-NEXT: .amdhsa_exception_fp_ieee_underflow 0
// ASM-NEXT: .amdhsa_exception_fp_ieee_inexact 0
// ASM-NEXT: .amdhsa_exception_int_div_zero 0
// ASM-NEXT: .end_amdhsa_kernel
