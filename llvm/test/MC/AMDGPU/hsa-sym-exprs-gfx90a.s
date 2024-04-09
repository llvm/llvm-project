// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx90a -filetype=obj < %s > %t
// RUN: llvm-objdump -s -j .rodata %t | FileCheck --check-prefix=OBJDUMP %s

// When going from asm -> asm, the expressions should remain the same (i.e., symbolic).
// When going from asm -> obj, the expressions should get resolved (through fixups),

// OBJDUMP: Contents of section .rodata
// expr_defined_later
// OBJDUMP-NEXT: 0000 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0010 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0020 00000000 00000000 00000000 00000100
// OBJDUMP-NEXT: 0030 0000ac04 81000000 00000000 00000000
// expr_defined
// OBJDUMP-NEXT: 0040 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0050 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0060 00000000 00000000 00000000 00000100
// OBJDUMP-NEXT: 0070 0000ac04 81000000 00000000 00000000

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
  .amdhsa_system_sgpr_private_segment_wavefront_offset defined_boolean
  .amdhsa_dx10_clamp defined_boolean
  .amdhsa_ieee_mode defined_boolean
  .amdhsa_fp16_overflow defined_boolean
  .amdhsa_tg_split defined_boolean
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
.end_amdhsa_kernel

.set defined_boolean, 1

.p2align 6
.amdhsa_kernel expr_defined
  .amdhsa_system_sgpr_private_segment_wavefront_offset defined_boolean
  .amdhsa_dx10_clamp defined_boolean
  .amdhsa_ieee_mode defined_boolean
  .amdhsa_fp16_overflow defined_boolean
  .amdhsa_tg_split defined_boolean
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
.end_amdhsa_kernel

// ASM: .amdhsa_kernel expr_defined_later
// ASM-NEXT: .amdhsa_group_segment_fixed_size 0
// ASM-NEXT: .amdhsa_private_segment_fixed_size 0
// ASM-NEXT: .amdhsa_kernarg_size 0
// ASM-NEXT: .amdhsa_user_sgpr_count (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&62)>>1
// ASM-NEXT: .amdhsa_user_sgpr_private_segment_buffer 0
// ASM-NEXT: .amdhsa_user_sgpr_dispatch_ptr 0
// ASM-NEXT: .amdhsa_user_sgpr_queue_ptr 0
// ASM-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
// ASM-NEXT: .amdhsa_user_sgpr_dispatch_id 0
// ASM-NEXT: .amdhsa_user_sgpr_flat_scratch_init 0
// ASM-NEXT: .amdhsa_user_sgpr_kernarg_preload_length 0
// ASM-NEXT: .amdhsa_user_sgpr_kernarg_preload_offset 0
// ASM-NEXT: .amdhsa_user_sgpr_private_segment_size 0
// ASM-NEXT: .amdhsa_system_sgpr_private_segment_wavefront_offset (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&1)>>0
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_x (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&128)>>7
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_y (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&256)>>8
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_z (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&512)>>9
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_info (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&1024)>>10
// ASM-NEXT: .amdhsa_system_vgpr_workitem_id (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&6144)>>11
// ASM-NEXT: .amdhsa_next_free_vgpr 0
// ASM-NEXT: .amdhsa_next_free_sgpr 0
// ASM-NEXT: .amdhsa_accum_offset (((((((0&(~65536))|(defined_boolean<<16))&(~63))|(0<<0))&63)>>0)+1)*4
// ASM-NEXT: .amdhsa_reserve_xnack_mask 1
// ASM-NEXT: .amdhsa_float_round_mode_32 (((((((((((((((((0&(~786432))|(3<<18))&(~2097152))|(1<<21))&(~8388608))|(1<<23))&(~2097152))|(defined_boolean<<21))&(~8388608))|(defined_boolean<<23))&(~67108864))|(defined_boolean<<26))&(~63))|(0<<0))&(~960))|(0<<6))&12288)>>12
// ASM-NEXT: .amdhsa_float_round_mode_16_64 (((((((((((((((((0&(~786432))|(3<<18))&(~2097152))|(1<<21))&(~8388608))|(1<<23))&(~2097152))|(defined_boolean<<21))&(~8388608))|(defined_boolean<<23))&(~67108864))|(defined_boolean<<26))&(~63))|(0<<0))&(~960))|(0<<6))&49152)>>14
// ASM-NEXT: .amdhsa_float_denorm_mode_32 (((((((((((((((((0&(~786432))|(3<<18))&(~2097152))|(1<<21))&(~8388608))|(1<<23))&(~2097152))|(defined_boolean<<21))&(~8388608))|(defined_boolean<<23))&(~67108864))|(defined_boolean<<26))&(~63))|(0<<0))&(~960))|(0<<6))&196608)>>16
// ASM-NEXT: .amdhsa_float_denorm_mode_16_64 (((((((((((((((((0&(~786432))|(3<<18))&(~2097152))|(1<<21))&(~8388608))|(1<<23))&(~2097152))|(defined_boolean<<21))&(~8388608))|(defined_boolean<<23))&(~67108864))|(defined_boolean<<26))&(~63))|(0<<0))&(~960))|(0<<6))&786432)>>18
// ASM-NEXT: .amdhsa_dx10_clamp (((((((((((((((((0&(~786432))|(3<<18))&(~2097152))|(1<<21))&(~8388608))|(1<<23))&(~2097152))|(defined_boolean<<21))&(~8388608))|(defined_boolean<<23))&(~67108864))|(defined_boolean<<26))&(~63))|(0<<0))&(~960))|(0<<6))&2097152)>>21
// ASM-NEXT: .amdhsa_ieee_mode (((((((((((((((((0&(~786432))|(3<<18))&(~2097152))|(1<<21))&(~8388608))|(1<<23))&(~2097152))|(defined_boolean<<21))&(~8388608))|(defined_boolean<<23))&(~67108864))|(defined_boolean<<26))&(~63))|(0<<0))&(~960))|(0<<6))&8388608)>>23
// ASM-NEXT: .amdhsa_fp16_overflow (((((((((((((((((0&(~786432))|(3<<18))&(~2097152))|(1<<21))&(~8388608))|(1<<23))&(~2097152))|(defined_boolean<<21))&(~8388608))|(defined_boolean<<23))&(~67108864))|(defined_boolean<<26))&(~63))|(0<<0))&(~960))|(0<<6))&67108864)>>26
// ASM-NEXT: .amdhsa_tg_split (((((0&(~65536))|(defined_boolean<<16))&(~63))|(0<<0))&65536)>>16
// ASM-NEXT: .amdhsa_exception_fp_ieee_invalid_op (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&16777216)>>24
// ASM-NEXT: .amdhsa_exception_fp_denorm_src (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&33554432)>>25
// ASM-NEXT: .amdhsa_exception_fp_ieee_div_zero (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&67108864)>>26
// ASM-NEXT: .amdhsa_exception_fp_ieee_overflow (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&134217728)>>27
// ASM-NEXT: .amdhsa_exception_fp_ieee_underflow (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&268435456)>>28
// ASM-NEXT: .amdhsa_exception_fp_ieee_inexact (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&536870912)>>29
// ASM-NEXT: .amdhsa_exception_int_div_zero (((((((0&(~128))|(1<<7))&(~1))|(defined_boolean<<0))&(~62))|(0<<1))&1073741824)>>30
// ASM-NEXT: .end_amdhsa_kernel

// ASM:       .set defined_boolean, 1
// ASM-NEXT:  .no_dead_strip defined_boolean

// ASM: .amdhsa_kernel expr_defined
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
// ASM-NEXT: .amdhsa_system_sgpr_private_segment_wavefront_offset 1
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_x 1
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_y 0
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_z 0
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_info 0
// ASM-NEXT: .amdhsa_system_vgpr_workitem_id 0
// ASM-NEXT: .amdhsa_next_free_vgpr 0
// ASM-NEXT: .amdhsa_next_free_sgpr 0
// ASM-NEXT: .amdhsa_accum_offset 4
// ASM-NEXT: .amdhsa_reserve_xnack_mask 1
// ASM-NEXT: .amdhsa_float_round_mode_32 0
// ASM-NEXT: .amdhsa_float_round_mode_16_64 0
// ASM-NEXT: .amdhsa_float_denorm_mode_32 0
// ASM-NEXT: .amdhsa_float_denorm_mode_16_64 3
// ASM-NEXT: .amdhsa_dx10_clamp 1
// ASM-NEXT: .amdhsa_ieee_mode 1
// ASM-NEXT: .amdhsa_fp16_overflow 1
// ASM-NEXT: .amdhsa_tg_split 1
// ASM-NEXT: .amdhsa_exception_fp_ieee_invalid_op 0
// ASM-NEXT: .amdhsa_exception_fp_denorm_src 0
// ASM-NEXT: .amdhsa_exception_fp_ieee_div_zero 0
// ASM-NEXT: .amdhsa_exception_fp_ieee_overflow 0
// ASM-NEXT: .amdhsa_exception_fp_ieee_underflow 0
// ASM-NEXT: .amdhsa_exception_fp_ieee_inexact 0
// ASM-NEXT: .amdhsa_exception_int_div_zero 0
// ASM-NEXT: .end_amdhsa_kernel
