// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx700 < %s | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj < %s > %t
// RUN: llvm-objdump -s -j .rodata %t | FileCheck --check-prefix=OBJDUMP %s

// When going from asm -> asm, the expressions should remain the same (i.e., symbolic).
// When going from asm -> obj, the expressions should get resolved (through fixups),

// OBJDUMP: Contents of section .rodata
// expr_defined_later
// OBJDUMP-NEXT: 0000 2b000000 2c000000 00000000 00000000
// OBJDUMP-NEXT: 0010 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0020 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0030 8bf1af00 801f007f 00080000 00000000
// expr_defined
// OBJDUMP-NEXT: 0040 2a000000 2b000000 00000000 00000000
// OBJDUMP-NEXT: 0050 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0060 00000000 00000000 00000000 00000000
// OBJDUMP-NEXT: 0070 8af1af00 801f007f 00080000 00000000

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
  .amdhsa_private_segment_fixed_size defined_value+3
  .amdhsa_system_vgpr_workitem_id defined_2_bits
  .amdhsa_float_round_mode_32 defined_2_bits
  .amdhsa_float_round_mode_16_64 defined_2_bits
  .amdhsa_float_denorm_mode_32 defined_2_bits
  .amdhsa_float_denorm_mode_16_64 defined_2_bits
  .amdhsa_system_sgpr_workgroup_id_x defined_boolean
  .amdhsa_system_sgpr_workgroup_id_y defined_boolean
  .amdhsa_system_sgpr_workgroup_id_z defined_boolean
  .amdhsa_system_sgpr_workgroup_info defined_boolean
  .amdhsa_exception_fp_ieee_invalid_op defined_boolean
  .amdhsa_exception_fp_denorm_src defined_boolean
  .amdhsa_exception_fp_ieee_div_zero defined_boolean
  .amdhsa_exception_fp_ieee_overflow defined_boolean
  .amdhsa_exception_fp_ieee_underflow defined_boolean
  .amdhsa_exception_fp_ieee_inexact defined_boolean
  .amdhsa_exception_int_div_zero defined_boolean
  .amdhsa_uses_dynamic_stack defined_boolean
  .amdhsa_next_free_vgpr defined_value+4
  .amdhsa_next_free_sgpr defined_value+5
  .amdhsa_reserve_vcc defined_boolean
  .amdhsa_reserve_flat_scratch defined_boolean
.end_amdhsa_kernel

.set defined_value, 41
.set defined_2_bits, 3
.set defined_boolean, 1

.p2align 6
.amdhsa_kernel expr_defined
  .amdhsa_group_segment_fixed_size defined_value+1
  .amdhsa_private_segment_fixed_size defined_value+2
  .amdhsa_system_vgpr_workitem_id defined_2_bits
  .amdhsa_float_round_mode_32 defined_2_bits
  .amdhsa_float_round_mode_16_64 defined_2_bits
  .amdhsa_float_denorm_mode_32 defined_2_bits
  .amdhsa_float_denorm_mode_16_64 defined_2_bits
  .amdhsa_system_sgpr_workgroup_id_x defined_boolean
  .amdhsa_system_sgpr_workgroup_id_y defined_boolean
  .amdhsa_system_sgpr_workgroup_id_z defined_boolean
  .amdhsa_system_sgpr_workgroup_info defined_boolean
  .amdhsa_exception_fp_ieee_invalid_op defined_boolean
  .amdhsa_exception_fp_denorm_src defined_boolean
  .amdhsa_exception_fp_ieee_div_zero defined_boolean
  .amdhsa_exception_fp_ieee_overflow defined_boolean
  .amdhsa_exception_fp_ieee_underflow defined_boolean
  .amdhsa_exception_fp_ieee_inexact defined_boolean
  .amdhsa_exception_int_div_zero defined_boolean
  .amdhsa_uses_dynamic_stack defined_boolean
  .amdhsa_next_free_vgpr defined_value+3
  .amdhsa_next_free_sgpr defined_value+4
  .amdhsa_reserve_vcc defined_boolean
  .amdhsa_reserve_flat_scratch defined_boolean
.end_amdhsa_kernel

// ASM: .amdhsa_kernel expr_defined_later
// ASM-NEXT: .amdhsa_group_segment_fixed_size defined_value+2
// ASM-NEXT: .amdhsa_private_segment_fixed_size defined_value+3
// ASM-NEXT: .amdhsa_kernarg_size 0
// ASM-NEXT: .amdhsa_user_sgpr_count 0
// ASM-NEXT: .amdhsa_user_sgpr_private_segment_buffer 0
// ASM-NEXT: .amdhsa_user_sgpr_dispatch_ptr 0
// ASM-NEXT: .amdhsa_user_sgpr_queue_ptr 0
// ASM-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
// ASM-NEXT: .amdhsa_user_sgpr_dispatch_id 0
// ASM-NEXT: .amdhsa_user_sgpr_flat_scratch_init 0
// ASM-NEXT: .amdhsa_user_sgpr_private_segment_size 0
// ASM-NEXT: .amdhsa_system_sgpr_private_segment_wavefront_offset ((((((((((((((((((((((((128|(defined_2_bits<<11))&(~128))|(defined_boolean<<7))&(~256))|(defined_boolean<<8))&(~512))|(defined_boolean<<9))&(~1024))|(defined_boolean<<10))&(~16777216))|(defined_boolean<<24))&(~33554432))|(defined_boolean<<25))&(~67108864))|(defined_boolean<<26))&(~134217728))|(defined_boolean<<27))&(~268435456))|(defined_boolean<<28))&(~536870912))|(defined_boolean<<29))&(~1073741824))|(defined_boolean<<30))&(~62))&1
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_x (((((((((((((((((((((((((128|(defined_2_bits<<11))&(~128))|(defined_boolean<<7))&(~256))|(defined_boolean<<8))&(~512))|(defined_boolean<<9))&(~1024))|(defined_boolean<<10))&(~16777216))|(defined_boolean<<24))&(~33554432))|(defined_boolean<<25))&(~67108864))|(defined_boolean<<26))&(~134217728))|(defined_boolean<<27))&(~268435456))|(defined_boolean<<28))&(~536870912))|(defined_boolean<<29))&(~1073741824))|(defined_boolean<<30))&(~62))&128)>>7
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_y (((((((((((((((((((((((((128|(defined_2_bits<<11))&(~128))|(defined_boolean<<7))&(~256))|(defined_boolean<<8))&(~512))|(defined_boolean<<9))&(~1024))|(defined_boolean<<10))&(~16777216))|(defined_boolean<<24))&(~33554432))|(defined_boolean<<25))&(~67108864))|(defined_boolean<<26))&(~134217728))|(defined_boolean<<27))&(~268435456))|(defined_boolean<<28))&(~536870912))|(defined_boolean<<29))&(~1073741824))|(defined_boolean<<30))&(~62))&256)>>8
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_z (((((((((((((((((((((((((128|(defined_2_bits<<11))&(~128))|(defined_boolean<<7))&(~256))|(defined_boolean<<8))&(~512))|(defined_boolean<<9))&(~1024))|(defined_boolean<<10))&(~16777216))|(defined_boolean<<24))&(~33554432))|(defined_boolean<<25))&(~67108864))|(defined_boolean<<26))&(~134217728))|(defined_boolean<<27))&(~268435456))|(defined_boolean<<28))&(~536870912))|(defined_boolean<<29))&(~1073741824))|(defined_boolean<<30))&(~62))&512)>>9
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_info (((((((((((((((((((((((((128|(defined_2_bits<<11))&(~128))|(defined_boolean<<7))&(~256))|(defined_boolean<<8))&(~512))|(defined_boolean<<9))&(~1024))|(defined_boolean<<10))&(~16777216))|(defined_boolean<<24))&(~33554432))|(defined_boolean<<25))&(~67108864))|(defined_boolean<<26))&(~134217728))|(defined_boolean<<27))&(~268435456))|(defined_boolean<<28))&(~536870912))|(defined_boolean<<29))&(~1073741824))|(defined_boolean<<30))&(~62))&1024)>>10
// ASM-NEXT: .amdhsa_system_vgpr_workitem_id (((((((((((((((((((((((((128|(defined_2_bits<<11))&(~128))|(defined_boolean<<7))&(~256))|(defined_boolean<<8))&(~512))|(defined_boolean<<9))&(~1024))|(defined_boolean<<10))&(~16777216))|(defined_boolean<<24))&(~33554432))|(defined_boolean<<25))&(~67108864))|(defined_boolean<<26))&(~134217728))|(defined_boolean<<27))&(~268435456))|(defined_boolean<<28))&(~536870912))|(defined_boolean<<29))&(~1073741824))|(defined_boolean<<30))&(~62))&6144)>>11
// ASM-NEXT: .amdhsa_next_free_vgpr defined_value+4
// ASM-NEXT: .amdhsa_next_free_sgpr defined_value+5
// ASM-NEXT: .amdhsa_reserve_vcc defined_boolean
// ASM-NEXT: .amdhsa_reserve_flat_scratch defined_boolean
// ASM-NEXT: .amdhsa_float_round_mode_32 ((((((((((((11272192|(defined_2_bits<<12))&(~49152))|(defined_2_bits<<14))&(~196608))|(defined_2_bits<<16))&(~786432))|(defined_2_bits<<18))&(~63))|(((alignto(max(defined_value+4, 1), 4))/4)-1))&(~960))|((((alignto(max((defined_value+5)+(extrasgprs(defined_boolean, defined_boolean, 0)), 1), 8))/8)-1)<<6))&12288)>>12
// ASM-NEXT: .amdhsa_float_round_mode_16_64 ((((((((((((11272192|(defined_2_bits<<12))&(~49152))|(defined_2_bits<<14))&(~196608))|(defined_2_bits<<16))&(~786432))|(defined_2_bits<<18))&(~63))|(((alignto(max(defined_value+4, 1), 4))/4)-1))&(~960))|((((alignto(max((defined_value+5)+(extrasgprs(defined_boolean, defined_boolean, 0)), 1), 8))/8)-1)<<6))&49152)>>14
// ASM-NEXT: .amdhsa_float_denorm_mode_32 ((((((((((((11272192|(defined_2_bits<<12))&(~49152))|(defined_2_bits<<14))&(~196608))|(defined_2_bits<<16))&(~786432))|(defined_2_bits<<18))&(~63))|(((alignto(max(defined_value+4, 1), 4))/4)-1))&(~960))|((((alignto(max((defined_value+5)+(extrasgprs(defined_boolean, defined_boolean, 0)), 1), 8))/8)-1)<<6))&196608)>>16
// ASM-NEXT: .amdhsa_float_denorm_mode_16_64 ((((((((((((11272192|(defined_2_bits<<12))&(~49152))|(defined_2_bits<<14))&(~196608))|(defined_2_bits<<16))&(~786432))|(defined_2_bits<<18))&(~63))|(((alignto(max(defined_value+4, 1), 4))/4)-1))&(~960))|((((alignto(max((defined_value+5)+(extrasgprs(defined_boolean, defined_boolean, 0)), 1), 8))/8)-1)<<6))&786432)>>18
// ASM-NEXT: .amdhsa_dx10_clamp 1
// ASM-NEXT: .amdhsa_ieee_mode 1
// ASM-NEXT: .amdhsa_exception_fp_ieee_invalid_op (((((((((((((((((((((((((128|(defined_2_bits<<11))&(~128))|(defined_boolean<<7))&(~256))|(defined_boolean<<8))&(~512))|(defined_boolean<<9))&(~1024))|(defined_boolean<<10))&(~16777216))|(defined_boolean<<24))&(~33554432))|(defined_boolean<<25))&(~67108864))|(defined_boolean<<26))&(~134217728))|(defined_boolean<<27))&(~268435456))|(defined_boolean<<28))&(~536870912))|(defined_boolean<<29))&(~1073741824))|(defined_boolean<<30))&(~62))&16777216)>>24
// ASM-NEXT: .amdhsa_exception_fp_denorm_src (((((((((((((((((((((((((128|(defined_2_bits<<11))&(~128))|(defined_boolean<<7))&(~256))|(defined_boolean<<8))&(~512))|(defined_boolean<<9))&(~1024))|(defined_boolean<<10))&(~16777216))|(defined_boolean<<24))&(~33554432))|(defined_boolean<<25))&(~67108864))|(defined_boolean<<26))&(~134217728))|(defined_boolean<<27))&(~268435456))|(defined_boolean<<28))&(~536870912))|(defined_boolean<<29))&(~1073741824))|(defined_boolean<<30))&(~62))&33554432)>>25
// ASM-NEXT: .amdhsa_exception_fp_ieee_div_zero (((((((((((((((((((((((((128|(defined_2_bits<<11))&(~128))|(defined_boolean<<7))&(~256))|(defined_boolean<<8))&(~512))|(defined_boolean<<9))&(~1024))|(defined_boolean<<10))&(~16777216))|(defined_boolean<<24))&(~33554432))|(defined_boolean<<25))&(~67108864))|(defined_boolean<<26))&(~134217728))|(defined_boolean<<27))&(~268435456))|(defined_boolean<<28))&(~536870912))|(defined_boolean<<29))&(~1073741824))|(defined_boolean<<30))&(~62))&67108864)>>26
// ASM-NEXT: .amdhsa_exception_fp_ieee_overflow (((((((((((((((((((((((((128|(defined_2_bits<<11))&(~128))|(defined_boolean<<7))&(~256))|(defined_boolean<<8))&(~512))|(defined_boolean<<9))&(~1024))|(defined_boolean<<10))&(~16777216))|(defined_boolean<<24))&(~33554432))|(defined_boolean<<25))&(~67108864))|(defined_boolean<<26))&(~134217728))|(defined_boolean<<27))&(~268435456))|(defined_boolean<<28))&(~536870912))|(defined_boolean<<29))&(~1073741824))|(defined_boolean<<30))&(~62))&134217728)>>27
// ASM-NEXT: .amdhsa_exception_fp_ieee_underflow (((((((((((((((((((((((((128|(defined_2_bits<<11))&(~128))|(defined_boolean<<7))&(~256))|(defined_boolean<<8))&(~512))|(defined_boolean<<9))&(~1024))|(defined_boolean<<10))&(~16777216))|(defined_boolean<<24))&(~33554432))|(defined_boolean<<25))&(~67108864))|(defined_boolean<<26))&(~134217728))|(defined_boolean<<27))&(~268435456))|(defined_boolean<<28))&(~536870912))|(defined_boolean<<29))&(~1073741824))|(defined_boolean<<30))&(~62))&268435456)>>28
// ASM-NEXT: .amdhsa_exception_fp_ieee_inexact (((((((((((((((((((((((((128|(defined_2_bits<<11))&(~128))|(defined_boolean<<7))&(~256))|(defined_boolean<<8))&(~512))|(defined_boolean<<9))&(~1024))|(defined_boolean<<10))&(~16777216))|(defined_boolean<<24))&(~33554432))|(defined_boolean<<25))&(~67108864))|(defined_boolean<<26))&(~134217728))|(defined_boolean<<27))&(~268435456))|(defined_boolean<<28))&(~536870912))|(defined_boolean<<29))&(~1073741824))|(defined_boolean<<30))&(~62))&536870912)>>29
// ASM-NEXT: .amdhsa_exception_int_div_zero (((((((((((((((((((((((((128|(defined_2_bits<<11))&(~128))|(defined_boolean<<7))&(~256))|(defined_boolean<<8))&(~512))|(defined_boolean<<9))&(~1024))|(defined_boolean<<10))&(~16777216))|(defined_boolean<<24))&(~33554432))|(defined_boolean<<25))&(~67108864))|(defined_boolean<<26))&(~134217728))|(defined_boolean<<27))&(~268435456))|(defined_boolean<<28))&(~536870912))|(defined_boolean<<29))&(~1073741824))|(defined_boolean<<30))&(~62))&1073741824)>>30
// ASM-NEXT: .end_amdhsa_kernel

// ASM:       .set defined_value, 41
// ASM-NEXT:  .no_dead_strip defined_value
// ASM-NEXT:  .set defined_2_bits, 3
// ASM-NEXT:  .no_dead_strip defined_2_bits
// ASM-NEXT:  .set defined_boolean, 1
// ASM-NEXT:  .no_dead_strip defined_boolean

// ASM: .amdhsa_kernel expr_defined
// ASM-NEXT: .amdhsa_group_segment_fixed_size 42
// ASM-NEXT: .amdhsa_private_segment_fixed_size 43
// ASM-NEXT: .amdhsa_kernarg_size 0
// ASM-NEXT: .amdhsa_user_sgpr_count 0
// ASM-NEXT: .amdhsa_user_sgpr_private_segment_buffer 0
// ASM-NEXT: .amdhsa_user_sgpr_dispatch_ptr 0
// ASM-NEXT: .amdhsa_user_sgpr_queue_ptr 0
// ASM-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
// ASM-NEXT: .amdhsa_user_sgpr_dispatch_id 0
// ASM-NEXT: .amdhsa_user_sgpr_flat_scratch_init 0
// ASM-NEXT: .amdhsa_user_sgpr_private_segment_size 0
// ASM-NEXT: .amdhsa_system_sgpr_private_segment_wavefront_offset 0
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_x 1
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_y 1
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_id_z 1
// ASM-NEXT: .amdhsa_system_sgpr_workgroup_info 1
// ASM-NEXT: .amdhsa_system_vgpr_workitem_id 3
// ASM-NEXT: .amdhsa_next_free_vgpr 44
// ASM-NEXT: .amdhsa_next_free_sgpr 45
// ASM-NEXT: .amdhsa_reserve_vcc 1 
// ASM-NEXT: .amdhsa_reserve_flat_scratch 1
// ASM-NEXT: .amdhsa_float_round_mode_32 3
// ASM-NEXT: .amdhsa_float_round_mode_16_64 3
// ASM-NEXT: .amdhsa_float_denorm_mode_32 3
// ASM-NEXT: .amdhsa_float_denorm_mode_16_64 3
// ASM-NEXT: .amdhsa_dx10_clamp 1
// ASM-NEXT: .amdhsa_ieee_mode 1
// ASM-NEXT: .amdhsa_exception_fp_ieee_invalid_op 1
// ASM-NEXT: .amdhsa_exception_fp_denorm_src 1
// ASM-NEXT: .amdhsa_exception_fp_ieee_div_zero 1
// ASM-NEXT: .amdhsa_exception_fp_ieee_overflow 1
// ASM-NEXT: .amdhsa_exception_fp_ieee_underflow 1
// ASM-NEXT: .amdhsa_exception_fp_ieee_inexact 1
// ASM-NEXT: .amdhsa_exception_int_div_zero 1
// ASM-NEXT: .end_amdhsa_kernel
