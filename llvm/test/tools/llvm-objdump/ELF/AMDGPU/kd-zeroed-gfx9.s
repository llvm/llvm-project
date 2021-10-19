;; Entirely zeroed kernel descriptor (for GFX9).

; RUN: llvm-mc %s --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=-xnack -filetype=obj -o %t1
; RUN: llvm-objdump --disassemble-symbols=my_kernel.kd %t1 \
; RUN: | tail -n +7 | llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=-xnack -filetype=obj -o %t2
; RUN: diff %t1 %t2

; RUN: llvm-objdump -s -j .text %t1 | FileCheck --check-prefix=OBJDUMP %s

; OBJDUMP:      0000 00000000 00000000 00000000 00000000
; OBJDUMP-NEXT: 0010 00000000 00000000 00000000 00000000
; OBJDUMP-NEXT: 0020 00000000 00000000 00000000 00000000
; OBJDUMP-NEXT: 0030 00000000 00000000 00000000 00000000

;; This file and kd-zeroed-raw.s produce the same output for the kernel
;; descriptor - a block of 64 zeroed bytes.

.amdhsa_kernel my_kernel
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_next_free_vgpr 0
  .amdhsa_reserve_vcc 0
  .amdhsa_reserve_flat_scratch 0
  .amdhsa_reserve_xnack_mask 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_float_round_mode_32 0
  .amdhsa_float_round_mode_16_64 0
  .amdhsa_float_denorm_mode_32 0
  .amdhsa_float_denorm_mode_16_64 0
  .amdhsa_dx10_clamp 0
  .amdhsa_ieee_mode 0
  .amdhsa_fp16_overflow 0
  .amdhsa_system_sgpr_private_segment_wavefront_offset 0
  .amdhsa_system_sgpr_workgroup_id_x 0
  .amdhsa_system_sgpr_workgroup_id_y 0
  .amdhsa_system_sgpr_workgroup_id_z 0
  .amdhsa_system_sgpr_workgroup_info 0
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_exception_fp_ieee_invalid_op 0
  .amdhsa_exception_fp_denorm_src 0
  .amdhsa_exception_fp_ieee_div_zero 0
  .amdhsa_exception_fp_ieee_overflow 0
  .amdhsa_exception_fp_ieee_underflow 0
  .amdhsa_exception_fp_ieee_inexact 0
  .amdhsa_exception_int_div_zero 0
  .amdhsa_user_sgpr_private_segment_buffer 0
  .amdhsa_user_sgpr_dispatch_ptr 0
  .amdhsa_user_sgpr_queue_ptr 0
  .amdhsa_user_sgpr_kernarg_segment_ptr 0
  .amdhsa_user_sgpr_dispatch_id 0
  .amdhsa_user_sgpr_flat_scratch_init 0
  .amdhsa_user_sgpr_private_segment_size 0
.end_amdhsa_kernel
