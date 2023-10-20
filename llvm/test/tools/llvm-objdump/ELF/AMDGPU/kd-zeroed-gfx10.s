;; Entirely zeroed kernel descriptor (for GFX10).

; RUN: llvm-mc %s --triple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=-xnack -filetype=obj -o %t
; RUN: llvm-objdump -s -d -j .text %t | FileCheck --check-prefix=OBJDUMP %s

;; TODO:
;; This file and kd-zeroed-raw.s should produce the same output for the kernel
;; descriptor - a block of 64 zeroed bytes. But looks like the assembler sets
;; the FWD_PROGRESS bit in COMPUTE_PGM_RSRC1 to 1 even when the directive
;; mentions 0 (see line 36).

;; Check the raw bytes right now.

; OBJDUMP-LABEL: Contents of section .text:
; OBJDUMP-NEXT: 0000 00000000 00000000 00000000 00000000
; OBJDUMP-NEXT: 0010 00000000 00000000 00000000 00000000
; OBJDUMP-NEXT: 0020 00000000 00000000 00000000 00000000
; OBJDUMP-NEXT: 0030 01000000 00000000 00000000 00000000
; OBJDUMP-EMPTY:

; OBJDUMP-LABEL: Disassembly of section .text:
; OBJDUMP-EMPTY:
; OBJDUMP-NEXT: 0000000000000000 <my_kernel.kd>:
; OBJDUMP-NEXT: .amdhsa_kernel my_kernel
; OBJDUMP-NEXT:         .amdhsa_group_segment_fixed_size 0
; OBJDUMP-NEXT:         .amdhsa_private_segment_fixed_size 0
; OBJDUMP-NEXT:         .amdhsa_kernarg_size 0
; OBJDUMP-NEXT:         .amdhsa_shared_vgpr_count 0
; OBJDUMP-NEXT:         .amdhsa_next_free_vgpr 8
; OBJDUMP-NEXT:         .amdhsa_reserve_vcc 0
; OBJDUMP-NEXT:         .amdhsa_reserve_flat_scratch 0
; OBJDUMP-NEXT:         .amdhsa_reserve_xnack_mask 0
; OBJDUMP-NEXT:         .amdhsa_next_free_sgpr 8
; OBJDUMP-NEXT:         .amdhsa_float_round_mode_32 0
; OBJDUMP-NEXT:         .amdhsa_float_round_mode_16_64 0
; OBJDUMP-NEXT:         .amdhsa_float_denorm_mode_32 0
; OBJDUMP-NEXT:         .amdhsa_float_denorm_mode_16_64 0
; OBJDUMP-NEXT:         .amdhsa_dx10_clamp 0
; OBJDUMP-NEXT:         .amdhsa_ieee_mode 0
; OBJDUMP-NEXT:         .amdhsa_fp16_overflow 0
; OBJDUMP-NEXT:         .amdhsa_workgroup_processor_mode 0
; OBJDUMP-NEXT:         .amdhsa_memory_ordered 0
; OBJDUMP-NEXT:         .amdhsa_forward_progress 0
; OBJDUMP-NEXT:         .amdhsa_system_sgpr_private_segment_wavefront_offset 0
; OBJDUMP-NEXT:         .amdhsa_system_sgpr_workgroup_id_x 0
; OBJDUMP-NEXT:         .amdhsa_system_sgpr_workgroup_id_y 0
; OBJDUMP-NEXT:         .amdhsa_system_sgpr_workgroup_id_z 0
; OBJDUMP-NEXT:         .amdhsa_system_sgpr_workgroup_info 0
; OBJDUMP-NEXT:         .amdhsa_system_vgpr_workitem_id 0
; OBJDUMP-NEXT:         .amdhsa_exception_fp_ieee_invalid_op 0
; OBJDUMP-NEXT:         .amdhsa_exception_fp_denorm_src 0
; OBJDUMP-NEXT:         .amdhsa_exception_fp_ieee_div_zero 0
; OBJDUMP-NEXT:         .amdhsa_exception_fp_ieee_overflow 0
; OBJDUMP-NEXT:         .amdhsa_exception_fp_ieee_underflow 0
; OBJDUMP-NEXT:         .amdhsa_exception_fp_ieee_inexact 0
; OBJDUMP-NEXT:         .amdhsa_exception_int_div_zero 0
; OBJDUMP-NEXT:         .amdhsa_user_sgpr_private_segment_buffer 0
; OBJDUMP-NEXT:         .amdhsa_user_sgpr_dispatch_ptr 0
; OBJDUMP-NEXT:         .amdhsa_user_sgpr_queue_ptr 0
; OBJDUMP-NEXT:         .amdhsa_user_sgpr_kernarg_segment_ptr 0
; OBJDUMP-NEXT:         .amdhsa_user_sgpr_dispatch_id 0
; OBJDUMP-NEXT:         .amdhsa_user_sgpr_flat_scratch_init 0
; OBJDUMP-NEXT:         .amdhsa_user_sgpr_private_segment_size 0
; OBJDUMP-NEXT:         .amdhsa_wavefront_size32 0

.amdhsa_kernel my_kernel
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_next_free_vgpr 8
  .amdhsa_reserve_vcc 0
  .amdhsa_reserve_flat_scratch 0
  .amdhsa_reserve_xnack_mask 0
  .amdhsa_next_free_sgpr 8
  .amdhsa_float_round_mode_32 0
  .amdhsa_float_round_mode_16_64 0
  .amdhsa_float_denorm_mode_32 0
  .amdhsa_float_denorm_mode_16_64 0
  .amdhsa_dx10_clamp 0
  .amdhsa_ieee_mode 0
  .amdhsa_fp16_overflow 0
  .amdhsa_workgroup_processor_mode 0
  .amdhsa_memory_ordered 0
  .amdhsa_forward_progress 0
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
  .amdhsa_wavefront_size32 0
.end_amdhsa_kernel
