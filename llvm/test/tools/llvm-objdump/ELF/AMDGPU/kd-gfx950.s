;; Test disassembly for gfx950 kernel descriptor.

; RUN: rm -rf %t && split-file %s %t && cd %t

;--- 1.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=-xnack -filetype=obj -mcpu=gfx950 < 1.s > 1.o
; RUN: llvm-objdump --disassemble-symbols=kernel.kd 1.o | tail -n +7 | tee 1-disasm.s | FileCheck 1.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=-xnack -filetype=obj -mcpu=gfx950 < 1-disasm.s > 1-disasm.o
; FIxMe: cmp 1.o 1-disasm.o
; CHECK: .amdhsa_kernel kernel
; CHECK-NEXT:	.amdhsa_group_segment_fixed_size 163840
; CHECK-NEXT:	.amdhsa_private_segment_fixed_size 0
; CHECK-NEXT:	.amdhsa_kernarg_size 0
; CHECK-NEXT:	.amdhsa_accum_offset 4
; CHECK-NEXT:	.amdhsa_tg_split 0
; CHECK-NEXT:	.amdhsa_next_free_vgpr 8
; CHECK-NEXT:	.amdhsa_reserve_vcc 0
; CHECK-NEXT:	.amdhsa_reserve_xnack_mask 0
; CHECK-NEXT:	.amdhsa_next_free_sgpr 8
; CHECK-NEXT:	.amdhsa_float_round_mode_32 0
; CHECK-NEXT:	.amdhsa_float_round_mode_16_64 0
; CHECK-NEXT:	.amdhsa_float_denorm_mode_32 0
; CHECK-NEXT:	.amdhsa_float_denorm_mode_16_64 3
; CHECK-NEXT:	.amdhsa_dx10_clamp 1
; CHECK-NEXT:	.amdhsa_ieee_mode 1
; CHECK-NEXT:	.amdhsa_fp16_overflow 0
; CHECK-NEXT:	.amdhsa_enable_private_segment 0
; CHECK-NEXT:	.amdhsa_system_sgpr_workgroup_id_x 1
; CHECK-NEXT:	.amdhsa_system_sgpr_workgroup_id_y 0
; CHECK-NEXT:	.amdhsa_system_sgpr_workgroup_id_z 0
; CHECK-NEXT:	.amdhsa_system_sgpr_workgroup_info 0
; CHECK-NEXT:	.amdhsa_system_vgpr_workitem_id 0
; CHECK-NEXT:	.amdhsa_exception_fp_ieee_invalid_op 0
; CHECK-NEXT:	.amdhsa_exception_fp_denorm_src 0
; CHECK-NEXT:	.amdhsa_exception_fp_ieee_div_zero 0
; CHECK-NEXT:	.amdhsa_exception_fp_ieee_overflow 0
; CHECK-NEXT:	.amdhsa_exception_fp_ieee_underflow 0
; CHECK-NEXT:	.amdhsa_exception_fp_ieee_inexact 0
; CHECK-NEXT:	.amdhsa_exception_int_div_zero 0
; CHECK-NEXT:	.amdhsa_user_sgpr_dispatch_ptr 0
; CHECK-NEXT:	.amdhsa_user_sgpr_queue_ptr 0
; CHECK-NEXT:	.amdhsa_user_sgpr_kernarg_segment_ptr 0
; CHECK-NEXT:	.amdhsa_user_sgpr_dispatch_id 0
; CHECK-NEXT:	.amdhsa_user_sgpr_private_segment_size 0
; CHECK-NEXT:	.amdhsa_uses_dynamic_stack 0
; CHECK-NEXT:.end_amdhsa_kernel
.amdhsa_kernel kernel
  .amdhsa_group_segment_fixed_size 163840
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
.end_amdhsa_kernel
