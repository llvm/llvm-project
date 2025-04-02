;; Test disassembly for gfx1250 kernel descriptor.

; RUN: rm -rf %t && split-file %s %t && cd %t

;--- 1.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 < 1.s > 1.o
; RUN: llvm-objdump --disassemble-symbols=kernel.kd 1.o | tail -n +7 | tee 1-disasm.s | FileCheck 1.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 < 1-disasm.s > 1-disasm.o
; RUN: cmp 1.o 1-disasm.o
; CHECK: .amdhsa_kernel kernel
; CHECK-NEXT: .amdhsa_group_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_private_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_kernarg_size 0
; CHECK-NEXT: .amdhsa_inst_pref_size 0
; CHECK-NEXT: ; GLG_EN 0
; CHECK-NEXT: .amdhsa_named_barrier_count 0
; CHECK-NEXT: ; ENABLE_DYNAMIC_VGPR 0
; CHECK-NEXT: ; TCP_SPLIT 0
; CHECK-NEXT: ; ENABLE_DIDT_THROTTLE 0
; CHECK-NEXT: ; IMAGE_OP 0
; CHECK-NEXT: .amdhsa_next_free_vgpr 32
; CHECK-NEXT: .amdhsa_reserve_vcc 0
; CHECK-NEXT: .amdhsa_reserve_xnack_mask 0
; CHECK-NEXT: .amdhsa_next_free_sgpr 8
; CHECK-NEXT: .amdhsa_float_round_mode_32 0
; CHECK-NEXT: .amdhsa_float_round_mode_16_64 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_32 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_16_64 3
; CHECK-NEXT: .amdhsa_fp16_overflow 0
; CHECK-NEXT: ; FLAT_SCRATCH_IS_NV 0
; CHECK-NEXT: .amdhsa_memory_ordered 1
; CHECK-NEXT: .amdhsa_forward_progress 1
; CHECK-NEXT: .amdhsa_round_robin_scheduling 0
; CHECK-NEXT: .amdhsa_enable_private_segment 0
; CHECK-NEXT: .amdhsa_system_sgpr_workgroup_id_x 1
; CHECK-NEXT: .amdhsa_system_sgpr_workgroup_id_y 0
; CHECK-NEXT: .amdhsa_system_sgpr_workgroup_id_z 0
; CHECK-NEXT: .amdhsa_system_sgpr_workgroup_info 0
; CHECK-NEXT: .amdhsa_system_vgpr_workitem_id 0
; CHECK-NEXT: .amdhsa_exception_fp_ieee_invalid_op 0
; CHECK-NEXT: .amdhsa_exception_fp_denorm_src 0
; CHECK-NEXT: .amdhsa_exception_fp_ieee_div_zero 0
; CHECK-NEXT: .amdhsa_exception_fp_ieee_overflow 0
; CHECK-NEXT: .amdhsa_exception_fp_ieee_underflow 0
; CHECK-NEXT: .amdhsa_exception_fp_ieee_inexact 0
; CHECK-NEXT: .amdhsa_exception_int_div_zero 0
; CHECK-NEXT: .amdhsa_user_sgpr_dispatch_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_queue_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_dispatch_id 0
; CHECK-NEXT: .amdhsa_user_sgpr_private_segment_size 0
; CHECK-NEXT: .amdhsa_uses_cu_stores 1
; CHECK-NEXT: .amdhsa_wavefront_size32 1
; CHECK-NEXT: .amdhsa_uses_dynamic_stack 0
; CHECK-NEXT: .end_amdhsa_kernel
.amdhsa_kernel kernel
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
  .amdhsa_inst_pref_size 0
  .amdhsa_uses_cu_stores 1
.end_amdhsa_kernel

;--- 2.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 < 2.s > 2.o
; RUN: llvm-objdump --disassemble-symbols=kernel.kd 2.o | tail -n +7 | tee 2-disasm.s | FileCheck 2.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 < 2-disasm.s > 2-disasm.o
; RUN: cmp 2.o 2-disasm.o
; CHECK: .amdhsa_kernel kernel
; CHECK-NEXT: .amdhsa_group_segment_fixed_size 393216
; CHECK-NEXT: .amdhsa_private_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_kernarg_size 0
; CHECK-NEXT: .amdhsa_inst_pref_size 63
; CHECK-NEXT: ; GLG_EN 0
; CHECK-NEXT: .amdhsa_named_barrier_count 7
; CHECK-NEXT: ; ENABLE_DYNAMIC_VGPR 0
; CHECK-NEXT: ; TCP_SPLIT 0
; CHECK-NEXT: ; ENABLE_DIDT_THROTTLE 0
; CHECK-NEXT: ; IMAGE_OP 0
; CHECK-NEXT: .amdhsa_next_free_vgpr 32
; CHECK-NEXT: .amdhsa_reserve_vcc 0
; CHECK-NEXT: .amdhsa_reserve_xnack_mask 0
; CHECK-NEXT: .amdhsa_next_free_sgpr 8
; CHECK-NEXT: .amdhsa_float_round_mode_32 0
; CHECK-NEXT: .amdhsa_float_round_mode_16_64 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_32 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_16_64 3
; CHECK-NEXT: .amdhsa_fp16_overflow 0
; CHECK-NEXT: ; FLAT_SCRATCH_IS_NV 0
; CHECK-NEXT: .amdhsa_memory_ordered 1
; CHECK-NEXT: .amdhsa_forward_progress 1
; CHECK-NEXT: .amdhsa_round_robin_scheduling 0
; CHECK-NEXT: .amdhsa_enable_private_segment 0
; CHECK-NEXT: .amdhsa_system_sgpr_workgroup_id_x 1
; CHECK-NEXT: .amdhsa_system_sgpr_workgroup_id_y 0
; CHECK-NEXT: .amdhsa_system_sgpr_workgroup_id_z 0
; CHECK-NEXT: .amdhsa_system_sgpr_workgroup_info 0
; CHECK-NEXT: .amdhsa_system_vgpr_workitem_id 0
; CHECK-NEXT: .amdhsa_exception_fp_ieee_invalid_op 0
; CHECK-NEXT: .amdhsa_exception_fp_denorm_src 0
; CHECK-NEXT: .amdhsa_exception_fp_ieee_div_zero 0
; CHECK-NEXT: .amdhsa_exception_fp_ieee_overflow 0
; CHECK-NEXT: .amdhsa_exception_fp_ieee_underflow 0
; CHECK-NEXT: .amdhsa_exception_fp_ieee_inexact 0
; CHECK-NEXT: .amdhsa_exception_int_div_zero 0
; CHECK-NEXT: .amdhsa_user_sgpr_dispatch_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_queue_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_dispatch_id 0
; CHECK-NEXT: .amdhsa_user_sgpr_private_segment_size 0
; CHECK-NEXT: .amdhsa_uses_cu_stores 0
; CHECK-NEXT: .amdhsa_wavefront_size32 1
; CHECK-NEXT: .amdhsa_uses_dynamic_stack 0
; CHECK-NEXT: .end_amdhsa_kernel
.amdhsa_kernel kernel
  .amdhsa_group_segment_fixed_size 393216
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
  .amdhsa_named_barrier_count 7
  .amdhsa_uses_cu_stores 0
  .amdhsa_inst_pref_size 63
.end_amdhsa_kernel
