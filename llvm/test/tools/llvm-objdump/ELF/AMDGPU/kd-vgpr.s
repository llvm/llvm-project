;; Test disassembly for GRANULATED_WORKITEM_VGPR_COUNT in the kernel descriptor.

; RUN: rm -rf %t && split-file %s %t && cd %t

;--- 1.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=-xnack -filetype=obj -mcpu=gfx908 < 1.s > 1.o
; RUN: echo '.amdhsa_code_object_version 5' > 1-disasm.s
; RUN: llvm-objdump --disassemble-symbols=kernel.kd 1.o | tail -n +7 | tee -a 1-disasm.s | FileCheck 1.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=-xnack -filetype=obj -mcpu=gfx908 < 1-disasm.s > 1-disasm.o
; RUN: cmp 1.o 1-disasm.o
; CHECK: .amdhsa_kernel kernel
; CHECK-NEXT: .amdhsa_group_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_private_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_kernarg_size 0
; CHECK-NEXT: .amdhsa_next_free_vgpr 24
; CHECK-NEXT: .amdhsa_reserve_vcc 0
; CHECK-NEXT: .amdhsa_reserve_flat_scratch 0
; CHECK-NEXT: .amdhsa_reserve_xnack_mask 0
; CHECK-NEXT: .amdhsa_next_free_sgpr 8
; CHECK-NEXT: .amdhsa_float_round_mode_32 0
; CHECK-NEXT: .amdhsa_float_round_mode_16_64 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_32 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_16_64 3
; CHECK-NEXT: .amdhsa_dx10_clamp 1
; CHECK-NEXT: .amdhsa_ieee_mode 1
; CHECK-NEXT: .amdhsa_fp16_overflow 0
; CHECK-NEXT: .amdhsa_system_sgpr_private_segment_wavefront_offset 0
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
; CHECK-NEXT: .amdhsa_user_sgpr_private_segment_buffer 0
; CHECK-NEXT: .amdhsa_user_sgpr_dispatch_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_queue_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_dispatch_id 0
; CHECK-NEXT: .amdhsa_user_sgpr_flat_scratch_init 0
; CHECK-NEXT: .amdhsa_user_sgpr_private_segment_size 0
; CHECK-NEXT: .amdhsa_uses_dynamic_stack 0
; CHECK-NEXT: .end_amdhsa_kernel
.amdhsa_code_object_version 5
.amdhsa_kernel kernel
  .amdhsa_next_free_vgpr 23
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

;--- 2.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=-xnack -filetype=obj -mcpu=gfx908 < 2.s > 2.o
; RUN: echo '.amdhsa_code_object_version 5' > 2-disasm.s
; RUN: llvm-objdump --disassemble-symbols=kernel.kd 2.o | tail -n +7 | tee -a 2-disasm.s | FileCheck 2.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=-xnack -filetype=obj -mcpu=gfx908 < 2-disasm.s > 2-disasm.o
; RUN: cmp 2.o 2-disasm.o
; CHECK: .amdhsa_kernel kernel
; CHECK-NEXT: .amdhsa_group_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_private_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_kernarg_size 0
; CHECK-NEXT: .amdhsa_next_free_vgpr 16
; CHECK-NEXT: .amdhsa_reserve_vcc 0
; CHECK-NEXT: .amdhsa_reserve_flat_scratch 0
; CHECK-NEXT: .amdhsa_reserve_xnack_mask 0
; CHECK-NEXT: .amdhsa_next_free_sgpr 8
; CHECK-NEXT: .amdhsa_float_round_mode_32 0
; CHECK-NEXT: .amdhsa_float_round_mode_16_64 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_32 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_16_64 3
; CHECK-NEXT: .amdhsa_dx10_clamp 1
; CHECK-NEXT: .amdhsa_ieee_mode 1
; CHECK-NEXT: .amdhsa_fp16_overflow 0
; CHECK-NEXT: .amdhsa_system_sgpr_private_segment_wavefront_offset 0
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
; CHECK-NEXT: .amdhsa_user_sgpr_private_segment_buffer 0
; CHECK-NEXT: .amdhsa_user_sgpr_dispatch_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_queue_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_dispatch_id 0
; CHECK-NEXT: .amdhsa_user_sgpr_flat_scratch_init 0
; CHECK-NEXT: .amdhsa_user_sgpr_private_segment_size 0
; CHECK-NEXT: .amdhsa_uses_dynamic_stack 0
; CHECK-NEXT: .end_amdhsa_kernel
.amdhsa_code_object_version 5
.amdhsa_kernel kernel
  .amdhsa_next_free_vgpr 14
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

;--- 3.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=-xnack -filetype=obj -mcpu=gfx908 < 3.s > 3.o
; RUN: echo '.amdhsa_code_object_version 5' > 3-disasm.s
; RUN: llvm-objdump --disassemble-symbols=kernel.kd 3.o | tail -n +7 | tee -a 3-disasm.s | FileCheck 3.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=-xnack -filetype=obj -mcpu=gfx908 < 3-disasm.s > 3-disasm.o
; RUN: cmp 3.o 3-disasm.o
; CHECK: .amdhsa_kernel kernel
; CHECK-NEXT: .amdhsa_group_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_private_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_kernarg_size 0
; CHECK-NEXT: .amdhsa_next_free_vgpr 32
; CHECK-NEXT: .amdhsa_reserve_vcc 0
; CHECK-NEXT: .amdhsa_reserve_flat_scratch 0
; CHECK-NEXT: .amdhsa_reserve_xnack_mask 0
; CHECK-NEXT: .amdhsa_next_free_sgpr 8
; CHECK-NEXT: .amdhsa_float_round_mode_32 0
; CHECK-NEXT: .amdhsa_float_round_mode_16_64 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_32 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_16_64 3
; CHECK-NEXT: .amdhsa_dx10_clamp 1
; CHECK-NEXT: .amdhsa_ieee_mode 1
; CHECK-NEXT: .amdhsa_fp16_overflow 0
; CHECK-NEXT: .amdhsa_system_sgpr_private_segment_wavefront_offset 0
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
; CHECK-NEXT: .amdhsa_user_sgpr_private_segment_buffer 0
; CHECK-NEXT: .amdhsa_user_sgpr_dispatch_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_queue_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_dispatch_id 0
; CHECK-NEXT: .amdhsa_user_sgpr_flat_scratch_init 0
; CHECK-NEXT: .amdhsa_user_sgpr_private_segment_size 0
; CHECK-NEXT: .amdhsa_uses_dynamic_stack 0
; CHECK-NEXT: .end_amdhsa_kernel
.amdhsa_code_object_version 5
.amdhsa_kernel kernel
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel
