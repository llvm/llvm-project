;; Test disassembly for gfx11 kernel descriptor.

; RUN: rm -rf %t && split-file %s %t && cd %t

;--- 1.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=+wavefrontsize32,-wavefrontsize64 -filetype=obj -mcpu=gfx1100 < 1.s > 1.o
; RUN: echo '.amdhsa_code_object_version 5' > 1-disasm.s
; RUN: llvm-objdump --disassemble-symbols=kernel.kd 1.o | tail -n +7 | tee -a 1-disasm.s | FileCheck 1.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=+wavefrontsize32,-wavefrontsize64 -filetype=obj -mcpu=gfx1100 < 1-disasm.s > 1-disasm.o
; RUN: cmp 1.o 1-disasm.o
; CHECK: .amdhsa_kernel kernel
; CHECK-NEXT: .amdhsa_group_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_private_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_kernarg_size 0
; CHECK-NEXT: ; SHARED_VGPR_COUNT 0
; CHECK-NEXT: ; INST_PREF_SIZE 0
; CHECK-NEXT: ; TRAP_ON_START 0
; CHECK-NEXT: ; TRAP_ON_END 0
; CHECK-NEXT: ; IMAGE_OP 0
; CHECK-NEXT: .amdhsa_next_free_vgpr 32
; CHECK-NEXT: .amdhsa_reserve_vcc 0
; CHECK-NEXT: .amdhsa_reserve_xnack_mask 0
; CHECK-NEXT: .amdhsa_next_free_sgpr 8
; CHECK-NEXT: .amdhsa_float_round_mode_32 0
; CHECK-NEXT: .amdhsa_float_round_mode_16_64 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_32 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_16_64 3
; CHECK-NEXT: .amdhsa_dx10_clamp 1
; CHECK-NEXT: .amdhsa_ieee_mode 1
; CHECK-NEXT: .amdhsa_fp16_overflow 0
; CHECK-NEXT: .amdhsa_workgroup_processor_mode 1
; CHECK-NEXT: .amdhsa_memory_ordered 1
; CHECK-NEXT: .amdhsa_forward_progress 0
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
; CHECK-NEXT: .amdhsa_wavefront_size32 1
; CHECK-NEXT: .amdhsa_uses_dynamic_stack 0
; CHECK-NEXT: .end_amdhsa_kernel
.amdhsa_code_object_version 5
.amdhsa_kernel kernel
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

;--- 2.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=+wavefrontsize64,-wavefrontsize32 -filetype=obj -mcpu=gfx1100 < 2.s > 2.o
; RUN: echo '.amdhsa_code_object_version 5' > 2-disasm.s
; RUN: llvm-objdump --disassemble-symbols=kernel.kd 2.o | tail -n +7 | tee -a 2-disasm.s | FileCheck 2.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=+wavefrontsize64,-wavefrontsize32 -filetype=obj -mcpu=gfx1100 < 2-disasm.s > 2-disasm.o
; RUN: cmp 2.o 2-disasm.o
; CHECK: .amdhsa_kernel kernel
; CHECK-NEXT: .amdhsa_group_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_private_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_kernarg_size 0
; CHECK-NEXT: .amdhsa_shared_vgpr_count 0
; CHECK-NEXT: ; INST_PREF_SIZE 0
; CHECK-NEXT: ; TRAP_ON_START 0
; CHECK-NEXT: ; TRAP_ON_END 0
; CHECK-NEXT: ; IMAGE_OP 0
; CHECK-NEXT: .amdhsa_next_free_vgpr 32
; CHECK-NEXT: .amdhsa_reserve_vcc 0
; CHECK-NEXT: .amdhsa_reserve_xnack_mask 0
; CHECK-NEXT: .amdhsa_next_free_sgpr 8
; CHECK-NEXT: .amdhsa_float_round_mode_32 0
; CHECK-NEXT: .amdhsa_float_round_mode_16_64 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_32 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_16_64 3
; CHECK-NEXT: .amdhsa_dx10_clamp 1
; CHECK-NEXT: .amdhsa_ieee_mode 1
; CHECK-NEXT: .amdhsa_fp16_overflow 0
; CHECK-NEXT: .amdhsa_workgroup_processor_mode 1
; CHECK-NEXT: .amdhsa_memory_ordered 1
; CHECK-NEXT: .amdhsa_forward_progress 0
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
; CHECK-NEXT: .amdhsa_wavefront_size32 0
; CHECK-NEXT: .amdhsa_uses_dynamic_stack 0
; CHECK-NEXT: .end_amdhsa_kernel
.amdhsa_code_object_version 5
.amdhsa_kernel kernel
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
  .amdhsa_shared_vgpr_count 0
.end_amdhsa_kernel

;--- 3.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=+wavefrontsize64,-wavefrontsize32 -filetype=obj -mcpu=gfx1100 < 3.s > 3.o
; RUN: echo '.amdhsa_code_object_version 5' > 3-disasm.s
; RUN: llvm-objdump --disassemble-symbols=kernel.kd 3.o | tail -n +7 | tee -a 3-disasm.s | FileCheck 3.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=+wavefrontsize64,-wavefrontsize32 -filetype=obj -mcpu=gfx1100 < 3-disasm.s > 3-disasm.o
; RUN: cmp 3.o 3-disasm.o
; CHECK: .amdhsa_kernel kernel
; CHECK-NEXT: .amdhsa_group_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_private_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_kernarg_size 0
; CHECK-NEXT: .amdhsa_shared_vgpr_count 1
; CHECK-NEXT: ; INST_PREF_SIZE 0
; CHECK-NEXT: ; TRAP_ON_START 0
; CHECK-NEXT: ; TRAP_ON_END 0
; CHECK-NEXT: ; IMAGE_OP 0
; CHECK-NEXT: .amdhsa_next_free_vgpr 32
; CHECK-NEXT: .amdhsa_reserve_vcc 0
; CHECK-NEXT: .amdhsa_reserve_xnack_mask 0
; CHECK-NEXT: .amdhsa_next_free_sgpr 8
; CHECK-NEXT: .amdhsa_float_round_mode_32 0
; CHECK-NEXT: .amdhsa_float_round_mode_16_64 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_32 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_16_64 3
; CHECK-NEXT: .amdhsa_dx10_clamp 1
; CHECK-NEXT: .amdhsa_ieee_mode 1
; CHECK-NEXT: .amdhsa_fp16_overflow 0
; CHECK-NEXT: .amdhsa_workgroup_processor_mode 1
; CHECK-NEXT: .amdhsa_memory_ordered 1
; CHECK-NEXT: .amdhsa_forward_progress 0
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
; CHECK-NEXT: .amdhsa_wavefront_size32 0
; CHECK-NEXT: .amdhsa_uses_dynamic_stack 0
; CHECK-NEXT: .end_amdhsa_kernel
.amdhsa_code_object_version 5
.amdhsa_kernel kernel
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
  .amdhsa_shared_vgpr_count 1
.end_amdhsa_kernel

;--- 4.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=+wavefrontsize64,-wavefrontsize32 -filetype=obj -mcpu=gfx1100 < 4.s > 4.o
; RUN: echo '.amdhsa_code_object_version 5' > 4-disasm.s
; RUN: llvm-objdump --disassemble-symbols=kernel.kd 4.o | tail -n +7 | tee -a 4-disasm.s | FileCheck 4.s
; RUN: llvm-mc --triple=amdgcn-amd-amdhsa -mattr=+wavefrontsize64,-wavefrontsize32 -filetype=obj -mcpu=gfx1100 < 4-disasm.s > 4-disasm.o
; RUN: cmp 4.o 4-disasm.o
; CHECK: .amdhsa_kernel kernel
; CHECK-NEXT: .amdhsa_group_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_private_segment_fixed_size 0
; CHECK-NEXT: .amdhsa_kernarg_size 0
; CHECK-NEXT: .amdhsa_shared_vgpr_count 1
; CHECK-NEXT: ; INST_PREF_SIZE 0
; CHECK-NEXT: ; TRAP_ON_START 0
; CHECK-NEXT: ; TRAP_ON_END 0
; CHECK-NEXT: ; IMAGE_OP 0
; CHECK-NEXT: .amdhsa_next_free_vgpr 32
; CHECK-NEXT: .amdhsa_reserve_vcc 0
; CHECK-NEXT: .amdhsa_reserve_xnack_mask 0
; CHECK-NEXT: .amdhsa_next_free_sgpr 8
; CHECK-NEXT: .amdhsa_float_round_mode_32 0
; CHECK-NEXT: .amdhsa_float_round_mode_16_64 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_32 0
; CHECK-NEXT: .amdhsa_float_denorm_mode_16_64 3
; CHECK-NEXT: .amdhsa_dx10_clamp 1
; CHECK-NEXT: .amdhsa_ieee_mode 1
; CHECK-NEXT: .amdhsa_fp16_overflow 0
; CHECK-NEXT: .amdhsa_workgroup_processor_mode 1
; CHECK-NEXT: .amdhsa_memory_ordered 1
; CHECK-NEXT: .amdhsa_forward_progress 0
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
; CHECK-NEXT: .amdhsa_wavefront_size32 0
; CHECK-NEXT: .amdhsa_uses_dynamic_stack 0
; CHECK-NEXT: .end_amdhsa_kernel
.amdhsa_code_object_version 5
.amdhsa_kernel kernel
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
  .amdhsa_shared_vgpr_count 1
  .amdhsa_wavefront_size32 0
.end_amdhsa_kernel
