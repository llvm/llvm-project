; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji < %s | FileCheck -check-prefix=VI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefix=GFX9 %s

; Make sure the stack is never realigned for entry functions.

define amdgpu_kernel void @max_alignment_128() #0 {
; VI-LABEL: max_alignment_128:
; VI:       ; %bb.0:
; VI-NEXT:    s_add_u32 s0, s0, s17
; VI-NEXT:    s_addc_u32 s1, s1, 0
; VI-NEXT:    v_mov_b32_e32 v0, 3
; VI-NEXT:    buffer_store_byte v0, off, s[0:3], 0
; VI-NEXT:    s_waitcnt vmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v0, 9
; VI-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:128
; VI-NEXT:    s_waitcnt vmcnt(0)
; VI-NEXT:    s_endpgm
; VI-NEXT:    .section .rodata,"a"
; VI-NEXT:    .p2align 6
; VI-NEXT:    .amdhsa_kernel max_alignment_128
; VI-NEXT:     .amdhsa_group_segment_fixed_size 0
; VI-NEXT:     .amdhsa_private_segment_fixed_size max_alignment_128.private_seg_size
; VI-NEXT:     .amdhsa_kernarg_size 56
; VI-NEXT:     .amdhsa_user_sgpr_count 14
; VI-NEXT:     .amdhsa_user_sgpr_private_segment_buffer 1
; VI-NEXT:     .amdhsa_user_sgpr_dispatch_ptr 1
; VI-NEXT:     .amdhsa_user_sgpr_queue_ptr 1
; VI-NEXT:     .amdhsa_user_sgpr_kernarg_segment_ptr 1
; VI-NEXT:     .amdhsa_user_sgpr_dispatch_id 1
; VI-NEXT:     .amdhsa_user_sgpr_flat_scratch_init 1
; VI-NEXT:     .amdhsa_user_sgpr_private_segment_size 0
; VI-NEXT:     .amdhsa_system_sgpr_private_segment_wavefront_offset (((((alignto(max_alignment_128.private_seg_size*64, 1024))/1024)>0)||(max_alignment_128.has_dyn_sized_stack|max_alignment_128.has_recursion))|5020)&1
; VI-NEXT:     .amdhsa_system_sgpr_workgroup_id_x 1
; VI-NEXT:     .amdhsa_system_sgpr_workgroup_id_y 1
; VI-NEXT:     .amdhsa_system_sgpr_workgroup_id_z 1
; VI-NEXT:     .amdhsa_system_sgpr_workgroup_info 0
; VI-NEXT:     .amdhsa_system_vgpr_workitem_id 2
; VI-NEXT:     .amdhsa_next_free_vgpr max(totalnumvgprs(max_alignment_128.num_agpr, max_alignment_128.num_vgpr), 1, 0)
; VI-NEXT:     .amdhsa_next_free_sgpr (max(max_alignment_128.num_sgpr+(extrasgprs(max_alignment_128.uses_vcc, max_alignment_128.uses_flat_scratch, 0)), 1, 0))-(extrasgprs(max_alignment_128.uses_vcc, max_alignment_128.uses_flat_scratch, 0))
; VI-NEXT:     .amdhsa_reserve_vcc max_alignment_128.uses_vcc
; VI-NEXT:     .amdhsa_reserve_flat_scratch max_alignment_128.uses_flat_scratch
; VI-NEXT:     .amdhsa_float_round_mode_32 0
; VI-NEXT:     .amdhsa_float_round_mode_16_64 0
; VI-NEXT:     .amdhsa_float_denorm_mode_32 3
; VI-NEXT:     .amdhsa_float_denorm_mode_16_64 3
; VI-NEXT:     .amdhsa_dx10_clamp 1
; VI-NEXT:     .amdhsa_ieee_mode 1
; VI-NEXT:     .amdhsa_exception_fp_ieee_invalid_op 0
; VI-NEXT:     .amdhsa_exception_fp_denorm_src 0
; VI-NEXT:     .amdhsa_exception_fp_ieee_div_zero 0
; VI-NEXT:     .amdhsa_exception_fp_ieee_overflow 0
; VI-NEXT:     .amdhsa_exception_fp_ieee_underflow 0
; VI-NEXT:     .amdhsa_exception_fp_ieee_inexact 0
; VI-NEXT:     .amdhsa_exception_int_div_zero 0
; VI-NEXT:    .end_amdhsa_kernel
; VI-NEXT:    .text
; VI:         .set max_alignment_128.num_vgpr, 1
; VI-NEXT:    .set max_alignment_128.num_agpr, 0
; VI-NEXT:    .set max_alignment_128.num_sgpr, 18
; VI-NEXT:    .set max_alignment_128.private_seg_size, 256
; VI-NEXT:    .set max_alignment_128.uses_vcc, 0
; VI-NEXT:    .set max_alignment_128.uses_flat_scratch, 0
; VI-NEXT:    .set max_alignment_128.has_dyn_sized_stack, 0
; VI-NEXT:    .set max_alignment_128.has_recursion, 0
; VI-NEXT:    .set max_alignment_128.has_indirect_call, 0
;
; GFX9-LABEL: max_alignment_128:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_add_u32 s0, s0, s17
; GFX9-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-NEXT:    v_mov_b32_e32 v0, 3
; GFX9-NEXT:    buffer_store_byte v0, off, s[0:3], 0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v0, 9
; GFX9-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:128
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    s_endpgm
; GFX9-NEXT:    .section .rodata,"a"
; GFX9-NEXT:    .p2align 6
; GFX9-NEXT:    .amdhsa_kernel max_alignment_128
; GFX9-NEXT:     .amdhsa_group_segment_fixed_size 0
; GFX9-NEXT:     .amdhsa_private_segment_fixed_size max_alignment_128.private_seg_size
; GFX9-NEXT:     .amdhsa_kernarg_size 56
; GFX9-NEXT:     .amdhsa_user_sgpr_count 14
; GFX9-NEXT:     .amdhsa_user_sgpr_private_segment_buffer 1
; GFX9-NEXT:     .amdhsa_user_sgpr_dispatch_ptr 1
; GFX9-NEXT:     .amdhsa_user_sgpr_queue_ptr 1
; GFX9-NEXT:     .amdhsa_user_sgpr_kernarg_segment_ptr 1
; GFX9-NEXT:     .amdhsa_user_sgpr_dispatch_id 1
; GFX9-NEXT:     .amdhsa_user_sgpr_flat_scratch_init 1
; GFX9-NEXT:     .amdhsa_user_sgpr_private_segment_size 0
; GFX9-NEXT:     .amdhsa_system_sgpr_private_segment_wavefront_offset (((((alignto(max_alignment_128.private_seg_size*64, 1024))/1024)>0)||(max_alignment_128.has_dyn_sized_stack|max_alignment_128.has_recursion))|5020)&1
; GFX9-NEXT:     .amdhsa_system_sgpr_workgroup_id_x 1
; GFX9-NEXT:     .amdhsa_system_sgpr_workgroup_id_y 1
; GFX9-NEXT:     .amdhsa_system_sgpr_workgroup_id_z 1
; GFX9-NEXT:     .amdhsa_system_sgpr_workgroup_info 0
; GFX9-NEXT:     .amdhsa_system_vgpr_workitem_id 2
; GFX9-NEXT:     .amdhsa_next_free_vgpr max(totalnumvgprs(max_alignment_128.num_agpr, max_alignment_128.num_vgpr), 1, 0)
; GFX9-NEXT:     .amdhsa_next_free_sgpr (max(max_alignment_128.num_sgpr+(extrasgprs(max_alignment_128.uses_vcc, max_alignment_128.uses_flat_scratch, 1)), 1, 0))-(extrasgprs(max_alignment_128.uses_vcc, max_alignment_128.uses_flat_scratch, 1))
; GFX9-NEXT:     .amdhsa_reserve_vcc max_alignment_128.uses_vcc
; GFX9-NEXT:     .amdhsa_reserve_flat_scratch max_alignment_128.uses_flat_scratch
; GFX9-NEXT:     .amdhsa_reserve_xnack_mask 1
; GFX9-NEXT:     .amdhsa_float_round_mode_32 0
; GFX9-NEXT:     .amdhsa_float_round_mode_16_64 0
; GFX9-NEXT:     .amdhsa_float_denorm_mode_32 3
; GFX9-NEXT:     .amdhsa_float_denorm_mode_16_64 3
; GFX9-NEXT:     .amdhsa_dx10_clamp 1
; GFX9-NEXT:     .amdhsa_ieee_mode 1
; GFX9-NEXT:     .amdhsa_fp16_overflow 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_invalid_op 0
; GFX9-NEXT:     .amdhsa_exception_fp_denorm_src 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_div_zero 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_overflow 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_underflow 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_inexact 0
; GFX9-NEXT:     .amdhsa_exception_int_div_zero 0
; GFX9-NEXT:    .end_amdhsa_kernel
; GFX9-NEXT:    .text
; GFX9:         .set max_alignment_128.num_vgpr, 1
; GFX9-NEXT:    .set max_alignment_128.num_agpr, 0
; GFX9-NEXT:    .set max_alignment_128.num_sgpr, 18
; GFX9-NEXT:    .set max_alignment_128.private_seg_size, 256
; GFX9-NEXT:    .set max_alignment_128.uses_vcc, 0
; GFX9-NEXT:    .set max_alignment_128.uses_flat_scratch, 0
; GFX9-NEXT:    .set max_alignment_128.has_dyn_sized_stack, 0
; GFX9-NEXT:    .set max_alignment_128.has_recursion, 0
; GFX9-NEXT:    .set max_alignment_128.has_indirect_call, 0
  %clutter = alloca i8, addrspace(5) ; Force non-zero offset for next alloca
  store volatile i8 3, ptr addrspace(5) %clutter
  %alloca.align = alloca i32, align 128, addrspace(5)
  store volatile i32 9, ptr addrspace(5) %alloca.align, align 128
  ret void
}

define amdgpu_kernel void @stackrealign_attr() #1 {
; VI-LABEL: stackrealign_attr:
; VI:       ; %bb.0:
; VI-NEXT:    s_add_u32 s0, s0, s17
; VI-NEXT:    s_addc_u32 s1, s1, 0
; VI-NEXT:    v_mov_b32_e32 v0, 3
; VI-NEXT:    buffer_store_byte v0, off, s[0:3], 0
; VI-NEXT:    s_waitcnt vmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v0, 9
; VI-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; VI-NEXT:    s_waitcnt vmcnt(0)
; VI-NEXT:    s_endpgm
; VI-NEXT:    .section .rodata,"a"
; VI-NEXT:    .p2align 6
; VI-NEXT:    .amdhsa_kernel stackrealign_attr
; VI-NEXT:     .amdhsa_group_segment_fixed_size 0
; VI-NEXT:     .amdhsa_private_segment_fixed_size stackrealign_attr.private_seg_size
; VI-NEXT:     .amdhsa_kernarg_size 56
; VI-NEXT:     .amdhsa_user_sgpr_count 14
; VI-NEXT:     .amdhsa_user_sgpr_private_segment_buffer 1
; VI-NEXT:     .amdhsa_user_sgpr_dispatch_ptr 1
; VI-NEXT:     .amdhsa_user_sgpr_queue_ptr 1
; VI-NEXT:     .amdhsa_user_sgpr_kernarg_segment_ptr 1
; VI-NEXT:     .amdhsa_user_sgpr_dispatch_id 1
; VI-NEXT:     .amdhsa_user_sgpr_flat_scratch_init 1
; VI-NEXT:     .amdhsa_user_sgpr_private_segment_size 0
; VI-NEXT:     .amdhsa_system_sgpr_private_segment_wavefront_offset (((((alignto(stackrealign_attr.private_seg_size*64, 1024))/1024)>0)||(stackrealign_attr.has_dyn_sized_stack|stackrealign_attr.has_recursion))|5020)&1
; VI-NEXT:     .amdhsa_system_sgpr_workgroup_id_x 1
; VI-NEXT:     .amdhsa_system_sgpr_workgroup_id_y 1
; VI-NEXT:     .amdhsa_system_sgpr_workgroup_id_z 1
; VI-NEXT:     .amdhsa_system_sgpr_workgroup_info 0
; VI-NEXT:     .amdhsa_system_vgpr_workitem_id 2
; VI-NEXT:     .amdhsa_next_free_vgpr max(totalnumvgprs(stackrealign_attr.num_agpr, stackrealign_attr.num_vgpr), 1, 0)
; VI-NEXT:     .amdhsa_next_free_sgpr (max(stackrealign_attr.num_sgpr+(extrasgprs(stackrealign_attr.uses_vcc, stackrealign_attr.uses_flat_scratch, 0)), 1, 0))-(extrasgprs(stackrealign_attr.uses_vcc, stackrealign_attr.uses_flat_scratch, 0))
; VI-NEXT:     .amdhsa_reserve_vcc stackrealign_attr.uses_vcc
; VI-NEXT:     .amdhsa_reserve_flat_scratch stackrealign_attr.uses_flat_scratch
; VI-NEXT:     .amdhsa_float_round_mode_32 0
; VI-NEXT:     .amdhsa_float_round_mode_16_64 0
; VI-NEXT:     .amdhsa_float_denorm_mode_32 3
; VI-NEXT:     .amdhsa_float_denorm_mode_16_64 3
; VI-NEXT:     .amdhsa_dx10_clamp 1
; VI-NEXT:     .amdhsa_ieee_mode 1
; VI-NEXT:     .amdhsa_exception_fp_ieee_invalid_op 0
; VI-NEXT:     .amdhsa_exception_fp_denorm_src 0
; VI-NEXT:     .amdhsa_exception_fp_ieee_div_zero 0
; VI-NEXT:     .amdhsa_exception_fp_ieee_overflow 0
; VI-NEXT:     .amdhsa_exception_fp_ieee_underflow 0
; VI-NEXT:     .amdhsa_exception_fp_ieee_inexact 0
; VI-NEXT:     .amdhsa_exception_int_div_zero 0
; VI-NEXT:    .end_amdhsa_kernel
; VI-NEXT:    .text
; VI:         .set stackrealign_attr.num_vgpr, 1
; VI-NEXT:    .set stackrealign_attr.num_agpr, 0
; VI-NEXT:    .set stackrealign_attr.num_sgpr, 18
; VI-NEXT:    .set stackrealign_attr.private_seg_size, 12
; VI-NEXT:    .set stackrealign_attr.uses_vcc, 0
; VI-NEXT:    .set stackrealign_attr.uses_flat_scratch, 0
; VI-NEXT:    .set stackrealign_attr.has_dyn_sized_stack, 0
; VI-NEXT:    .set stackrealign_attr.has_recursion, 0
; VI-NEXT:    .set stackrealign_attr.has_indirect_call, 0
;
; GFX9-LABEL: stackrealign_attr:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_add_u32 s0, s0, s17
; GFX9-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-NEXT:    v_mov_b32_e32 v0, 3
; GFX9-NEXT:    buffer_store_byte v0, off, s[0:3], 0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v0, 9
; GFX9-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    s_endpgm
; GFX9-NEXT:    .section .rodata,"a"
; GFX9-NEXT:    .p2align 6
; GFX9-NEXT:    .amdhsa_kernel stackrealign_attr
; GFX9-NEXT:     .amdhsa_group_segment_fixed_size 0
; GFX9-NEXT:     .amdhsa_private_segment_fixed_size stackrealign_attr.private_seg_size
; GFX9-NEXT:     .amdhsa_kernarg_size 56
; GFX9-NEXT:     .amdhsa_user_sgpr_count 14
; GFX9-NEXT:     .amdhsa_user_sgpr_private_segment_buffer 1
; GFX9-NEXT:     .amdhsa_user_sgpr_dispatch_ptr 1
; GFX9-NEXT:     .amdhsa_user_sgpr_queue_ptr 1
; GFX9-NEXT:     .amdhsa_user_sgpr_kernarg_segment_ptr 1
; GFX9-NEXT:     .amdhsa_user_sgpr_dispatch_id 1
; GFX9-NEXT:     .amdhsa_user_sgpr_flat_scratch_init 1
; GFX9-NEXT:     .amdhsa_user_sgpr_private_segment_size 0
; GFX9-NEXT:     .amdhsa_system_sgpr_private_segment_wavefront_offset (((((alignto(stackrealign_attr.private_seg_size*64, 1024))/1024)>0)||(stackrealign_attr.has_dyn_sized_stack|stackrealign_attr.has_recursion))|5020)&1
; GFX9-NEXT:     .amdhsa_system_sgpr_workgroup_id_x 1
; GFX9-NEXT:     .amdhsa_system_sgpr_workgroup_id_y 1
; GFX9-NEXT:     .amdhsa_system_sgpr_workgroup_id_z 1
; GFX9-NEXT:     .amdhsa_system_sgpr_workgroup_info 0
; GFX9-NEXT:     .amdhsa_system_vgpr_workitem_id 2
; GFX9-NEXT:     .amdhsa_next_free_vgpr max(totalnumvgprs(stackrealign_attr.num_agpr, stackrealign_attr.num_vgpr), 1, 0)
; GFX9-NEXT:     .amdhsa_next_free_sgpr (max(stackrealign_attr.num_sgpr+(extrasgprs(stackrealign_attr.uses_vcc, stackrealign_attr.uses_flat_scratch, 1)), 1, 0))-(extrasgprs(stackrealign_attr.uses_vcc, stackrealign_attr.uses_flat_scratch, 1))
; GFX9-NEXT:     .amdhsa_reserve_vcc stackrealign_attr.uses_vcc
; GFX9-NEXT:     .amdhsa_reserve_flat_scratch stackrealign_attr.uses_flat_scratch
; GFX9-NEXT:     .amdhsa_reserve_xnack_mask 1
; GFX9-NEXT:     .amdhsa_float_round_mode_32 0
; GFX9-NEXT:     .amdhsa_float_round_mode_16_64 0
; GFX9-NEXT:     .amdhsa_float_denorm_mode_32 3
; GFX9-NEXT:     .amdhsa_float_denorm_mode_16_64 3
; GFX9-NEXT:     .amdhsa_dx10_clamp 1
; GFX9-NEXT:     .amdhsa_ieee_mode 1
; GFX9-NEXT:     .amdhsa_fp16_overflow 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_invalid_op 0
; GFX9-NEXT:     .amdhsa_exception_fp_denorm_src 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_div_zero 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_overflow 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_underflow 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_inexact 0
; GFX9-NEXT:     .amdhsa_exception_int_div_zero 0
; GFX9-NEXT:    .end_amdhsa_kernel
; GFX9-NEXT:    .text
; GFX9:         .set stackrealign_attr.num_vgpr, 1
; GFX9-NEXT:    .set stackrealign_attr.num_agpr, 0
; GFX9-NEXT:    .set stackrealign_attr.num_sgpr, 18
; GFX9-NEXT:    .set stackrealign_attr.private_seg_size, 12
; GFX9-NEXT:    .set stackrealign_attr.uses_vcc, 0
; GFX9-NEXT:    .set stackrealign_attr.uses_flat_scratch, 0
; GFX9-NEXT:    .set stackrealign_attr.has_dyn_sized_stack, 0
; GFX9-NEXT:    .set stackrealign_attr.has_recursion, 0
; GFX9-NEXT:    .set stackrealign_attr.has_indirect_call, 0
  %clutter = alloca i8, addrspace(5) ; Force non-zero offset for next alloca
  store volatile i8 3, ptr addrspace(5) %clutter
  %alloca.align = alloca i32, align 4, addrspace(5)
  store volatile i32 9, ptr addrspace(5) %alloca.align, align 4
  ret void
}

define amdgpu_kernel void @alignstack_attr() #2 {
; VI-LABEL: alignstack_attr:
; VI:       ; %bb.0:
; VI-NEXT:    s_add_u32 s0, s0, s17
; VI-NEXT:    s_addc_u32 s1, s1, 0
; VI-NEXT:    v_mov_b32_e32 v0, 3
; VI-NEXT:    buffer_store_byte v0, off, s[0:3], 0
; VI-NEXT:    s_waitcnt vmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v0, 9
; VI-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; VI-NEXT:    s_waitcnt vmcnt(0)
; VI-NEXT:    s_endpgm
; VI-NEXT:    .section .rodata,"a"
; VI-NEXT:    .p2align 6
; VI-NEXT:    .amdhsa_kernel alignstack_attr
; VI-NEXT:     .amdhsa_group_segment_fixed_size 0
; VI-NEXT:     .amdhsa_private_segment_fixed_size alignstack_attr.private_seg_size
; VI-NEXT:     .amdhsa_kernarg_size 56
; VI-NEXT:     .amdhsa_user_sgpr_count 14
; VI-NEXT:     .amdhsa_user_sgpr_private_segment_buffer 1
; VI-NEXT:     .amdhsa_user_sgpr_dispatch_ptr 1
; VI-NEXT:     .amdhsa_user_sgpr_queue_ptr 1
; VI-NEXT:     .amdhsa_user_sgpr_kernarg_segment_ptr 1
; VI-NEXT:     .amdhsa_user_sgpr_dispatch_id 1
; VI-NEXT:     .amdhsa_user_sgpr_flat_scratch_init 1
; VI-NEXT:     .amdhsa_user_sgpr_private_segment_size 0
; VI-NEXT:     .amdhsa_system_sgpr_private_segment_wavefront_offset (((((alignto(alignstack_attr.private_seg_size*64, 1024))/1024)>0)||(alignstack_attr.has_dyn_sized_stack|alignstack_attr.has_recursion))|5020)&1
; VI-NEXT:     .amdhsa_system_sgpr_workgroup_id_x 1
; VI-NEXT:     .amdhsa_system_sgpr_workgroup_id_y 1
; VI-NEXT:     .amdhsa_system_sgpr_workgroup_id_z 1
; VI-NEXT:     .amdhsa_system_sgpr_workgroup_info 0
; VI-NEXT:     .amdhsa_system_vgpr_workitem_id 2
; VI-NEXT:     .amdhsa_next_free_vgpr max(totalnumvgprs(alignstack_attr.num_agpr, alignstack_attr.num_vgpr), 1, 0)
; VI-NEXT:     .amdhsa_next_free_sgpr (max(alignstack_attr.num_sgpr+(extrasgprs(alignstack_attr.uses_vcc, alignstack_attr.uses_flat_scratch, 0)), 1, 0))-(extrasgprs(alignstack_attr.uses_vcc, alignstack_attr.uses_flat_scratch, 0))
; VI-NEXT:     .amdhsa_reserve_vcc alignstack_attr.uses_vcc
; VI-NEXT:     .amdhsa_reserve_flat_scratch alignstack_attr.uses_flat_scratch
; VI-NEXT:     .amdhsa_float_round_mode_32 0
; VI-NEXT:     .amdhsa_float_round_mode_16_64 0
; VI-NEXT:     .amdhsa_float_denorm_mode_32 3
; VI-NEXT:     .amdhsa_float_denorm_mode_16_64 3
; VI-NEXT:     .amdhsa_dx10_clamp 1
; VI-NEXT:     .amdhsa_ieee_mode 1
; VI-NEXT:     .amdhsa_exception_fp_ieee_invalid_op 0
; VI-NEXT:     .amdhsa_exception_fp_denorm_src 0
; VI-NEXT:     .amdhsa_exception_fp_ieee_div_zero 0
; VI-NEXT:     .amdhsa_exception_fp_ieee_overflow 0
; VI-NEXT:     .amdhsa_exception_fp_ieee_underflow 0
; VI-NEXT:     .amdhsa_exception_fp_ieee_inexact 0
; VI-NEXT:     .amdhsa_exception_int_div_zero 0
; VI-NEXT:    .end_amdhsa_kernel
; VI-NEXT:    .text
; VI:         .set alignstack_attr.num_vgpr, 1
; VI-NEXT:    .set alignstack_attr.num_agpr, 0
; VI-NEXT:    .set alignstack_attr.num_sgpr, 18
; VI-NEXT:    .set alignstack_attr.private_seg_size, 128
; VI-NEXT:    .set alignstack_attr.uses_vcc, 0
; VI-NEXT:    .set alignstack_attr.uses_flat_scratch, 0
; VI-NEXT:    .set alignstack_attr.has_dyn_sized_stack, 0
; VI-NEXT:    .set alignstack_attr.has_recursion, 0
; VI-NEXT:    .set alignstack_attr.has_indirect_call, 0
;
; GFX9-LABEL: alignstack_attr:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_add_u32 s0, s0, s17
; GFX9-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-NEXT:    v_mov_b32_e32 v0, 3
; GFX9-NEXT:    buffer_store_byte v0, off, s[0:3], 0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v0, 9
; GFX9-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    s_endpgm
; GFX9-NEXT:    .section .rodata,"a"
; GFX9-NEXT:    .p2align 6
; GFX9-NEXT:    .amdhsa_kernel alignstack_attr
; GFX9-NEXT:     .amdhsa_group_segment_fixed_size 0
; GFX9-NEXT:     .amdhsa_private_segment_fixed_size alignstack_attr.private_seg_size
; GFX9-NEXT:     .amdhsa_kernarg_size 56
; GFX9-NEXT:     .amdhsa_user_sgpr_count 14
; GFX9-NEXT:     .amdhsa_user_sgpr_private_segment_buffer 1
; GFX9-NEXT:     .amdhsa_user_sgpr_dispatch_ptr 1
; GFX9-NEXT:     .amdhsa_user_sgpr_queue_ptr 1
; GFX9-NEXT:     .amdhsa_user_sgpr_kernarg_segment_ptr 1
; GFX9-NEXT:     .amdhsa_user_sgpr_dispatch_id 1
; GFX9-NEXT:     .amdhsa_user_sgpr_flat_scratch_init 1
; GFX9-NEXT:     .amdhsa_user_sgpr_private_segment_size 0
; GFX9-NEXT:     .amdhsa_system_sgpr_private_segment_wavefront_offset (((((alignto(alignstack_attr.private_seg_size*64, 1024))/1024)>0)||(alignstack_attr.has_dyn_sized_stack|alignstack_attr.has_recursion))|5020)&1
; GFX9-NEXT:     .amdhsa_system_sgpr_workgroup_id_x 1
; GFX9-NEXT:     .amdhsa_system_sgpr_workgroup_id_y 1
; GFX9-NEXT:     .amdhsa_system_sgpr_workgroup_id_z 1
; GFX9-NEXT:     .amdhsa_system_sgpr_workgroup_info 0
; GFX9-NEXT:     .amdhsa_system_vgpr_workitem_id 2
; GFX9-NEXT:     .amdhsa_next_free_vgpr max(totalnumvgprs(alignstack_attr.num_agpr, alignstack_attr.num_vgpr), 1, 0)
; GFX9-NEXT:     .amdhsa_next_free_sgpr (max(alignstack_attr.num_sgpr+(extrasgprs(alignstack_attr.uses_vcc, alignstack_attr.uses_flat_scratch, 1)), 1, 0))-(extrasgprs(alignstack_attr.uses_vcc, alignstack_attr.uses_flat_scratch, 1))
; GFX9-NEXT:     .amdhsa_reserve_vcc alignstack_attr.uses_vcc
; GFX9-NEXT:     .amdhsa_reserve_flat_scratch alignstack_attr.uses_flat_scratch
; GFX9-NEXT:     .amdhsa_reserve_xnack_mask 1
; GFX9-NEXT:     .amdhsa_float_round_mode_32 0
; GFX9-NEXT:     .amdhsa_float_round_mode_16_64 0
; GFX9-NEXT:     .amdhsa_float_denorm_mode_32 3
; GFX9-NEXT:     .amdhsa_float_denorm_mode_16_64 3
; GFX9-NEXT:     .amdhsa_dx10_clamp 1
; GFX9-NEXT:     .amdhsa_ieee_mode 1
; GFX9-NEXT:     .amdhsa_fp16_overflow 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_invalid_op 0
; GFX9-NEXT:     .amdhsa_exception_fp_denorm_src 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_div_zero 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_overflow 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_underflow 0
; GFX9-NEXT:     .amdhsa_exception_fp_ieee_inexact 0
; GFX9-NEXT:     .amdhsa_exception_int_div_zero 0
; GFX9-NEXT:    .end_amdhsa_kernel
; GFX9-NEXT:    .text
; GFX9:         .set alignstack_attr.num_vgpr, 1
; GFX9-NEXT:    .set alignstack_attr.num_agpr, 0
; GFX9-NEXT:    .set alignstack_attr.num_sgpr, 18
; GFX9-NEXT:    .set alignstack_attr.private_seg_size, 128
; GFX9-NEXT:    .set alignstack_attr.uses_vcc, 0
; GFX9-NEXT:    .set alignstack_attr.uses_flat_scratch, 0
; GFX9-NEXT:    .set alignstack_attr.has_dyn_sized_stack, 0
; GFX9-NEXT:    .set alignstack_attr.has_recursion, 0
; GFX9-NEXT:    .set alignstack_attr.has_indirect_call, 0
  %clutter = alloca i8, addrspace(5) ; Force non-zero offset for next alloca
  store volatile i8 3, ptr addrspace(5) %clutter
  %alloca.align = alloca i32, align 4, addrspace(5)
  store volatile i32 9, ptr addrspace(5) %alloca.align, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "stackrealign" }
attributes #2 = { nounwind alignstack=128 }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
