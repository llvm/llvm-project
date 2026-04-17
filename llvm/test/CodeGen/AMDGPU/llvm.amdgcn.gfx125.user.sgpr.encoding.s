	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	many_inreg_i32                  ; -- Begin function many_inreg_i32
	.p2align	8
	.type	many_inreg_i32,@function
many_inreg_i32:                         ; @many_inreg_i32
; %bb.0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel many_inreg_i32
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 400
		.amdhsa_user_sgpr_count 32
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_kernarg_preload_length 24
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 32
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 1
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	many_inreg_i32, .Lfunc_end0-many_inreg_i32
                                        ; -- End function
	.set many_inreg_i32.num_vgpr, 0
	.set many_inreg_i32.num_agpr, 0
	.set many_inreg_i32.numbered_sgpr, 0
	.set many_inreg_i32.num_named_barrier, 0
	.set many_inreg_i32.private_seg_size, 0
	.set many_inreg_i32.uses_vcc, 0
	.set many_inreg_i32.uses_flat_scratch, 0
	.set many_inreg_i32.has_dyn_sized_stack, 0
	.set many_inreg_i32.has_recursion, 0
	.set many_inreg_i32.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 12
; TotalNumSgprs: 32
; NumVgprs: 0
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 32
; NumVGPRsForWavesPerEU: 1
; NamedBarCnt: 0
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 32
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .name:           a0
        .offset:         0
        .size:           4
        .value_kind:     by_value
      - .name:           a1
        .offset:         4
        .size:           4
        .value_kind:     by_value
      - .name:           a2
        .offset:         8
        .size:           4
        .value_kind:     by_value
      - .name:           a3
        .offset:         12
        .size:           4
        .value_kind:     by_value
      - .name:           a4
        .offset:         16
        .size:           4
        .value_kind:     by_value
      - .name:           a5
        .offset:         20
        .size:           4
        .value_kind:     by_value
      - .name:           a6
        .offset:         24
        .size:           4
        .value_kind:     by_value
      - .name:           a7
        .offset:         28
        .size:           4
        .value_kind:     by_value
      - .name:           a8
        .offset:         32
        .size:           4
        .value_kind:     by_value
      - .name:           a9
        .offset:         36
        .size:           4
        .value_kind:     by_value
      - .name:           a10
        .offset:         40
        .size:           4
        .value_kind:     by_value
      - .name:           a11
        .offset:         44
        .size:           4
        .value_kind:     by_value
      - .name:           a12
        .offset:         48
        .size:           4
        .value_kind:     by_value
      - .name:           a13
        .offset:         52
        .size:           4
        .value_kind:     by_value
      - .name:           a14
        .offset:         56
        .size:           4
        .value_kind:     by_value
      - .name:           a15
        .offset:         60
        .size:           4
        .value_kind:     by_value
      - .name:           a16
        .offset:         64
        .size:           4
        .value_kind:     by_value
      - .name:           a17
        .offset:         68
        .size:           4
        .value_kind:     by_value
      - .name:           a18
        .offset:         72
        .size:           4
        .value_kind:     by_value
      - .name:           a19
        .offset:         76
        .size:           4
        .value_kind:     by_value
      - .name:           a20
        .offset:         80
        .size:           4
        .value_kind:     by_value
      - .name:           a21
        .offset:         84
        .size:           4
        .value_kind:     by_value
      - .name:           a22
        .offset:         88
        .size:           4
        .value_kind:     by_value
      - .name:           a23
        .offset:         92
        .size:           4
        .value_kind:     by_value
      - .name:           a24
        .offset:         96
        .size:           4
        .value_kind:     by_value
      - .name:           a25
        .offset:         100
        .size:           4
        .value_kind:     by_value
      - .name:           a26
        .offset:         104
        .size:           4
        .value_kind:     by_value
      - .name:           a27
        .offset:         108
        .size:           4
        .value_kind:     by_value
      - .name:           a28
        .offset:         112
        .size:           4
        .value_kind:     by_value
      - .name:           a29
        .offset:         116
        .size:           4
        .value_kind:     by_value
      - .name:           a30
        .offset:         120
        .size:           4
        .value_kind:     by_value
      - .name:           a31
        .offset:         124
        .size:           4
        .value_kind:     by_value
      - .name:           a32
        .offset:         128
        .size:           4
        .value_kind:     by_value
      - .name:           a33
        .offset:         132
        .size:           4
        .value_kind:     by_value
      - .name:           a34
        .offset:         136
        .size:           4
        .value_kind:     by_value
      - .name:           a35
        .offset:         140
        .size:           4
        .value_kind:     by_value
      - .offset:         144
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         148
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         152
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         156
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         158
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         160
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         162
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         164
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         166
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         184
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         192
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         200
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         208
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         224
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         232
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         240
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         248
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         256
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         344
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 400
    .max_flat_workgroup_size: 1024
    .name:           many_inreg_i32
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         many_inreg_i32.kd
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
