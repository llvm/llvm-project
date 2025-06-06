	.section	.AMDGPU.config,"",@progbits
	.long	47176
	.long	11468929
	.long	47180
	.long	5008
	.long	47200
	.long	0
	.long	4
	.long	0
	.long	8
	.long	0
	.text
	.protected	main                    ; -- Begin function main
	.globl	main
	.p2align	8
	.type	main,@function
main:                                   ; @main
; %bb.0:                                ; %entry
	s_load_dword s0, s[4:5], 0x3c
	v_and_b32_e32 v0, 0x3c0, v0
	v_lshrrev_b32_e32 v0, 6, v0
	s_waitcnt lgkmcnt(0)
	s_addk_i32 s0, 0x3ff
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 22
	s_add_i32 s0, s0, s1
	s_ashr_i32 s6, s0, 10
	s_cmp_lt_i32 s6, 1
	v_readfirstlane_b32 s0, v0
	s_cbranch_scc1 .LBB0_3
; %bb.1:                                ; %for.body.lr.ph.i
	s_lshr_b32 s2, s0, 2
	s_lshl_b32 s0, s0, 8
	v_mbcnt_lo_u32_b32 v0, -1, 0
	s_and_b32 s3, s0, 0x300
	s_load_dwordx2 s[0:1], s[4:5], 0x24
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_lshrrev_b32_e32 v1, 6, v0
	v_add_u32_e32 v0, v0, v1
	s_add_i32 s3, s3, s2
	v_add_lshl_u32 v0, s3, v0, 2
	s_mov_b32 s2, 4
	s_mov_b32 s3, 0x20000
.LBB0_2:                                ; %for.body.i
                                        ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(0)
	buffer_load_dwordx4 v[2:5], v0, s[0:3], 0 offen
	buffer_load_dwordx4 v[6:9], v0, s[0:3], 0 offen
	s_add_i32 s6, s6, -1
	s_cmp_lg_u32 s6, 0
	s_waitcnt vmcnt(0)
	v_pk_add_f32 v[4:5], v[4:5], v[8:9]
	v_pk_add_f32 v[2:3], v[2:3], v[6:7]
	buffer_store_dwordx4 v[2:5], v0, s[0:3], 0 offen
	v_add_u32_e32 v0, 0x1000, v0
	s_cbranch_scc1 .LBB0_2
.LBB0_3:                                ; %add.exit
	s_endpgm
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
                                        ; -- End function
	.set main.num_vgpr, 10
	.set main.num_agpr, 0
	.set main.numbered_sgpr, 11
	.set main.private_seg_size, 0
	.set main.uses_vcc, 0
	.set main.uses_flat_scratch, 0
	.set main.has_dyn_sized_stack, 0
	.set main.has_recursion, 0
	.set main.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 200
; TotalNumSgprs: 17
; NumVgprs: 10
; NumAgprs: 0
; TotalNumVgprs: 10
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 17
; NumVGPRsForWavesPerEU: 10
; AccumOffset: 12
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 2
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.section	.AMDGPU.csdata,"",@progbits
	.section	".note.GNU-stack","",@progbits
	.amd_amdgpu_isa "amdgcn----gfx942"
