	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	bad_rp                          ; -- Begin function bad_rp
	.p2align	8
	.type	bad_rp,@function
bad_rp:                                 ; @bad_rp
; %bb.0:
	s_load_dword s0, s[4:5], 0x0
	s_load_dword s1, s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v0, s0
	ds_read_b128 a[0:3], v0
	ds_read_b128 a[4:7], v0 offset:16
	ds_read_b128 a[8:11], v0 offset:32
	ds_read_b128 a[12:15], v0 offset:48
	ds_read_b128 a[16:19], v0 offset:64
	ds_read_b128 a[20:23], v0 offset:80
	ds_read_b128 a[24:27], v0 offset:96
	ds_read_b128 a[28:31], v0 offset:112
	ds_read_b128 a[32:35], v0 offset:128
	ds_read_b128 a[36:39], v0 offset:144
	ds_read_b128 a[40:43], v0 offset:160
	ds_read_b128 a[44:47], v0 offset:176
	ds_read_b128 a[48:51], v0 offset:192
	ds_read_b128 a[52:55], v0 offset:208
	ds_read_b128 a[56:59], v0 offset:224
	ds_read_b128 a[60:63], v0 offset:240
	s_bitcmp1_b32 s1, 0
	s_cselect_b64 s[0:1], -1, 0
	s_xor_b64 s[0:1], s[0:1], -1
.LBB0_1:                                ; %bb.1
                                        ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(14)
	v_mfma_f32_32x32x16_f16 v[240:255], a[0:3], a[0:3], 0
	s_andn2_b64 vcc, exec, s[0:1]
	v_mfma_f32_32x32x16_f16 v[224:239], a[4:7], a[4:7], v[240:255]
	s_waitcnt lgkmcnt(13)
	v_mfma_f32_32x32x16_f16 v[208:223], a[8:11], a[8:11], v[224:239]
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x16_f16 v[192:207], a[12:15], a[12:15], v[208:223]
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_32x32x16_f16 v[176:191], a[16:19], a[16:19], v[192:207]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[160:175], a[20:23], a[20:23], v[176:191]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[144:159], a[24:27], a[24:27], v[160:175]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[128:143], a[28:31], a[28:31], v[144:159]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[112:127], a[32:35], a[32:35], v[128:143]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[96:111], a[36:39], a[36:39], v[112:127]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[80:95], a[40:43], a[40:43], v[96:111]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[64:79], a[44:47], a[44:47], v[80:95]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[48:63], a[48:51], a[48:51], v[64:79]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[32:47], a[52:55], a[52:55], v[48:63]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[16:31], a[56:59], a[56:59], v[32:47]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[0:15], a[60:63], a[60:63], v[16:31]
	s_cbranch_vccnz .LBB0_1
; %bb.2:                                ; %bb.2
	s_load_dwordx2 s[0:1], s[4:5], 0x8
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[168:169], s[0:1]
	flat_store_dwordx4 v[168:169], v[244:247] offset:16
	flat_store_dwordx4 v[168:169], v[240:243]
	flat_store_dwordx4 v[168:169], v[224:227] offset:32
	flat_store_dwordx4 v[168:169], v[228:231] offset:48
	flat_store_dwordx4 v[168:169], v[208:211] offset:64
	flat_store_dwordx4 v[168:169], v[212:215] offset:80
	flat_store_dwordx4 v[168:169], v[192:195] offset:96
	flat_store_dwordx4 v[168:169], v[196:199] offset:112
	flat_store_dwordx4 v[168:169], v[180:183] offset:144
	flat_store_dwordx4 v[168:169], v[176:179] offset:128
	flat_store_dwordx4 v[168:169], v[160:163] offset:160
	flat_store_dwordx4 v[168:169], v[164:167] offset:176
	flat_store_dwordx4 v[168:169], v[148:151] offset:208
	flat_store_dwordx4 v[168:169], v[156:159] offset:240
	flat_store_dwordx4 v[168:169], v[152:155] offset:224
	flat_store_dwordx4 v[168:169], v[144:147] offset:192
	flat_store_dwordx4 v[168:169], v[140:143] offset:272
	flat_store_dwordx4 v[168:169], v[136:139] offset:256
	flat_store_dwordx4 v[168:169], v[132:135] offset:240
	flat_store_dwordx4 v[168:169], v[128:131] offset:224
	flat_store_dwordx4 v[168:169], v[124:127] offset:304
	flat_store_dwordx4 v[168:169], v[120:123] offset:288
	flat_store_dwordx4 v[168:169], v[116:119] offset:272
	flat_store_dwordx4 v[168:169], v[112:115] offset:256
	flat_store_dwordx4 v[168:169], v[108:111] offset:336
	flat_store_dwordx4 v[168:169], v[104:107] offset:320
	flat_store_dwordx4 v[168:169], v[100:103] offset:304
	flat_store_dwordx4 v[168:169], v[96:99] offset:288
	flat_store_dwordx4 v[168:169], v[92:95] offset:368
	flat_store_dwordx4 v[168:169], v[88:91] offset:352
	flat_store_dwordx4 v[168:169], v[84:87] offset:336
	flat_store_dwordx4 v[168:169], v[80:83] offset:320
	flat_store_dwordx4 v[168:169], v[76:79] offset:400
	flat_store_dwordx4 v[168:169], v[72:75] offset:384
	flat_store_dwordx4 v[168:169], v[68:71] offset:368
	flat_store_dwordx4 v[168:169], v[64:67] offset:352
	flat_store_dwordx4 v[168:169], v[60:63] offset:432
	flat_store_dwordx4 v[168:169], v[56:59] offset:416
	flat_store_dwordx4 v[168:169], v[52:55] offset:400
	flat_store_dwordx4 v[168:169], v[48:51] offset:384
	flat_store_dwordx4 v[168:169], v[44:47] offset:464
	flat_store_dwordx4 v[168:169], v[40:43] offset:448
	flat_store_dwordx4 v[168:169], v[36:39] offset:432
	flat_store_dwordx4 v[168:169], v[32:35] offset:416
	flat_store_dwordx4 v[168:169], v[28:31] offset:496
	flat_store_dwordx4 v[168:169], v[24:27] offset:480
	flat_store_dwordx4 v[168:169], v[20:23] offset:464
	flat_store_dwordx4 v[168:169], v[16:19] offset:448
	flat_store_dwordx4 v[168:169], v[12:15] offset:528
	flat_store_dwordx4 v[168:169], v[8:11] offset:512
	flat_store_dwordx4 v[168:169], v[4:7] offset:496
	flat_store_dwordx4 v[168:169], v[0:3] offset:480
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel bad_rp
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 8
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 320
		.amdhsa_next_free_sgpr 6
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
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
	.size	bad_rp, .Lfunc_end0-bad_rp
                                        ; -- End function
	.set bad_rp.num_vgpr, 256
	.set bad_rp.num_agpr, 64
	.set bad_rp.numbered_sgpr, 6
	.set bad_rp.private_seg_size, 0
	.set bad_rp.uses_vcc, 1
	.set bad_rp.uses_flat_scratch, 0
	.set bad_rp.has_dyn_sized_stack, 0
	.set bad_rp.has_recursion, 0
	.set bad_rp.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 796
; TotalNumSgprs: 12
; NumVgprs: 256
; NumAgprs: 64
; TotalNumVgprs: 320
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 39
; NumSGPRsForWavesPerEU: 12
; NumVGPRsForWavesPerEU: 320
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.globl	good_rp                         ; -- Begin function good_rp
	.p2align	8
	.type	good_rp,@function
good_rp:                                ; @good_rp
; %bb.0:
	s_load_dword s0, s[4:5], 0x10
	s_load_dword s1, s[4:5], 0x0
	s_waitcnt lgkmcnt(0)
	s_bitcmp1_b32 s0, 0
	v_mov_b32_e32 v0, s1
	ds_read_b128 v[176:179], v0
	ds_read_b128 v[180:183], v0 offset:16
	ds_read_b128 v[184:187], v0 offset:32
	ds_read_b128 v[188:191], v0 offset:48
	ds_read_b128 v[192:195], v0 offset:64
	ds_read_b128 v[196:199], v0 offset:80
	ds_read_b128 v[200:203], v0 offset:96
	ds_read_b128 v[204:207], v0 offset:112
	ds_read_b128 v[208:211], v0 offset:128
	ds_read_b128 v[212:215], v0 offset:144
	ds_read_b128 v[216:219], v0 offset:160
	s_cselect_b64 s[0:1], -1, 0
	s_xor_b64 s[0:1], s[0:1], -1
	v_cndmask_b32_e64 v0, 0, 1, s[0:1]
	v_cmp_ne_u32_e64 s[0:1], 1, v0
.LBB1_1:                                ; %bb.1
                                        ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_32x32x16_f16 v[160:175], v[176:179], v[176:179], 0
	s_and_b64 vcc, exec, s[0:1]
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_32x32x16_f16 v[144:159], v[180:183], v[180:183], v[160:175]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x16_f16 v[128:143], v[184:187], v[184:187], v[144:159]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 v[112:127], v[188:191], v[188:191], v[128:143]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 v[96:111], v[192:195], v[192:195], v[112:127]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 v[80:95], v[196:199], v[196:199], v[96:111]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 v[64:79], v[200:203], v[200:203], v[80:95]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 v[48:63], v[204:207], v[204:207], v[64:79]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 v[32:47], v[208:211], v[208:211], v[48:63]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 v[16:31], v[212:215], v[212:215], v[32:47]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 v[0:15], v[216:219], v[216:219], v[16:31]
	s_cbranch_vccnz .LBB1_1
; %bb.2:                                ; %bb.2
	s_load_dwordx2 s[0:1], s[4:5], 0x8
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[88:89], s[0:1]
	flat_store_dwordx4 v[88:89], v[164:167] offset:16
	flat_store_dwordx4 v[88:89], v[160:163]
	flat_store_dwordx4 v[88:89], v[144:147] offset:32
	flat_store_dwordx4 v[88:89], v[148:151] offset:48
	flat_store_dwordx4 v[88:89], v[128:131] offset:64
	flat_store_dwordx4 v[88:89], v[132:135] offset:80
	flat_store_dwordx4 v[88:89], v[112:115] offset:96
	flat_store_dwordx4 v[88:89], v[116:119] offset:112
	flat_store_dwordx4 v[88:89], v[100:103] offset:144
	flat_store_dwordx4 v[88:89], v[96:99] offset:128
	flat_store_dwordx4 v[88:89], v[80:83] offset:160
	flat_store_dwordx4 v[88:89], v[84:87] offset:176
	flat_store_dwordx4 v[88:89], v[68:71] offset:208
	flat_store_dwordx4 v[88:89], v[76:79] offset:240
	flat_store_dwordx4 v[88:89], v[72:75] offset:224
	flat_store_dwordx4 v[88:89], v[64:67] offset:192
	flat_store_dwordx4 v[88:89], v[60:63] offset:272
	flat_store_dwordx4 v[88:89], v[56:59] offset:256
	flat_store_dwordx4 v[88:89], v[52:55] offset:240
	flat_store_dwordx4 v[88:89], v[48:51] offset:224
	flat_store_dwordx4 v[88:89], v[44:47] offset:304
	flat_store_dwordx4 v[88:89], v[40:43] offset:288
	flat_store_dwordx4 v[88:89], v[36:39] offset:272
	flat_store_dwordx4 v[88:89], v[32:35] offset:256
	flat_store_dwordx4 v[88:89], v[28:31] offset:336
	flat_store_dwordx4 v[88:89], v[24:27] offset:320
	flat_store_dwordx4 v[88:89], v[20:23] offset:304
	flat_store_dwordx4 v[88:89], v[16:19] offset:288
	flat_store_dwordx4 v[88:89], v[12:15] offset:368
	flat_store_dwordx4 v[88:89], v[8:11] offset:352
	flat_store_dwordx4 v[88:89], v[4:7] offset:336
	flat_store_dwordx4 v[88:89], v[0:3] offset:320
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel good_rp
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 8
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 220
		.amdhsa_next_free_sgpr 6
		.amdhsa_accum_offset 220
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	good_rp, .Lfunc_end1-good_rp
                                        ; -- End function
	.set good_rp.num_vgpr, 220
	.set good_rp.num_agpr, 0
	.set good_rp.numbered_sgpr, 6
	.set good_rp.private_seg_size, 0
	.set good_rp.uses_vcc, 1
	.set good_rp.uses_flat_scratch, 0
	.set good_rp.has_dyn_sized_stack, 0
	.set good_rp.has_recursion, 0
	.set good_rp.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 556
; TotalNumSgprs: 12
; NumVgprs: 220
; NumAgprs: 0
; TotalNumVgprs: 220
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 27
; NumSGPRsForWavesPerEU: 12
; NumVGPRsForWavesPerEU: 220
; AccumOffset: 220
; Occupancy: 2
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 54
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     64
    .args:
      - .address_space:  local
        .name:           in0
        .offset:         0
        .pointee_align:  1
        .size:           4
        .value_kind:     dynamic_shared_pointer
      - .address_space:  generic
        .name:           out
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .name:           cond
        .offset:         16
        .size:           1
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         104
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         112
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         120
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         128
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         136
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         144
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
      - .offset:         224
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .max_flat_workgroup_size: 64
    .name:           bad_rp
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         bad_rp.kd
    .uses_dynamic_stack: false
    .vgpr_count:     320
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  local
        .name:           in0
        .offset:         0
        .pointee_align:  1
        .size:           4
        .value_kind:     dynamic_shared_pointer
      - .address_space:  generic
        .name:           out
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .name:           cond
        .offset:         16
        .size:           1
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         104
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         112
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         120
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         128
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         136
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         144
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
      - .offset:         224
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .max_flat_workgroup_size: 64
    .name:           good_rp
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         good_rp.kd
    .uses_dynamic_stack: false
    .vgpr_count:     220
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
