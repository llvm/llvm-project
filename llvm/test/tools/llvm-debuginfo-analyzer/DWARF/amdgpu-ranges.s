# REQUIRES: amdgpu-registered-target

# Regression test for:
#  - DW_AT_ranges not being read properly
#  - Instructions at DW_AT_high_pc of a scope incorrectly included in the scope

# test.hlsl
# 01 RWBuffer<float> u0 : register(u0);
# 02 RWBuffer<float> u1 : register(u1);
# 03 
# 04 [RootSignature("DescriptorTable(UAV(u0,numDescriptors=2))")]
# 05 [numthreads(64,1,1)]
# 06 void main(uint3 dtid : SV_DispatchThreadID) {
# 07   float my_var = u0[dtid.x];
# 08   [loop]
# 09   for (int i = 0; i < 10; i++) {
# 10     float my_local_var = i*2;
# 11     my_var += my_local_var;
# 12   }
# 13   u1[dtid.x] = my_var;
# 14 }

# RUN: llvm-mc %s --mcpu=gfx1100 -triple=amdgcn-amd-amdpal -filetype=obj -o - | \
# RUN: llvm-debuginfo-analyzer --attribute=all \
# RUN:                         --print=all \
# RUN:                         --output-sort=offset \
# RUN:                         - 2>&1 | \
# RUN: FileCheck %s

# Make sure these two ranges are present and point to the correct offsets
# CHECK: [005] {Range} Lines 10:9 [0x00000000b0:0x00000000c8]
# CHECK: [005] {Range} Lines 11:13 [0x00000000f4:0x0000000114]

# Make sure the offset 0x114 does not show up at the scope level 005 and 004
# CHECK-NOT: [0x0000000114][005]
# CHECK-NOT: [0x0000000114][004]
# CHECK: [0x0000000114][003]

	.file	0 "test.hlsl"
	.text
	.globl	_amdgpu_cs_main
	.p2align	8
	.type	_amdgpu_cs_main,@function
_amdgpu_cs_main:
.Lfunc_begin0:
	.loc	0 6 0
	.cfi_sections .debug_frame
	.cfi_startproc
	v_writelane_b32 v1, s4, 0
	s_mov_b32 s0, s2
	s_mov_b32 s4, s1
	v_readlane_b32 s1, v1, 0
	s_mov_b32 s8, s0
.Ltmp0:
	.loc	0 6 24 prologue_end
	s_getpc_b64 s[2:3]
	s_mov_b32 s1, 0x3ff
	v_and_b32_e64 v0, v0, s1
	s_mov_b32 s1, 6
	v_lshl_add_u32 v0, s0, s1, v0
	scratch_store_b32 off, v0, off offset:4
.Ltmp1:
	.loc	0 0 24 is_stmt 0
	s_mov_b32 s1, -1
	s_mov_b32 s0, 0
	s_mov_b32 s6, s0
	s_mov_b32 s7, s1
	.loc	0 7 25 is_stmt 1
	s_and_b64 s[2:3], s[2:3], s[6:7]
	s_mov_b32 s1, 0
	s_mov_b32 s5, s1
	s_or_b64 s[2:3], s[2:3], s[4:5]
	v_writelane_b32 v1, s2, 1
	v_writelane_b32 v1, s3, 2
.Ltmp2:
	s_load_b128 s[4:7], s[2:3], 0x0
	s_waitcnt lgkmcnt(0)
	buffer_load_format_x v0, v0, s[4:7], s0 idxen
.Ltmp3:
	.loc	0 9 10
	s_waitcnt vmcnt(0)
	scratch_store_b32 off, v0, off
.Ltmp4:
	v_writelane_b32 v1, s0, 3
.Ltmp5:
.LBB0_1:
	.loc	0 0 10 is_stmt 0
	v_readlane_b32 s0, v1, 3
	scratch_load_b32 v0, off, off
.Ltmp6:
	s_mov_b32 s1, 1
.Ltmp7:
	.loc	0 10 34 is_stmt 1
	s_lshl_b32 s2, s0, s1
	.loc	0 10 33 is_stmt 0
	v_cvt_f32_u32_e64 v2, s2
.Ltmp8:
	.loc	0 11 19 is_stmt 1
	s_waitcnt vmcnt(0)
	v_add_f32_e64 v0, v0, v2
.Ltmp9:
	.loc	0 9 35
	s_add_i32 s0, s0, s1
.Ltmp10:
	.loc	0 0 35 is_stmt 0
	s_mov_b32 s1, 10
	.loc	0 9 28
	s_cmp_lg_u32 s0, s1
	v_mov_b32_e32 v2, v0
.Ltmp11:
	.loc	0 0 28
	scratch_store_b32 off, v2, off
	v_writelane_b32 v1, s0, 3
.Ltmp12:
	.loc	0 9 10
	scratch_store_b32 off, v0, off offset:8
.Ltmp13:
	s_cbranch_scc1 .LBB0_1
.Ltmp14:
	.loc	0 11 19 is_stmt 1
	v_readlane_b32 s0, v1, 1
.Ltmp15:
	v_readlane_b32 s1, v1, 2
	scratch_load_b32 v0, off, off offset:4
	scratch_load_b32 v6, off, off offset:8
.Ltmp16:
	.loc	0 13 21
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v2, v6
	v_mov_b32_e32 v3, v6
	v_mov_b32_e32 v4, v6
	v_mov_b32_e32 v5, v6
	s_load_b128 s[0:3], s[0:1], 0x20
	s_mov_b32 s4, 0
	s_waitcnt lgkmcnt(0)
	buffer_store_format_xyzw v[2:5], v0, s[0:3], s4 idxen
	.loc	0 14 8
	s_endpgm
.Ltmp17:
.Lfunc_end0:
	.size	_amdgpu_cs_main, .Lfunc_end0-_amdgpu_cs_main
	.cfi_endproc

	.set _amdgpu_cs_main.num_vgpr, 7
	.set _amdgpu_cs_main.num_agpr, 0
	.set _amdgpu_cs_main.numbered_sgpr, 11
	.set _amdgpu_cs_main.num_named_barrier, 0
	.set _amdgpu_cs_main.private_seg_size, 16
	.set _amdgpu_cs_main.uses_vcc, 0
	.set _amdgpu_cs_main.uses_flat_scratch, 0
	.set _amdgpu_cs_main.has_dyn_sized_stack, 0
	.set _amdgpu_cs_main.has_recursion, 0
	.set _amdgpu_cs_main.has_indirect_call, 0
	.set _amdgpu_cs_main.num_vgpr_rank_sum, 0
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.section	.debug_loclists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0
.Ldebug_list_header_start0:
	.short	5
	.byte	8
	.byte	0
	.long	4
.Lloclists_table_base0:
	.long	.Ldebug_loc0-.Lloclists_table_base0
	.long	.Ldebug_loc1-.Lloclists_table_base0
	.long	.Ldebug_loc2-.Lloclists_table_base0
	.long	.Ldebug_loc3-.Lloclists_table_base0
.Ldebug_loc0:
	.byte	4
	.uleb128 .Ltmp1-.Lfunc_begin0
	.uleb128 .Ltmp3-.Lfunc_begin0
	.byte	5
	.byte	144
	.byte	128
	.byte	20
	.byte	147
	.byte	4
	.byte	0
.Ldebug_loc1:
	.byte	4
	.uleb128 .Ltmp3-.Lfunc_begin0
	.uleb128 .Ltmp4-.Lfunc_begin0
	.byte	3
	.byte	144
	.byte	128
	.byte	20
	.byte	4
	.uleb128 .Ltmp6-.Lfunc_begin0
	.uleb128 .Ltmp13-.Lfunc_begin0
	.byte	3
	.byte	144
	.byte	128
	.byte	20
	.byte	4
	.uleb128 .Ltmp16-.Lfunc_begin0
	.uleb128 .Lfunc_end0-.Lfunc_begin0
	.byte	3
	.byte	144
	.byte	134
	.byte	20
	.byte	0
.Ldebug_loc2:
	.byte	4
	.uleb128 .Ltmp2-.Lfunc_begin0
	.uleb128 .Ltmp5-.Lfunc_begin0
	.byte	3
	.byte	17
	.byte	0
	.byte	159
	.byte	4
	.uleb128 .Ltmp6-.Lfunc_begin0
	.uleb128 .Ltmp15-.Lfunc_begin0
	.byte	2
	.byte	144
	.byte	32
	.byte	0
.Ldebug_loc3:
	.byte	4
	.uleb128 .Ltmp8-.Lfunc_begin0
	.uleb128 .Ltmp11-.Lfunc_begin0
	.byte	3
	.byte	144
	.byte	130
	.byte	20
	.byte	0
.Ldebug_list_header_end0:
	.section	.debug_abbrev,"",@progbits
	.byte	1
	.byte	17
	.byte	1
	.byte	37
	.byte	37
	.byte	19
	.byte	5
	.byte	3
	.byte	37
	.byte	114
	.byte	23
	.byte	16
	.byte	23
	.byte	17
	.byte	27
	.byte	18
	.byte	6
	.byte	115
	.byte	23
	.byte	116
	.byte	23
	.ascii	"\214\001"
	.byte	23
	.byte	0
	.byte	0
	.byte	2
	.byte	52
	.byte	0
	.byte	3
	.byte	37
	.byte	73
	.byte	19
	.byte	63
	.byte	25
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	110
	.byte	37
	.byte	0
	.byte	0
	.byte	3
	.byte	2
	.byte	1
	.byte	3
	.byte	37
	.byte	11
	.byte	11
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.ascii	"\210\001"
	.byte	15
	.byte	0
	.byte	0
	.byte	4
	.byte	47
	.byte	0
	.byte	73
	.byte	19
	.byte	3
	.byte	37
	.byte	0
	.byte	0
	.byte	5
	.byte	36
	.byte	0
	.byte	3
	.byte	37
	.byte	62
	.byte	11
	.byte	11
	.byte	11
	.byte	0
	.byte	0
	.byte	6
	.byte	46
	.byte	1
	.byte	17
	.byte	27
	.byte	18
	.byte	6
	.byte	3
	.byte	37
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	63
	.byte	25
	.byte	0
	.byte	0
	.byte	7
	.byte	5
	.byte	0
	.byte	2
	.byte	34
	.byte	3
	.byte	37
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	73
	.byte	19
	.byte	0
	.byte	0
	.byte	8
	.byte	52
	.byte	0
	.byte	2
	.byte	34
	.byte	3
	.byte	37
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	73
	.byte	19
	.byte	0
	.byte	0
	.byte	9
	.byte	11
	.byte	1
	.byte	17
	.byte	27
	.byte	18
	.byte	6
	.byte	0
	.byte	0
	.byte	10
	.byte	11
	.byte	1
	.byte	85
	.byte	35
	.byte	0
	.byte	0
	.byte	11
	.byte	22
	.byte	0
	.byte	73
	.byte	19
	.byte	3
	.byte	37
	.byte	0
	.byte	0
	.byte	12
	.byte	2
	.byte	1
	.byte	3
	.byte	37
	.byte	11
	.byte	11
	.ascii	"\210\001"
	.byte	15
	.byte	0
	.byte	0
	.byte	13
	.byte	48
	.byte	0
	.byte	73
	.byte	19
	.byte	3
	.byte	37
	.byte	28
	.byte	13
	.byte	0
	.byte	0
	.byte	14
	.byte	13
	.byte	0
	.byte	3
	.byte	37
	.byte	73
	.byte	19
	.ascii	"\210\001"
	.byte	15
	.byte	56
	.byte	11
	.byte	50
	.byte	11
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0
.Ldebug_info_start0:
	.short	5
	.byte	1
	.byte	8
	.long	.debug_abbrev
	.byte	1
	.byte	0
	.short	4
	.byte	1
	.long	.Lstr_offsets_base0
	.long	.Lline_table_start0
	.byte	0
	.long	.Lfunc_end0-.Lfunc_begin0
	.long	.Laddr_table_base0
	.long	.Lrnglists_table_base0
	.long	.Lloclists_table_base0
	.byte	2
	.byte	2
	.long	51

	.byte	0
	.byte	1
	.byte	6
	.byte	3
	.byte	5
	.byte	4
	.byte	0
	.byte	1
	.byte	4
	.byte	4
	.long	64
	.byte	4
	.byte	0
	.byte	5
	.byte	3
	.byte	4
	.byte	4
	.byte	2
	.byte	7
	.long	51

	.byte	0
	.byte	2
	.byte	8
	.byte	6
	.byte	0
	.long	.Lfunc_end0-.Lfunc_begin0
	.byte	9
	.byte	0
	.byte	6

	.byte	7
	.byte	0
	.byte	10
	.byte	0
	.byte	6
	.long	133
	.byte	8
	.byte	1
	.byte	19
	.byte	0
	.byte	7
	.long	64
	.byte	9
	.byte	1
	.long	.Ltmp16-.Ltmp3
	.byte	8
	.byte	2
	.byte	20
	.byte	0
	.byte	9
	.long	188
	.byte	10
	.byte	0
	.byte	8
	.byte	3
	.byte	21
	.byte	0
	.byte	10
	.long	64
	.byte	0
	.byte	0
	.byte	0
	.byte	11
	.long	139
	.byte	18
	.byte	12
	.byte	17
	.byte	12
	.byte	4
	.byte	4
	.long	184
	.byte	4
	.byte	13
	.long	188
	.byte	13
	.byte	3
	.byte	14
	.byte	14
	.long	184
	.byte	4
	.byte	0
	.byte	1
	.byte	14
	.byte	15
	.long	184
	.byte	4
	.byte	4
	.byte	1
	.byte	14
	.byte	16
	.long	184
	.byte	4
	.byte	8
	.byte	1
	.byte	0
	.byte	5
	.byte	11
	.byte	7
	.byte	4
	.byte	5
	.byte	12
	.byte	5
	.byte	4
	.byte	0
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end1-.Ldebug_list_header_start1
.Ldebug_list_header_start1:
	.short	5
	.byte	8
	.byte	0
	.long	1
.Lrnglists_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	4
	.uleb128 .Ltmp7-.Lfunc_begin0
	.uleb128 .Ltmp9-.Lfunc_begin0
	.byte	4
	.uleb128 .Ltmp14-.Lfunc_begin0
	.uleb128 .Ltmp16-.Lfunc_begin0
	.byte	0
.Ldebug_list_header_end1:
	.section	.debug_str_offsets,"",@progbits
	.long	92
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"dxc"
.Linfo_string1:
	.asciz	"test.hlsl"
.Linfo_string2:
	.asciz	"u0"
.Linfo_string3:
	.asciz	"RWBuffer<float>"
.Linfo_string4:
	.asciz	"float"
.Linfo_string5:
	.asciz	"element"
.Linfo_string6:
	.asciz	"?u0@@3V?$RWBuffer@M@@A"
.Linfo_string7:
	.asciz	"u1"
.Linfo_string8:
	.asciz	"?u1@@3V?$RWBuffer@M@@A"
.Linfo_string9:
	.asciz	"main"
.Linfo_string10:
	.asciz	"dtid"
.Linfo_string11:
	.asciz	"uint3"
.Linfo_string12:
	.asciz	"vector<unsigned int, 3>"
.Linfo_string13:
	.asciz	"unsigned int"
.Linfo_string14:
	.asciz	"int"
.Linfo_string15:
	.asciz	"element_count"
.Linfo_string16:
	.asciz	"x"
.Linfo_string17:
	.asciz	"y"
.Linfo_string18:
	.asciz	"z"
.Linfo_string19:
	.asciz	"my_var"
.Linfo_string20:
	.asciz	"i"
.Linfo_string21:
	.asciz	"my_local_var"
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string3
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string13
	.long	.Linfo_string14
	.long	.Linfo_string15
	.long	.Linfo_string16
	.long	.Linfo_string17
	.long	.Linfo_string18
	.long	.Linfo_string12
	.long	.Linfo_string11
	.long	.Linfo_string19
	.long	.Linfo_string20
	.long	.Linfo_string21
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0
.Ldebug_addr_start0:
	.short	5
	.byte	8
	.byte	0
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Ltmp3
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
