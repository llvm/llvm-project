	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
	.amdhsa_code_object_version 5
	.globl	preserve_wwm_copy_dstreg        ; -- Begin function preserve_wwm_copy_dstreg
	.p2align	2
	.type	preserve_wwm_copy_dstreg,@function
preserve_wwm_copy_dstreg:               ; @preserve_wwm_copy_dstreg
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 s16, s33
	s_mov_b32 s33, s32
	s_xor_saveexec_b64 s[18:19], -1
	buffer_store_dword v33, off, s[0:3], s33 offset:144 ; 4-byte Folded Spill
	buffer_store_dword v2, off, s[0:3], s33 offset:152 ; 4-byte Folded Spill
	s_mov_b64 exec, -1
	buffer_store_dword v41, off, s[0:3], s33 ; 4-byte Folded Spill
	buffer_store_dword v40, off, s[0:3], s33 offset:148 ; 4-byte Folded Spill
	s_mov_b64 exec, s[18:19]
                                        ; implicit-def: $vgpr2
	s_mov_b32 s21, s15
	v_writelane_b32 v2, s6, 0
	v_writelane_b32 v2, s7, 1
	v_writelane_b32 v2, s21, 2
	s_mov_b32 s22, s14
	v_writelane_b32 v2, s22, 3
	s_mov_b32 s23, s13
	v_writelane_b32 v2, s23, 4
	s_mov_b32 s24, s12
	v_writelane_b32 v2, s24, 5
	s_mov_b64 s[26:27], s[10:11]
	v_writelane_b32 v2, s26, 6
	v_writelane_b32 v41, s34, 2
	v_writelane_b32 v2, s27, 7
	v_writelane_b32 v41, s35, 3
	v_writelane_b32 v2, s8, 8
	v_writelane_b32 v41, s16, 4
	v_writelane_b32 v2, s9, 9
	v_writelane_b32 v41, s30, 0
	v_writelane_b32 v2, s4, 10
	s_addk_i32 s32, 0x2800
	v_writelane_b32 v41, s31, 1
	v_mov_b32_e32 v32, v31
	buffer_store_dword v0, off, s[0:3], s33 offset:8 ; 4-byte Folded Spill
	s_waitcnt vmcnt(0)
	buffer_store_dword v1, off, s[0:3], s33 offset:12 ; 4-byte Folded Spill
	v_writelane_b32 v2, s5, 11
	s_or_saveexec_b64 s[34:35], -1
	v_mov_b32_e32 v33, v2
	s_mov_b64 exec, s[34:35]
	;;#ASMSTART
	; def v[0:31]
	;;#ASMEND
	buffer_store_dword v0, off, s[0:3], s33 offset:16 ; 4-byte Folded Spill
	s_waitcnt vmcnt(0)
	buffer_store_dword v1, off, s[0:3], s33 offset:20 ; 4-byte Folded Spill
	buffer_store_dword v2, off, s[0:3], s33 offset:24 ; 4-byte Folded Spill
	buffer_store_dword v3, off, s[0:3], s33 offset:28 ; 4-byte Folded Spill
	buffer_store_dword v4, off, s[0:3], s33 offset:32 ; 4-byte Folded Spill
	buffer_store_dword v5, off, s[0:3], s33 offset:36 ; 4-byte Folded Spill
	buffer_store_dword v6, off, s[0:3], s33 offset:40 ; 4-byte Folded Spill
	buffer_store_dword v7, off, s[0:3], s33 offset:44 ; 4-byte Folded Spill
	buffer_store_dword v8, off, s[0:3], s33 offset:48 ; 4-byte Folded Spill
	buffer_store_dword v9, off, s[0:3], s33 offset:52 ; 4-byte Folded Spill
	buffer_store_dword v10, off, s[0:3], s33 offset:56 ; 4-byte Folded Spill
	buffer_store_dword v11, off, s[0:3], s33 offset:60 ; 4-byte Folded Spill
	buffer_store_dword v12, off, s[0:3], s33 offset:64 ; 4-byte Folded Spill
	buffer_store_dword v13, off, s[0:3], s33 offset:68 ; 4-byte Folded Spill
	buffer_store_dword v14, off, s[0:3], s33 offset:72 ; 4-byte Folded Spill
	buffer_store_dword v15, off, s[0:3], s33 offset:76 ; 4-byte Folded Spill
	buffer_store_dword v16, off, s[0:3], s33 offset:80 ; 4-byte Folded Spill
	buffer_store_dword v17, off, s[0:3], s33 offset:84 ; 4-byte Folded Spill
	buffer_store_dword v18, off, s[0:3], s33 offset:88 ; 4-byte Folded Spill
	buffer_store_dword v19, off, s[0:3], s33 offset:92 ; 4-byte Folded Spill
	buffer_store_dword v20, off, s[0:3], s33 offset:96 ; 4-byte Folded Spill
	buffer_store_dword v21, off, s[0:3], s33 offset:100 ; 4-byte Folded Spill
	buffer_store_dword v22, off, s[0:3], s33 offset:104 ; 4-byte Folded Spill
	buffer_store_dword v23, off, s[0:3], s33 offset:108 ; 4-byte Folded Spill
	buffer_store_dword v24, off, s[0:3], s33 offset:112 ; 4-byte Folded Spill
	buffer_store_dword v25, off, s[0:3], s33 offset:116 ; 4-byte Folded Spill
	buffer_store_dword v26, off, s[0:3], s33 offset:120 ; 4-byte Folded Spill
	buffer_store_dword v27, off, s[0:3], s33 offset:124 ; 4-byte Folded Spill
	buffer_store_dword v28, off, s[0:3], s33 offset:128 ; 4-byte Folded Spill
	buffer_store_dword v29, off, s[0:3], s33 offset:132 ; 4-byte Folded Spill
	buffer_store_dword v30, off, s[0:3], s33 offset:136 ; 4-byte Folded Spill
	buffer_store_dword v31, off, s[0:3], s33 offset:140 ; 4-byte Folded Spill
	;;#ASMSTART
	; def v40
	;;#ASMEND
	;;#ASMSTART
	; def s11
	;;#ASMEND
	s_or_saveexec_b64 s[34:35], -1
	v_mov_b32_e32 v40, v33
	s_mov_b64 exec, s[34:35]
	v_writelane_b32 v40, s11, 12
	;;#ASMSTART
	; def s12
	;;#ASMEND
	v_writelane_b32 v40, s12, 13
	;;#ASMSTART
	; def s13
	;;#ASMEND
	v_writelane_b32 v40, s13, 14
	;;#ASMSTART
	; def s14
	;;#ASMEND
	v_writelane_b32 v40, s14, 15
	;;#ASMSTART
	; def s15
	;;#ASMEND
	v_writelane_b32 v40, s15, 16
	s_getpc_b64 s[10:11]
	s_add_u32 s10, s10, foo@gotpcrel32@lo+4
	s_addc_u32 s11, s11, foo@gotpcrel32@hi+12
	;;#ASMSTART
	; def s16
	;;#ASMEND
	v_writelane_b32 v40, s16, 17
	s_load_dwordx2 s[10:11], s[10:11], 0x0
	;;#ASMSTART
	; def s17
	;;#ASMEND
	v_writelane_b32 v40, s17, 18
	;;#ASMSTART
	; def s18
	;;#ASMEND
	v_writelane_b32 v40, s18, 19
	;;#ASMSTART
	; def s19
	;;#ASMEND
	v_writelane_b32 v40, s19, 20
	;;#ASMSTART
	; def s20
	;;#ASMEND
	v_writelane_b32 v40, s20, 21
	s_waitcnt lgkmcnt(0)
	v_writelane_b32 v40, s10, 22
	v_writelane_b32 v40, s11, 23
	s_or_saveexec_b64 s[34:35], -1
	s_mov_b64 exec, s[34:35]
	v_readlane_b32 s16, v40, 22
	s_mov_b32 s12, s24
	s_mov_b32 s13, s23
	s_mov_b32 s14, s22
	v_mov_b32_e32 v31, v32
	s_mov_b32 s15, s21
	s_mov_b64 s[10:11], s[26:27]
	v_readlane_b32 s17, v40, 23
	buffer_store_dword v32, off, s[0:3], s33 offset:4 ; 4-byte Folded Spill
	s_swappc_b64 s[30:31], s[16:17]
	s_or_saveexec_b64 s[34:35], -1
	s_mov_b64 exec, s[34:35]
	v_readlane_b32 s11, v40, 12
	;;#ASMSTART
	; use s11
	;;#ASMEND
	v_readlane_b32 s12, v40, 13
	;;#ASMSTART
	; use s12
	;;#ASMEND
	v_readlane_b32 s13, v40, 14
	;;#ASMSTART
	; use s13
	;;#ASMEND
	v_readlane_b32 s14, v40, 15
	;;#ASMSTART
	; use s14
	;;#ASMEND
	v_readlane_b32 s15, v40, 16
	;;#ASMSTART
	; use s15
	;;#ASMEND
	v_readlane_b32 s16, v40, 17
	;;#ASMSTART
	; use s16
	;;#ASMEND
	v_readlane_b32 s17, v40, 18
	;;#ASMSTART
	; use s17
	;;#ASMEND
	v_readlane_b32 s18, v40, 19
	;;#ASMSTART
	; use s18
	;;#ASMEND
	v_readlane_b32 s19, v40, 20
	;;#ASMSTART
	; use s19
	;;#ASMEND
	v_readlane_b32 s20, v40, 21
	;;#ASMSTART
	; use s20
	;;#ASMEND
	;;#ASMSTART
	; def s21
	;;#ASMEND
	;;#ASMSTART
	; def s22
	;;#ASMEND
	;;#ASMSTART
	; def s23
	;;#ASMEND
	;;#ASMSTART
	; def s24
	;;#ASMEND
	;;#ASMSTART
	; def s25
	;;#ASMEND
	;;#ASMSTART
	; def s26
	;;#ASMEND
	;;#ASMSTART
	; def s27
	;;#ASMEND
	;;#ASMSTART
	; def s28
	;;#ASMEND
	;;#ASMSTART
	; def s29
	;;#ASMEND
	buffer_load_dword v31, off, s[0:3], s33 offset:4 ; 4-byte Folded Reload
	v_writelane_b32 v40, s21, 24
	v_writelane_b32 v40, s22, 25
	v_writelane_b32 v40, s23, 26
	v_writelane_b32 v40, s24, 27
	v_writelane_b32 v40, s25, 28
	v_writelane_b32 v40, s26, 29
	v_writelane_b32 v40, s27, 30
	v_writelane_b32 v40, s28, 31
	v_writelane_b32 v40, s29, 32
	v_readlane_b32 s4, v40, 10
	v_readlane_b32 s6, v40, 0
	v_readlane_b32 s8, v40, 8
	v_readlane_b32 s10, v40, 6
	v_readlane_b32 s16, v40, 22
	v_readlane_b32 s12, v40, 5
	v_readlane_b32 s13, v40, 4
	v_readlane_b32 s14, v40, 3
	v_readlane_b32 s15, v40, 2
	v_readlane_b32 s5, v40, 11
	v_readlane_b32 s7, v40, 1
	v_readlane_b32 s9, v40, 9
	v_readlane_b32 s11, v40, 7
	v_readlane_b32 s17, v40, 23
	s_or_saveexec_b64 s[34:35], -1
	s_mov_b64 exec, s[34:35]
	s_swappc_b64 s[30:31], s[16:17]
	s_or_saveexec_b64 s[34:35], -1
	s_mov_b64 exec, s[34:35]
	v_readlane_b32 s21, v40, 24
	;;#ASMSTART
	; use s21
	;;#ASMEND
	v_readlane_b32 s22, v40, 25
	;;#ASMSTART
	; use s22
	;;#ASMEND
	v_readlane_b32 s23, v40, 26
	;;#ASMSTART
	; use s23
	;;#ASMEND
	v_readlane_b32 s24, v40, 27
	;;#ASMSTART
	; use s24
	;;#ASMEND
	v_readlane_b32 s25, v40, 28
	;;#ASMSTART
	; use s25
	;;#ASMEND
	v_readlane_b32 s26, v40, 29
	;;#ASMSTART
	; use s26
	;;#ASMEND
	v_readlane_b32 s27, v40, 30
	;;#ASMSTART
	; use s27
	;;#ASMEND
	v_readlane_b32 s28, v40, 31
	;;#ASMSTART
	; use s28
	;;#ASMEND
	v_readlane_b32 s29, v40, 32
	;;#ASMSTART
	; use s29
	;;#ASMEND
	buffer_load_dword v31, off, s[0:3], s33 offset:4 ; 4-byte Folded Reload
	v_readlane_b32 s4, v40, 10
	v_readlane_b32 s6, v40, 0
	v_readlane_b32 s8, v40, 8
	v_readlane_b32 s10, v40, 6
	v_readlane_b32 s16, v40, 22
	v_readlane_b32 s5, v40, 11
	v_readlane_b32 s7, v40, 1
	v_readlane_b32 s9, v40, 9
	v_readlane_b32 s11, v40, 7
	v_readlane_b32 s12, v40, 5
	v_readlane_b32 s13, v40, 4
	v_readlane_b32 s14, v40, 3
	v_readlane_b32 s15, v40, 2
	v_readlane_b32 s17, v40, 23
	s_or_saveexec_b64 s[34:35], -1
	s_mov_b64 exec, s[34:35]
	s_swappc_b64 s[30:31], s[16:17]
	buffer_load_dword v0, off, s[0:3], s33 offset:8 ; 4-byte Folded Reload
	buffer_load_dword v1, off, s[0:3], s33 offset:12 ; 4-byte Folded Reload
	buffer_load_dword v2, off, s[0:3], s33 offset:16 ; 4-byte Folded Reload
	buffer_load_dword v3, off, s[0:3], s33 offset:20 ; 4-byte Folded Reload
	buffer_load_dword v4, off, s[0:3], s33 offset:24 ; 4-byte Folded Reload
	buffer_load_dword v5, off, s[0:3], s33 offset:28 ; 4-byte Folded Reload
	buffer_load_dword v6, off, s[0:3], s33 offset:32 ; 4-byte Folded Reload
	buffer_load_dword v7, off, s[0:3], s33 offset:36 ; 4-byte Folded Reload
	buffer_load_dword v8, off, s[0:3], s33 offset:40 ; 4-byte Folded Reload
	buffer_load_dword v9, off, s[0:3], s33 offset:44 ; 4-byte Folded Reload
	buffer_load_dword v10, off, s[0:3], s33 offset:48 ; 4-byte Folded Reload
	buffer_load_dword v11, off, s[0:3], s33 offset:52 ; 4-byte Folded Reload
	buffer_load_dword v12, off, s[0:3], s33 offset:56 ; 4-byte Folded Reload
	buffer_load_dword v13, off, s[0:3], s33 offset:60 ; 4-byte Folded Reload
	buffer_load_dword v14, off, s[0:3], s33 offset:64 ; 4-byte Folded Reload
	buffer_load_dword v15, off, s[0:3], s33 offset:68 ; 4-byte Folded Reload
	buffer_load_dword v16, off, s[0:3], s33 offset:72 ; 4-byte Folded Reload
	buffer_load_dword v17, off, s[0:3], s33 offset:76 ; 4-byte Folded Reload
	buffer_load_dword v18, off, s[0:3], s33 offset:80 ; 4-byte Folded Reload
	buffer_load_dword v19, off, s[0:3], s33 offset:84 ; 4-byte Folded Reload
	buffer_load_dword v20, off, s[0:3], s33 offset:88 ; 4-byte Folded Reload
	buffer_load_dword v21, off, s[0:3], s33 offset:92 ; 4-byte Folded Reload
	buffer_load_dword v22, off, s[0:3], s33 offset:96 ; 4-byte Folded Reload
	buffer_load_dword v23, off, s[0:3], s33 offset:100 ; 4-byte Folded Reload
	buffer_load_dword v24, off, s[0:3], s33 offset:104 ; 4-byte Folded Reload
	buffer_load_dword v25, off, s[0:3], s33 offset:108 ; 4-byte Folded Reload
	buffer_load_dword v26, off, s[0:3], s33 offset:112 ; 4-byte Folded Reload
	buffer_load_dword v27, off, s[0:3], s33 offset:116 ; 4-byte Folded Reload
	buffer_load_dword v28, off, s[0:3], s33 offset:120 ; 4-byte Folded Reload
	buffer_load_dword v29, off, s[0:3], s33 offset:124 ; 4-byte Folded Reload
	buffer_load_dword v30, off, s[0:3], s33 offset:128 ; 4-byte Folded Reload
	buffer_load_dword v31, off, s[0:3], s33 offset:132 ; 4-byte Folded Reload
	buffer_load_dword v32, off, s[0:3], s33 offset:136 ; 4-byte Folded Reload
	buffer_load_dword v33, off, s[0:3], s33 offset:140 ; 4-byte Folded Reload
	v_readlane_b32 s31, v41, 1
	v_readlane_b32 s30, v41, 0
                                        ; kill: killed $vgpr40
	v_readlane_b32 s34, v41, 2
	v_readlane_b32 s35, v41, 3
	v_readlane_b32 s4, v41, 4
	s_waitcnt vmcnt(0)
	flat_store_dwordx4 v[0:1], v[30:33] offset:112
	s_waitcnt vmcnt(0)
	flat_store_dwordx4 v[0:1], v[26:29] offset:96
	s_waitcnt vmcnt(0)
	flat_store_dwordx4 v[0:1], v[22:25] offset:80
	s_waitcnt vmcnt(0)
	flat_store_dwordx4 v[0:1], v[18:21] offset:64
	s_waitcnt vmcnt(0)
	flat_store_dwordx4 v[0:1], v[14:17] offset:48
	s_waitcnt vmcnt(0)
	flat_store_dwordx4 v[0:1], v[10:13] offset:32
	s_waitcnt vmcnt(0)
	flat_store_dwordx4 v[0:1], v[6:9] offset:16
	s_waitcnt vmcnt(0)
	flat_store_dwordx4 v[0:1], v[2:5]
	s_waitcnt vmcnt(0)
	s_xor_saveexec_b64 s[6:7], -1
	buffer_load_dword v33, off, s[0:3], s33 offset:144 ; 4-byte Folded Reload
	buffer_load_dword v2, off, s[0:3], s33 offset:152 ; 4-byte Folded Reload
	s_mov_b64 exec, -1
	buffer_load_dword v41, off, s[0:3], s33 ; 4-byte Folded Reload
	buffer_load_dword v40, off, s[0:3], s33 offset:148 ; 4-byte Folded Reload
	s_mov_b64 exec, s[6:7]
	s_addk_i32 s32, 0xd800
	s_mov_b32 s33, s4
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	preserve_wwm_copy_dstreg, .Lfunc_end0-preserve_wwm_copy_dstreg
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 1688
; NumSgprs: 42
; NumVgprs: 42
; ScratchSize: 160
; MemoryBound: 0
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:  []
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
