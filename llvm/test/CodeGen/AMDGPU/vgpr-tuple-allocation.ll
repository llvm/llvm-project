; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX9 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX10 %s

declare void @extern_func()

define <4 x float> @non_preserved_vgpr_tuple8(<8 x i32> %rsrc, <4 x i32> %samp, float %bias, float %zcompare, float %s, float %t, float %clamp) {
; The vgpr tuple8 operand in image_gather4_c_b_cl instruction needs not be
; preserved across the call and should get 8 scratch registers.

; GFX9-LABEL: non_preserved_vgpr_tuple8:
; GFX9: buffer_store_dword v44, off, s[0:3], s32 offset:16 ; 4-byte Folded Spill
; GFX9: buffer_store_dword v40, off, s[0:3], s33 offset:12 ; 4-byte Folded Spill
; GFX9-NEXT: buffer_store_dword v41, off, s[0:3], s33 offset:8 ; 4-byte Folded Spill
; GFX9-NEXT: buffer_store_dword v42, off, s[0:3], s33 offset:4 ; 4-byte Folded Spill
; GFX9-NEXT: buffer_store_dword v43, off, s[0:3], s33 ; 4-byte Folded Spill

; GFX9: v_writelane_b32 v44, s30, 0
; GFX9-NEXT: v_mov_b32_e32 v36, v16
; GFX9-NEXT: v_mov_b32_e32 v35, v15
; GFX9-NEXT: v_mov_b32_e32 v34, v14
; GFX9-NEXT: v_mov_b32_e32 v33, v13
; GFX9-NEXT: v_mov_b32_e32 v32, v12

; GFX9: ;;#ASMSTART
; GFX9-NEXT: ;;#ASMEND

; GFX9: image_gather4_c_b_cl v[40:43], v[32:39], s[4:11], s[4:7] dmask:0x1
; GFX9-NEXT: s_getpc_b64 s[4:5]
; GFX9-NEXT: s_add_u32 s4, s4, extern_func@gotpcrel32@lo+4
; GFX9-NEXT: s_addc_u32 s5, s5, extern_func@gotpcrel32@hi+12
; GFX9-NEXT: s_load_dwordx2 s[4:5], s[4:5], 0x0
; GFX9-NEXT: s_waitcnt lgkmcnt(0)
; GFX9-NEXT: s_swappc_b64 s[30:31], s[4:5]

; GFX9: buffer_load_dword v43, off, s[0:3], s33 ; 4-byte Folded Reload
; GFX9-NEXT: buffer_load_dword v42, off, s[0:3], s33 offset:4 ; 4-byte Folded Reload
; GFX9-NEXT: buffer_load_dword v41, off, s[0:3], s33 offset:8 ; 4-byte Folded Reload
; GFX9-NEXT: buffer_load_dword v40, off, s[0:3], s33 offset:12 ; 4-byte Folded Reload
; GFX9: buffer_load_dword v44, off, s[0:3], s32 offset:16 ; 4-byte Folded Reload
; GFX9: s_setpc_b64 s[4:5]
;
; GFX10-LABEL: non_preserved_vgpr_tuple8:
; GFX10: buffer_store_dword v44, off, s[0:3], s32 offset:16 ; 4-byte Folded Spill
; GFX10: buffer_store_dword v40, off, s[0:3], s33 offset:12 ; 4-byte Folded Spill
; GFX10-NEXT: buffer_store_dword v41, off, s[0:3], s33 offset:8 ; 4-byte Folded Spill
; GFX10-NEXT: buffer_store_dword v42, off, s[0:3], s33 offset:4 ; 4-byte Folded Spill
; GFX10-NEXT: buffer_store_dword v43, off, s[0:3], s33 ; 4-byte Folded Spill

; GFX10: v_mov_b32_e32 v36, v16
; GFX10-NEXT: v_mov_b32_e32 v35, v15
; GFX10-NEXT: v_mov_b32_e32 v34, v14
; GFX10-NEXT: v_mov_b32_e32 v33, v13
; GFX10-NEXT: v_mov_b32_e32 v32, v12

; GFX10: ;;#ASMSTART
; GFX10-NEXT: ;;#ASMEND

; GFX10: image_gather4_c_b_cl v[40:43], v[32:39], s[4:11], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
; GFX10-NEXT: s_waitcnt_depctr 0xffe3
; GFX10-NEXT: s_getpc_b64 s[4:5]
; GFX10-NEXT: s_add_u32 s4, s4, extern_func@gotpcrel32@lo+4
; GFX10-NEXT: s_addc_u32 s5, s5, extern_func@gotpcrel32@hi+12
; GFX10-NEXT:                                 ; implicit-def: $vcc_hi
; GFX10-NEXT: s_load_dwordx2 s[4:5], s[4:5], 0x0
; GFX10-NEXT: s_waitcnt lgkmcnt(0)
; GFX10-NEXT: s_swappc_b64 s[30:31], s[4:5]

; GFX10: buffer_load_dword v43, off, s[0:3], s33
; GFX10-NEXT: buffer_load_dword v42, off, s[0:3], s33 offset:4
; GFX10-NEXT: buffer_load_dword v41, off, s[0:3], s33 offset:8
; GFX10-NEXT: buffer_load_dword v40, off, s[0:3], s33 offset:12

; GFX10: buffer_load_dword v44, off, s[0:3], s32 offset:16 ; 4-byte Folded Reload
; GFX10: s_setpc_b64 s[4:5]
main_body:
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}"() #0
  call void asm sideeffect "", "~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}"() #0
  call void asm sideeffect "", "~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23}"() #0
  call void asm sideeffect "", "~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"() #0
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.2d.v4f32.f32.f32(i32 1, float %bias, float %zcompare, float %s, float %t, float %clamp, <8 x i32> undef, <4 x i32> undef, i1 false, i32 0, i32 0)
  call void @extern_func()
  ret <4 x float> %v
}

define <4 x float> @call_preserved_vgpr_tuple8(<8 x i32> %rsrc, <4 x i32> %samp, float %bias, float %zcompare, float %s, float %t, float %clamp) {
; The vgpr tuple8 operand in image_gather4_c_b_cl instruction needs to be preserved
; across the call and should get allcoated to 8 CSRs.
; Only the lower 5 sub-registers of the tuple are preserved.
; The upper 3 sub-registers are unused.

; GFX9-LABEL: call_preserved_vgpr_tuple8:
; GFX9: buffer_store_dword v56, off, s[0:3], s32 offset:20 ; 4-byte Folded Spill
; GFX9: buffer_store_dword v40, off, s[0:3], s33 offset:16 ; 4-byte Folded Spill
; GFX9-NEXT: buffer_store_dword v41, off, s[0:3], s33 offset:12 ; 4-byte Folded Spill
; GFX9-NEXT: buffer_store_dword v42, off, s[0:3], s33 offset:8 ; 4-byte Folded Spill
; GFX9-NEXT: buffer_store_dword v43, off, s[0:3], s33 offset:4 ; 4-byte Folded Spill
; GFX9-NEXT: buffer_store_dword v44, off, s[0:3], s33 ; 4-byte Folded Spill

; GFX9: v_mov_b32_e32 v44, v16
; GFX9-NEXT: v_mov_b32_e32 v43, v15
; GFX9-NEXT: v_mov_b32_e32 v42, v14
; GFX9-NEXT: v_mov_b32_e32 v41, v13
; GFX9-NEXT: v_mov_b32_e32 v40, v12

; GFX9: image_gather4_c_b_cl v[0:3], v[40:47], s[36:43], s[4:7] dmask:0x1
; GFX9-NEXT: s_getpc_b64 s[4:5]
; GFX9-NEXT: s_add_u32 s4, s4, extern_func@gotpcrel32@lo+4
; GFX9-NEXT: s_addc_u32 s5, s5, extern_func@gotpcrel32@hi+12
; GFX9-NEXT: s_load_dwordx2 s[4:5], s[4:5], 0x0
; GFX9-NEXT: s_waitcnt vmcnt(0)
; GFX9-NEXT: global_store_dwordx4 v[0:1], v[0:3], off
; GFX9-NEXT: s_waitcnt lgkmcnt(0)
; GFX9-NEXT: s_swappc_b64 s[30:31], s[4:5]
; GFX9-NEXT: image_gather4_c_b_cl v[0:3], v[40:47], s[36:43], s[4:7] dmask:0x1

; GFX9: buffer_load_dword v44, off, s[0:3], s33 ; 4-byte Folded Reload
; GFX9-NEXT: buffer_load_dword v43, off, s[0:3], s33 offset:4 ; 4-byte Folded Reload
; GFX9-NEXT: buffer_load_dword v42, off, s[0:3], s33 offset:8 ; 4-byte Folded Reload
; GFX9-NEXT: buffer_load_dword v41, off, s[0:3], s33 offset:12 ; 4-byte Folded Reload
; GFX9-NEXT: buffer_load_dword v40, off, s[0:3], s33 offset:16 ; 4-byte Folded Reload

; GFX9: buffer_load_dword v56, off, s[0:3], s32 offset:20 ; 4-byte Folded Reload
; GFX9: s_setpc_b64 s[4:5]
;
; GFX10-LABEL: call_preserved_vgpr_tuple8:
; GFX10: buffer_store_dword v45, off, s[0:3], s32 offset:20 ; 4-byte Folded Spill
; GFX10: buffer_store_dword v40, off, s[0:3], s33 offset:16 ; 4-byte Folded Spill
; GFX10-NEXT: buffer_store_dword v41, off, s[0:3], s33 offset:12 ; 4-byte Folded Spill
; GFX10-NEXT: buffer_store_dword v42, off, s[0:3], s33 offset:8 ; 4-byte Folded Spill
; GFX10-NEXT: buffer_store_dword v43, off, s[0:3], s33 offset:4 ; 4-byte Folded Spill
; GFX10-NEXT: buffer_store_dword v44, off, s[0:3], s33 ; 4-byte Folded Spill


; GFX10:      image_gather4_c_b_cl v[0:3], v[12:19], s[36:43], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
; GFX10-NEXT: s_waitcnt_depctr 0xffe3
; GFX10-NEXT: s_getpc_b64 s[4:5]
; GFX10-NEXT: s_add_u32 s4, s4, extern_func@gotpcrel32@lo+4
; GFX10-NEXT: s_addc_u32 s5, s5, extern_func@gotpcrel32@hi+12
; GFX10-NEXT: v_mov_b32_e32 v41, v15
; GFX10-NEXT: s_load_dwordx2 s[4:5], s[4:5], 0x0
; GFX10-NEXT: v_mov_b32_e32 v42, v14
; GFX10-NEXT: v_mov_b32_e32 v43, v13
; GFX10-NEXT: v_mov_b32_e32 v44, v12
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GFX10-NEXT: s_waitcnt vmcnt(0)
; GFX10-NEXT: global_store_dwordx4 v[0:1], v[0:3], off
; GFX10-NEXT: s_waitcnt lgkmcnt(0)
; GFX10-NEXT: s_swappc_b64 s[30:31], s[4:5]
; GFX10-NEXT: image_gather4_c_b_cl v[0:3], [v44, v43, v42, v41, v40], s[36:43], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_2D

; GFX10: buffer_load_dword v44, off, s[0:3], s33
; GFX10-NEXT: buffer_load_dword v43, off, s[0:3], s33 offset:4
; GFX10-NEXT: buffer_load_dword v42, off, s[0:3], s33 offset:8
; GFX10-NEXT: buffer_load_dword v41, off, s[0:3], s33 offset:12
; GFX10-NEXT: buffer_load_dword v40, off, s[0:3], s33 offset:16
; GFX10: buffer_load_dword v45, off, s[0:3], s32 offset:20
; GFX10: s_setpc_b64 s[4:5]
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.2d.v4f32.f32.f32(i32 1, float %bias, float %zcompare, float %s, float %t, float %clamp, <8 x i32> undef, <4 x i32> undef, i1 false, i32 0, i32 0)
  store <4 x float> %v, <4 x float> addrspace(1)* undef
  call void @extern_func()
  %v1 = call <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.2d.v4f32.f32.f32(i32 1, float %bias, float %zcompare, float %s, float %t, float %clamp, <8 x i32> undef, <4 x i32> undef, i1 false, i32 0, i32 0)
  ret <4 x float> %v1
}

declare <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.2d.v4f32.f32.f32(i32 immarg, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

attributes #0 = { nounwind writeonly }
attributes #1 = { nounwind readonly }
