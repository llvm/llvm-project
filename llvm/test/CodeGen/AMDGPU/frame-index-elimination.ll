; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -mattr=-promote-alloca -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-promote-alloca -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

; Test that non-entry function frame indices are expanded properly to
; give an index relative to the scratch wave offset register

; Materialize into a mov. Make sure there isn't an unnecessary copy.
; GCN-LABEL: {{^}}func_mov_fi_i32:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN: s_sub_u32 s6, s5, s4

; CI-NEXT: v_lshr_b32_e64 [[SCALED:v[0-9]+]], s6, 6
; CI-NEXT: v_add_i32_e64 v0, s[6:7], 4, [[SCALED]]

; GFX9-NEXT: v_lshrrev_b32_e64 [[SCALED:v[0-9]+]], 6, s6
; GFX9-NEXT: v_add_u32_e32 v0, 4, [[SCALED]]

; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @func_mov_fi_i32() #0 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 addrspace(5)* %alloca, i32 addrspace(5)* addrspace(3)* undef
  ret void
}

; Materialize into an add of a constant offset from the FI.
; FIXME: Should be able to merge adds

; GCN-LABEL: {{^}}func_add_constant_to_fi_i32:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN: s_sub_u32 s6, s5, s4

; CI-NEXT: v_lshr_b32_e64 [[SCALED:v[0-9]+]], s6, 6
; CI-NEXT: v_add_i32_e64 v0, s[6:7], 4, [[SCALED]]
; CI-NEXT: v_add_i32_e32 v0, vcc, 4, v0

; GFX9-NEXT: v_lshrrev_b32_e64 [[SCALED:v[0-9]+]], 6, s6
; GFX9-NEXT: v_add_u32_e32 v0, 4, [[SCALED]]
; GFX9-NEXT: v_add_u32_e32 v0, 4, v0


; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @func_add_constant_to_fi_i32() #0 {
  %alloca = alloca [2 x i32], align 4, addrspace(5)
  %gep0 = getelementptr inbounds [2 x i32], [2 x i32] addrspace(5)* %alloca, i32 0, i32 1
  store volatile i32 addrspace(5)* %gep0, i32 addrspace(5)* addrspace(3)* undef
  ret void
}

; A user the materialized frame index can't be meaningfully folded
; into.

; GCN-LABEL: {{^}}func_other_fi_user_i32:
; GCN: s_sub_u32 s6, s5, s4

; CI-NEXT: v_lshr_b32_e64 [[SCALED:v[0-9]+]], s6, 6
; CI-NEXT: v_add_i32_e64 v0, s[6:7], 4, [[SCALED]]

; GFX9-NEXT: v_lshrrev_b32_e64 [[SCALED:v[0-9]+]], 6, s6
; GFX9-NEXT: v_add_u32_e32 v0, 4, [[SCALED]]

; GCN-NEXT: v_mul_lo_i32 v0, v0, 9
; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @func_other_fi_user_i32() #0 {
  %alloca = alloca [2 x i32], align 4, addrspace(5)
  %ptrtoint = ptrtoint [2 x i32] addrspace(5)* %alloca to i32
  %mul = mul i32 %ptrtoint, 9
  store volatile i32 %mul, i32 addrspace(3)* undef
  ret void
}

; GCN-LABEL: {{^}}func_store_private_arg_i32_ptr:
; GCN: v_mov_b32_e32 v1, 15{{$}}
; GCN: buffer_store_dword v1, v0, s[0:3], s4 offen{{$}}
define void @func_store_private_arg_i32_ptr(i32 addrspace(5)* %ptr) #0 {
  store volatile i32 15, i32 addrspace(5)* %ptr
  ret void
}

; GCN-LABEL: {{^}}func_load_private_arg_i32_ptr:
; GCN: s_waitcnt
; GCN-NEXT: buffer_load_dword v0, v0, s[0:3], s4 offen{{$}}
define void @func_load_private_arg_i32_ptr(i32 addrspace(5)* %ptr) #0 {
  %val = load volatile i32, i32 addrspace(5)* %ptr
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32_ptr:
; GCN: s_waitcnt
; GCN-NEXT: s_mov_b32 s5, s32
; GCN-NEXT: s_sub_u32 [[SUB_OFFSET:s[0-9]+]], s5, s4

; CI-NEXT: v_lshr_b32_e64 [[SHIFT:v[0-9]+]], [[SUB_OFFSET]], 6
; CI-NEXT: v_add_i32_e64 [[ADD:v[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 4, [[SHIFT]]
; CI-NEXT: v_add_i32_e32 v0, vcc, 4, [[ADD]]

; GFX9-NEXT: v_lshrrev_b32_e64 [[SHIFT:v[0-9]+]], 6, [[SUB_OFFSET]]
; GFX9-NEXT: v_add_u32_e32 [[ADD:v[0-9]+]], 4, [[SHIFT]]
; GFX9-NEXT: v_add_u32_e32 v0, 4, [[ADD]]

; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @void_func_byval_struct_i8_i32_ptr({ i8, i32 } addrspace(5)* byval %arg0) #0 {
  %gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %arg0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %arg0, i32 0, i32 1
  %load1 = load i32, i32 addrspace(5)* %gep1
  store volatile i32 addrspace(5)* %gep1, i32 addrspace(5)* addrspace(3)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32_ptr_value:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: s_mov_b32 s5, s32
; GCN-NEXT: buffer_load_ubyte v0, off, s[0:3], s5
; GCN_NEXT: buffer_load_dword v1, off, s[0:3], s5 offset:4
define void @void_func_byval_struct_i8_i32_ptr_value({ i8, i32 } addrspace(5)* byval %arg0) #0 {
  %gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %arg0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %arg0, i32 0, i32 1
  %load0 = load i8, i8 addrspace(5)* %gep0
  %load1 = load i32, i32 addrspace(5)* %gep1
  store volatile i8 %load0, i8 addrspace(3)* undef
  store volatile i32 %load1, i32 addrspace(3)* undef
  ret void
}

; FIXME: Should be able to see that this can use vaddr, but the
; FrameIndex is hidden behind a CopyFromReg in the second block.

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32_ptr_nonentry_block:
; GCN: s_sub_u32 [[SUB_OFFSET:s[0-9]+]], s5, s4

; CI: v_lshr_b32_e64 [[SHIFT:v[0-9]+]], [[SUB_OFFSET]], 6
; CI: v_add_i32_e64 [[ADD:v[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 4, [[SHIFT]]

; GFX9: v_lshrrev_b32_e64 [[SHIFT:v[0-9]+]], 6, [[SUB_OFFSET]]
; GFX9: v_add_u32_e32 [[ADD:v[0-9]+]], 4, [[SHIFT]]

; GCN: s_and_saveexec_b64

; CI: v_add_i32_e32 v0, vcc, 4, [[ADD]]
; CI: buffer_load_dword v1, v0, s[0:3], s4 offen{{$}}

; GFX9: v_add_u32_e32 v0, 4, [[ADD]]
; GFX9: buffer_load_dword v1, v{{[0-9]+}}, s[0:3], s4 offen offset:4{{$}}

; GCN: ds_write_b32
define void @void_func_byval_struct_i8_i32_ptr_nonentry_block({ i8, i32 } addrspace(5)* byval %arg0, i32 %arg2) #0 {
  %cmp = icmp eq i32 %arg2, 0
  br i1 %cmp, label %bb, label %ret

bb:
  %gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %arg0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %arg0, i32 0, i32 1
  %load1 = load volatile i32, i32 addrspace(5)* %gep1
  store volatile i32 addrspace(5)* %gep1, i32 addrspace(5)* addrspace(3)* undef
  br label %ret

ret:
  ret void
}

; Added offset can't be used with VOP3 add
; GCN-LABEL: {{^}}func_other_fi_user_non_inline_imm_offset_i32:
; GCN: s_sub_u32 s6, s5, s4
; GCN-DAG: s_movk_i32 s6, 0x204

; CI-DAG: v_lshr_b32_e64 [[SCALED:v[0-9]+]], s6, 6
; CI: v_add_i32_e64 v0, s[6:7], s6, [[SCALED]]

; GFX9-DAG: v_lshrrev_b32_e64 [[SCALED:v[0-9]+]], 6, s6
; GFX9: v_add_u32_e32 v0, s6, [[SCALED]]

; GCN: v_mul_lo_i32 v0, v0, 9
; GCN: ds_write_b32 v0, v0
define void @func_other_fi_user_non_inline_imm_offset_i32() #0 {
  %alloca0 = alloca [128 x i32], align 4, addrspace(5)
  %alloca1 = alloca [8 x i32], align 4, addrspace(5)
  %gep0 = getelementptr inbounds [128 x i32], [128 x i32] addrspace(5)* %alloca0, i32 0, i32 65
  %gep1 = getelementptr inbounds [8 x i32], [8 x i32] addrspace(5)* %alloca1, i32 0, i32 0
  store volatile i32 7, i32 addrspace(5)* %gep0
  %ptrtoint = ptrtoint i32 addrspace(5)* %gep1 to i32
  %mul = mul i32 %ptrtoint, 9
  store volatile i32 %mul, i32 addrspace(3)* undef
  ret void
}

; GCN-LABEL: {{^}}func_other_fi_user_non_inline_imm_offset_i32_vcc_live:
; GCN: s_sub_u32 [[DIFF:s[0-9]+]], s5, s4
; GCN-DAG: s_movk_i32 [[OFFSET:s[0-9]+]], 0x204

; CI-DAG: v_lshr_b32_e64 [[SCALED:v[0-9]+]], [[DIFF]], 6
; CI: v_add_i32_e64 v0, s{{\[[0-9]+:[0-9]+\]}}, [[OFFSET]], [[SCALED]]

; GFX9-DAG: v_lshrrev_b32_e64 [[SCALED:v[0-9]+]], 6, [[DIFF]]
; GFX9: v_add_u32_e32 v0, [[OFFSET]], [[SCALED]]

; GCN: v_mul_lo_i32 v0, v0, 9
; GCN: ds_write_b32 v0, v0
define void @func_other_fi_user_non_inline_imm_offset_i32_vcc_live() #0 {
  %alloca0 = alloca [128 x i32], align 4, addrspace(5)
  %alloca1 = alloca [8 x i32], align 4, addrspace(5)
  %vcc = call i64 asm sideeffect "; def $0", "={VCC}"()
  %gep0 = getelementptr inbounds [128 x i32], [128 x i32] addrspace(5)* %alloca0, i32 0, i32 65
  %gep1 = getelementptr inbounds [8 x i32], [8 x i32] addrspace(5)* %alloca1, i32 0, i32 0
  store volatile i32 7, i32 addrspace(5)* %gep0
  call void asm sideeffect "; use $0", "{VCC}"(i64 %vcc)
  %ptrtoint = ptrtoint i32 addrspace(5)* %gep1 to i32
  %mul = mul i32 %ptrtoint, 9
  store volatile i32 %mul, i32 addrspace(3)* undef
  ret void
}

declare void @func(<4 x float> addrspace(5)* nocapture) #0

; undef flag not preserved in eliminateFrameIndex when handling the
; stores in the middle block.

; GCN-LABEL: {{^}}undefined_stack_store_reg:
; GCN: s_and_saveexec_b64
; GCN: buffer_store_dword v0, off, s[0:3], s5 offset:
; GCN: buffer_store_dword v0, off, s[0:3], s5 offset:
; GCN: buffer_store_dword v0, off, s[0:3], s5 offset:
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s5 offset:
define void @undefined_stack_store_reg(float %arg, i32 %arg1) #0 {
bb:
  %tmp = alloca <4 x float>, align 16, addrspace(5)
  %tmp2 = insertelement <4 x float> undef, float %arg, i32 0
  store <4 x float> %tmp2, <4 x float> addrspace(5)* undef
  %tmp3 = icmp eq i32 %arg1, 0
  br i1 %tmp3, label %bb4, label %bb5

bb4:
  call void @func(<4 x float> addrspace(5)* nonnull undef)
  store <4 x float> %tmp2, <4 x float> addrspace(5)* %tmp, align 16
  call void @func(<4 x float> addrspace(5)* nonnull %tmp)
  br label %bb5

bb5:
  ret void
}

; GCN-LABEL: {{^}}alloca_ptr_nonentry_block:
; GCN: s_and_saveexec_b64
; GCN: buffer_load_dword v{{[0-9]+}}, off, s[0:3], s5 offset:12
define void @alloca_ptr_nonentry_block(i32 %arg0) #0 {
  %alloca0 = alloca { i8, i32 }, align 4, addrspace(5)
  %cmp = icmp eq i32 %arg0, 0
  br i1 %cmp, label %bb, label %ret

bb:
  %gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %alloca0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %alloca0, i32 0, i32 1
  %load1 = load volatile i32, i32 addrspace(5)* %gep1
  store volatile i32 addrspace(5)* %gep1, i32 addrspace(5)* addrspace(3)* undef
  br label %ret

ret:
  ret void
}

attributes #0 = { nounwind }
