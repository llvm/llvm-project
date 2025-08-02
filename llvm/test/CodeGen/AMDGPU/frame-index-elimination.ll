; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -mattr=-promote-alloca < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI,MUBUF %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-promote-alloca < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,GFX9-MUBUF,MUBUF %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-promote-alloca,+enable-flat-scratch < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,GFX9-FLATSCR %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -mattr=+real-true16 < %s | FileCheck --check-prefixes=GFX11-TRUE16 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -mattr=-real-true16 < %s | FileCheck --check-prefixes=GFX11-FAKE16 %s

; Test that non-entry function frame indices are expanded properly to
; give an index relative to the scratch wave offset register

; Materialize into a mov. Make sure there isn't an unnecessary copy.
; GCN-LABEL: {{^}}func_mov_fi_i32:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)

; CI-NEXT: v_lshr_b32_e64 v0, s32, 6
; GFX9-MUBUF-NEXT: v_lshrrev_b32_e64 v0, 6, s32

; GFX9-FLATSCR:     v_mov_b32_e32 v0, s32
; GFX9-FLATSCR-NOT: v_lshrrev_b32_e64

; MUBUF-NOT: v_mov

; GCN: ds_write_b32 v0, v0
define void @func_mov_fi_i32() #0 {
; CI-LABEL: func_mov_fi_i32:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    v_lshr_b32_e64 v0, s32, 6
; CI-NEXT:    s_mov_b32 m0, -1
; CI-NEXT:    ds_write_b32 v0, v0
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: func_mov_fi_i32:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    v_lshrrev_b32_e64 v0, 6, s32
; GFX9-MUBUF-NEXT:    ds_write_b32 v0, v0
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: func_mov_fi_i32:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v0, s32
; GFX9-FLATSCR-NEXT:    ds_write_b32 v0, v0
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: func_mov_fi_i32:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v0, s32
; GFX11-TRUE16-NEXT:    ds_store_b32 v0, v0
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: func_mov_fi_i32:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    v_mov_b32_e32 v0, s32
; GFX11-FAKE16-NEXT:    ds_store_b32 v0, v0
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
  %alloca = alloca i32, addrspace(5)
  store volatile ptr addrspace(5) %alloca, ptr addrspace(3) poison
  ret void
}

; Offset due to different objects
; GCN-LABEL: {{^}}func_mov_fi_i32_offset:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)

; CI-DAG: v_lshr_b32_e64 v0, s32, 6
; CI-NOT: v_mov
; CI: ds_write_b32 v0, v0
; CI-NEXT: v_lshr_b32_e64 [[SCALED:v[0-9]+]], s32, 6
; CI-NEXT: v_add_i32_e{{32|64}} v0, {{s\[[0-9]+:[0-9]+\]|vcc}}, 4, [[SCALED]]
; CI-NEXT: ds_write_b32 v0, v0

; GFX9-MUBUF-NEXT:   v_lshrrev_b32_e64 v0, 6, s32
; GFX9-FLATSCR:      v_mov_b32_e32 v0, s32
; GFX9-FLATSCR:      s_add_i32 [[ADD:[^,]+]], s32, 4
; GFX9-NEXT:         ds_write_b32 v0, v0
; GFX9-MUBUF-NEXT:   v_lshrrev_b32_e64 [[SCALED:v[0-9]+]], 6, s32
; GFX9-MUBUF-NEXT:   v_add_u32_e32 v0, 4, [[SCALED]]
; GFX9-FLATSCR-NEXT: v_mov_b32_e32 v0, [[ADD]]
; GFX9-NEXT:         ds_write_b32 v0, v0
define void @func_mov_fi_i32_offset() #0 {
; CI-LABEL: func_mov_fi_i32_offset:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    v_lshr_b32_e64 v0, s32, 6
; CI-NEXT:    s_mov_b32 m0, -1
; CI-NEXT:    ds_write_b32 v0, v0
; CI-NEXT:    v_lshr_b32_e64 v0, s32, 6
; CI-NEXT:    v_add_i32_e32 v0, vcc, 4, v0
; CI-NEXT:    ds_write_b32 v0, v0
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: func_mov_fi_i32_offset:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    v_lshrrev_b32_e64 v0, 6, s32
; GFX9-MUBUF-NEXT:    ds_write_b32 v0, v0
; GFX9-MUBUF-NEXT:    v_lshrrev_b32_e64 v0, 6, s32
; GFX9-MUBUF-NEXT:    v_add_u32_e32 v0, 4, v0
; GFX9-MUBUF-NEXT:    ds_write_b32 v0, v0
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: func_mov_fi_i32_offset:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v0, s32
; GFX9-FLATSCR-NEXT:    s_add_i32 s0, s32, 4
; GFX9-FLATSCR-NEXT:    ds_write_b32 v0, v0
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-FLATSCR-NEXT:    ds_write_b32 v0, v0
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: func_mov_fi_i32_offset:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_add_i32 s0, s32, 4
; GFX11-TRUE16-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-TRUE16-NEXT:    v_dual_mov_b32 v0, s32 :: v_dual_mov_b32 v1, s0
; GFX11-TRUE16-NEXT:    ds_store_b32 v0, v0
; GFX11-TRUE16-NEXT:    ds_store_b32 v0, v1
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: func_mov_fi_i32_offset:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_add_i32 s0, s32, 4
; GFX11-FAKE16-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-FAKE16-NEXT:    v_dual_mov_b32 v0, s32 :: v_dual_mov_b32 v1, s0
; GFX11-FAKE16-NEXT:    ds_store_b32 v0, v0
; GFX11-FAKE16-NEXT:    ds_store_b32 v0, v1
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
  %alloca0 = alloca i32, addrspace(5)
  %alloca1 = alloca i32, addrspace(5)
  store volatile ptr addrspace(5) %alloca0, ptr addrspace(3) poison
  store volatile ptr addrspace(5) %alloca1, ptr addrspace(3) poison
  ret void
}

; Materialize into an add of a constant offset from the FI.
; FIXME: Should be able to merge adds

; GCN-LABEL: {{^}}func_add_constant_to_fi_i32:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)

; CI: v_lshr_b32_e64 [[SCALED:v[0-9]+]], s32, 6
; CI-NEXT: v_add_i32_e32 v0, vcc, 4, [[SCALED]]

; GFX9-MUBUF:       v_lshrrev_b32_e64 [[SCALED:v[0-9]+]], 6, s32
; GFX9-MUBUF-NEXT:  v_add_u32_e32 v0, 4, [[SCALED]]

; FIXME: Should commute and shrink
; GFX9-FLATSCR: v_add_u32_e64 v0, 4, s32

; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @func_add_constant_to_fi_i32() #0 {
; CI-LABEL: func_add_constant_to_fi_i32:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    v_lshr_b32_e64 v1, s32, 6
; CI-NEXT:    v_add_i32_e32 v0, vcc, 4, v1
; CI-NEXT:    s_mov_b32 m0, -1
; CI-NEXT:    ds_write_b32 v0, v0
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: func_add_constant_to_fi_i32:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    v_lshrrev_b32_e64 v1, 6, s32
; GFX9-MUBUF-NEXT:    v_add_u32_e32 v0, 4, v1
; GFX9-MUBUF-NEXT:    ds_write_b32 v0, v0
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: func_add_constant_to_fi_i32:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    v_add_u32_e64 v0, 4, s32
; GFX9-FLATSCR-NEXT:    ds_write_b32 v0, v0
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: func_add_constant_to_fi_i32:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    v_add_nc_u32_e64 v0, 4, s32
; GFX11-TRUE16-NEXT:    ds_store_b32 v0, v0
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: func_add_constant_to_fi_i32:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    v_add_nc_u32_e64 v0, 4, s32
; GFX11-FAKE16-NEXT:    ds_store_b32 v0, v0
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
  %alloca = alloca [2 x i32], align 4, addrspace(5)
  %gep0 = getelementptr inbounds [2 x i32], ptr addrspace(5) %alloca, i32 0, i32 1
  store volatile ptr addrspace(5) %gep0, ptr addrspace(3) poison
  ret void
}

; A user the materialized frame index can't be meaningfully folded
; into.
; FIXME: Should use s_mul but the frame index always gets materialized into a
; vgpr

; GCN-LABEL: {{^}}func_other_fi_user_i32:
; MUBUF: s_lshr_b32 [[SCALED:s[0-9]+]], s32, 6
; MUBUF: s_mul_i32 [[MUL:s[0-9]+]], [[SCALED]], 9
; MUBUF: v_mov_b32_e32 v0, [[MUL]]

; GFX9-FLATSCR: s_mul_i32 [[MUL:s[0-9]+]], s32, 9
; GFX9-FLATSCR: v_mov_b32_e32 v0, [[MUL]]

; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @func_other_fi_user_i32() #0 {
; CI-LABEL: func_other_fi_user_i32:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_lshr_b32 s5, s32, 6
; CI-NEXT:    s_mul_i32 s4, s5, 9
; CI-NEXT:    v_mov_b32_e32 v0, s4
; CI-NEXT:    s_mov_b32 m0, -1
; CI-NEXT:    ds_write_b32 v0, v0
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: func_other_fi_user_i32:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_lshr_b32 s5, s32, 6
; GFX9-MUBUF-NEXT:    s_mul_i32 s4, s5, 9
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v0, s4
; GFX9-MUBUF-NEXT:    ds_write_b32 v0, v0
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: func_other_fi_user_i32:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_mul_i32 s0, s32, 9
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-FLATSCR-NEXT:    ds_write_b32 v0, v0
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: func_other_fi_user_i32:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_mul_i32 s0, s32, 9
; GFX11-TRUE16-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v0, s0
; GFX11-TRUE16-NEXT:    ds_store_b32 v0, v0
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: func_other_fi_user_i32:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_mul_i32 s0, s32, 9
; GFX11-FAKE16-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-FAKE16-NEXT:    v_mov_b32_e32 v0, s0
; GFX11-FAKE16-NEXT:    ds_store_b32 v0, v0
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
  %alloca = alloca [2 x i32], align 4, addrspace(5)
  %ptrtoint = ptrtoint ptr addrspace(5) %alloca to i32
  %mul = mul i32 %ptrtoint, 9
  store volatile i32 %mul, ptr addrspace(3) poison
  ret void
}

; GCN-LABEL: {{^}}func_store_private_arg_i32_ptr:
; GCN: v_mov_b32_e32 v1, 15{{$}}
; MUBUF:        buffer_store_dword v1, v0, s[0:3], 0 offen{{$}}
; GFX9-FLATSCR: scratch_store_dword v0, v1, off{{$}}
define void @func_store_private_arg_i32_ptr(ptr addrspace(5) %ptr) #0 {
; CI-LABEL: func_store_private_arg_i32_ptr:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    v_mov_b32_e32 v1, 15
; CI-NEXT:    buffer_store_dword v1, v0, s[0:3], 0 offen
; CI-NEXT:    s_waitcnt vmcnt(0)
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: func_store_private_arg_i32_ptr:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v1, 15
; GFX9-MUBUF-NEXT:    buffer_store_dword v1, v0, s[0:3], 0 offen
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0)
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: func_store_private_arg_i32_ptr:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v1, 15
; GFX9-FLATSCR-NEXT:    scratch_store_dword v0, v1, off
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: func_store_private_arg_i32_ptr:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v1, 15
; GFX11-TRUE16-NEXT:    scratch_store_b32 v0, v1, off dlc
; GFX11-TRUE16-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: func_store_private_arg_i32_ptr:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    v_mov_b32_e32 v1, 15
; GFX11-FAKE16-NEXT:    scratch_store_b32 v0, v1, off dlc
; GFX11-FAKE16-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
  store volatile i32 15, ptr addrspace(5) %ptr
  ret void
}

; GCN-LABEL: {{^}}func_load_private_arg_i32_ptr:
; GCN: s_waitcnt
; MUBUF-NEXT:        buffer_load_dword v0, v0, s[0:3], 0 offen glc{{$}}
; GFX9-FLATSCR-NEXT: scratch_load_dword v0, v0, off glc{{$}}
define void @func_load_private_arg_i32_ptr(ptr addrspace(5) %ptr) #0 {
; CI-LABEL: func_load_private_arg_i32_ptr:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    buffer_load_dword v0, v0, s[0:3], 0 offen glc
; CI-NEXT:    s_waitcnt vmcnt(0)
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: func_load_private_arg_i32_ptr:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    buffer_load_dword v0, v0, s[0:3], 0 offen glc
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0)
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: func_load_private_arg_i32_ptr:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    scratch_load_dword v0, v0, off glc
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: func_load_private_arg_i32_ptr:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    scratch_load_b32 v0, v0, off glc dlc
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: func_load_private_arg_i32_ptr:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    scratch_load_b32 v0, v0, off glc dlc
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
  %val = load volatile i32, ptr addrspace(5) %ptr
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32_ptr:
; GCN: s_waitcnt

; CI: v_lshr_b32_e64 [[SHIFT:v[0-9]+]], s32, 6
; CI-NEXT: v_or_b32_e32 v0, 4, [[SHIFT]]

; GFX9-MUBUF:      v_lshrrev_b32_e64 [[SHIFT:v[0-9]+]], 6, s32
; GFX9-MUBUF-NEXT: v_or_b32_e32 v0, 4, [[SHIFT]]

; GFX9-FLATSCR: v_or_b32_e64 v0, s32, 4

; GCN-NOT: v_mov
; GCN: ds_write_b32 v0, v0
define void @void_func_byval_struct_i8_i32_ptr(ptr addrspace(5) byval({ i8, i32 }) %arg0) #0 {
; CI-LABEL: void_func_byval_struct_i8_i32_ptr:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    v_lshr_b32_e64 v1, s32, 6
; CI-NEXT:    v_or_b32_e32 v0, 4, v1
; CI-NEXT:    s_mov_b32 m0, -1
; CI-NEXT:    ds_write_b32 v0, v0
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: void_func_byval_struct_i8_i32_ptr:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    v_lshrrev_b32_e64 v1, 6, s32
; GFX9-MUBUF-NEXT:    v_or_b32_e32 v0, 4, v1
; GFX9-MUBUF-NEXT:    ds_write_b32 v0, v0
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: void_func_byval_struct_i8_i32_ptr:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    v_or_b32_e64 v0, s32, 4
; GFX9-FLATSCR-NEXT:    ds_write_b32 v0, v0
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: void_func_byval_struct_i8_i32_ptr:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    v_or_b32_e64 v0, s32, 4
; GFX11-TRUE16-NEXT:    ds_store_b32 v0, v0
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: void_func_byval_struct_i8_i32_ptr:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    v_or_b32_e64 v0, s32, 4
; GFX11-FAKE16-NEXT:    ds_store_b32 v0, v0
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
  %gep0 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %arg0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %arg0, i32 0, i32 1
  %load1 = load i32, ptr addrspace(5) %gep1
  store volatile ptr addrspace(5) %gep1, ptr addrspace(3) poison
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32_ptr_value:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; MUBUF-NEXT: buffer_load_ubyte v0, off, s[0:3], s32
; MUBUF-NEXT: buffer_load_dword v1, off, s[0:3], s32 offset:4
; GFX9-FLATSCR-NEXT: scratch_load_ubyte v0, off, s32
; GFX9-FLATSCR-NEXT: scratch_load_dword v1, off, s32 offset:4
define void @void_func_byval_struct_i8_i32_ptr_value(ptr addrspace(5) byval({ i8, i32 }) %arg0) #0 {
; CI-LABEL: void_func_byval_struct_i8_i32_ptr_value:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    buffer_load_ubyte v0, off, s[0:3], s32
; CI-NEXT:    buffer_load_dword v1, off, s[0:3], s32 offset:4
; CI-NEXT:    s_mov_b32 m0, -1
; CI-NEXT:    s_waitcnt vmcnt(1)
; CI-NEXT:    ds_write_b8 v0, v0
; CI-NEXT:    s_waitcnt vmcnt(0)
; CI-NEXT:    ds_write_b32 v0, v1
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: void_func_byval_struct_i8_i32_ptr_value:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    buffer_load_ubyte v0, off, s[0:3], s32
; GFX9-MUBUF-NEXT:    buffer_load_dword v1, off, s[0:3], s32 offset:4
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(1)
; GFX9-MUBUF-NEXT:    ds_write_b8 v0, v0
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0)
; GFX9-MUBUF-NEXT:    ds_write_b32 v0, v1
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: void_func_byval_struct_i8_i32_ptr_value:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    scratch_load_ubyte v0, off, s32
; GFX9-FLATSCR-NEXT:    scratch_load_dword v1, off, s32 offset:4
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(1)
; GFX9-FLATSCR-NEXT:    ds_write_b8 v0, v0
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT:    ds_write_b32 v0, v1
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: void_func_byval_struct_i8_i32_ptr_value:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_clause 0x1
; GFX11-TRUE16-NEXT:    scratch_load_d16_u8 v0, off, s32
; GFX11-TRUE16-NEXT:    scratch_load_b32 v1, off, s32 offset:4
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(1)
; GFX11-TRUE16-NEXT:    ds_store_b8 v0, v0
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-TRUE16-NEXT:    ds_store_b32 v0, v1
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: void_func_byval_struct_i8_i32_ptr_value:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_clause 0x1
; GFX11-FAKE16-NEXT:    scratch_load_u8 v0, off, s32
; GFX11-FAKE16-NEXT:    scratch_load_b32 v1, off, s32 offset:4
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(1)
; GFX11-FAKE16-NEXT:    ds_store_b8 v0, v0
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-FAKE16-NEXT:    ds_store_b32 v0, v1
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
  %gep0 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %arg0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %arg0, i32 0, i32 1
  %load0 = load i8, ptr addrspace(5) %gep0
  %load1 = load i32, ptr addrspace(5) %gep1
  store volatile i8 %load0, ptr addrspace(3) poison
  store volatile i32 %load1, ptr addrspace(3) poison
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32_ptr_nonentry_block:

; GCN: s_and_saveexec_b64

; CI: buffer_load_dword v{{[0-9]+}}, off, s[0:3], s32 offset:4 glc{{$}}
; GFX9-MUBUF:   buffer_load_dword v{{[0-9]+}}, off, s[0:3], s32 offset:4 glc{{$}}
; GFX9-FLATSCR: scratch_load_dword v{{[0-9]+}}, off, s32 offset:4 glc{{$}}

; CI: v_lshr_b32_e64 [[SHIFT:v[0-9]+]], s32, 6
; CI: v_add_i32_e64 [[GEP:v[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 4, [[SHIFT]]

; GFX9-MUBUF: v_lshrrev_b32_e64 [[SP:v[0-9]+]], 6, s32
; GFX9-MUBUF: v_add_u32_e32 [[GEP:v[0-9]+]], 4, [[SP]]

; GFX9-FLATSCR: v_add_u32_e64 [[GEP:v[0-9]+]], 4, s32

; GCN: ds_write_b32 v{{[0-9]+}}, [[GEP]]
define void @void_func_byval_struct_i8_i32_ptr_nonentry_block(ptr addrspace(5) byval({ i8, i32 }) %arg0, i32 %arg2) #0 {
; CI-LABEL: void_func_byval_struct_i8_i32_ptr_nonentry_block:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; CI-NEXT:    s_and_saveexec_b64 s[4:5], vcc
; CI-NEXT:    s_cbranch_execz .LBB8_2
; CI-NEXT:  ; %bb.1: ; %bb
; CI-NEXT:    buffer_load_dword v0, off, s[0:3], s32 offset:4 glc
; CI-NEXT:    s_waitcnt vmcnt(0)
; CI-NEXT:    v_lshr_b32_e64 v1, s32, 6
; CI-NEXT:    v_add_i32_e64 v0, s[6:7], 4, v1
; CI-NEXT:    s_mov_b32 m0, -1
; CI-NEXT:    ds_write_b32 v0, v0
; CI-NEXT:  .LBB8_2: ; %ret
; CI-NEXT:    s_or_b64 exec, exec, s[4:5]
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: void_func_byval_struct_i8_i32_ptr_nonentry_block:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; GFX9-MUBUF-NEXT:    s_and_saveexec_b64 s[4:5], vcc
; GFX9-MUBUF-NEXT:    s_cbranch_execz .LBB8_2
; GFX9-MUBUF-NEXT:  ; %bb.1: ; %bb
; GFX9-MUBUF-NEXT:    buffer_load_dword v0, off, s[0:3], s32 offset:4 glc
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0)
; GFX9-MUBUF-NEXT:    v_lshrrev_b32_e64 v1, 6, s32
; GFX9-MUBUF-NEXT:    v_add_u32_e32 v0, 4, v1
; GFX9-MUBUF-NEXT:    ds_write_b32 v0, v0
; GFX9-MUBUF-NEXT:  .LBB8_2: ; %ret
; GFX9-MUBUF-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: void_func_byval_struct_i8_i32_ptr_nonentry_block:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; GFX9-FLATSCR-NEXT:    s_and_saveexec_b64 s[0:1], vcc
; GFX9-FLATSCR-NEXT:    s_cbranch_execz .LBB8_2
; GFX9-FLATSCR-NEXT:  ; %bb.1: ; %bb
; GFX9-FLATSCR-NEXT:    scratch_load_dword v0, off, s32 offset:4 glc
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT:    v_add_u32_e64 v0, 4, s32
; GFX9-FLATSCR-NEXT:    ds_write_b32 v0, v0
; GFX9-FLATSCR-NEXT:  .LBB8_2: ; %ret
; GFX9-FLATSCR-NEXT:    s_or_b64 exec, exec, s[0:1]
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: void_func_byval_struct_i8_i32_ptr_nonentry_block:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_mov_b32 s0, exec_lo
; GFX11-TRUE16-NEXT:    v_cmpx_eq_u32_e32 0, v0
; GFX11-TRUE16-NEXT:    s_cbranch_execz .LBB8_2
; GFX11-TRUE16-NEXT:  ; %bb.1: ; %bb
; GFX11-TRUE16-NEXT:    scratch_load_b32 v0, off, s32 offset:4 glc dlc
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-TRUE16-NEXT:    v_add_nc_u32_e64 v0, 4, s32
; GFX11-TRUE16-NEXT:    ds_store_b32 v0, v0
; GFX11-TRUE16-NEXT:  .LBB8_2: ; %ret
; GFX11-TRUE16-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: void_func_byval_struct_i8_i32_ptr_nonentry_block:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_mov_b32 s0, exec_lo
; GFX11-FAKE16-NEXT:    v_cmpx_eq_u32_e32 0, v0
; GFX11-FAKE16-NEXT:    s_cbranch_execz .LBB8_2
; GFX11-FAKE16-NEXT:  ; %bb.1: ; %bb
; GFX11-FAKE16-NEXT:    scratch_load_b32 v0, off, s32 offset:4 glc dlc
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-FAKE16-NEXT:    v_add_nc_u32_e64 v0, 4, s32
; GFX11-FAKE16-NEXT:    ds_store_b32 v0, v0
; GFX11-FAKE16-NEXT:  .LBB8_2: ; %ret
; GFX11-FAKE16-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
  %cmp = icmp eq i32 %arg2, 0
  br i1 %cmp, label %bb, label %ret

bb:
  %gep0 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %arg0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %arg0, i32 0, i32 1
  %load1 = load volatile i32, ptr addrspace(5) %gep1
  store volatile ptr addrspace(5) %gep1, ptr addrspace(3) poison
  br label %ret

ret:
  ret void
}

; Added offset can't be used with VOP3 add
; GCN-LABEL: {{^}}func_other_fi_user_non_inline_imm_offset_i32:

; MUBUF: s_lshr_b32 [[SCALED:s[0-9]+]], s32, 6
; MUBUF: s_addk_i32 [[SCALED]], 0x200

; MUBUF: s_mul_i32 [[Z:s[0-9]+]], [[SCALED]], 9
; MUBUF: v_mov_b32_e32 [[VZ:v[0-9]+]], [[Z]]

; GFX9-FLATSCR: s_add_i32 [[SZ:[^,]+]], s32, 0x200
; GFX9-FLATSCR: s_mul_i32 [[MUL:s[0-9]+]], [[SZ]], 9
; GFX9-FLATSCR: v_mov_b32_e32 [[VZ:v[0-9]+]], [[MUL]]

; GCN: ds_write_b32 v0, [[VZ]]
define void @func_other_fi_user_non_inline_imm_offset_i32() #0 {
; CI-LABEL: func_other_fi_user_non_inline_imm_offset_i32:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_lshr_b32 s5, s32, 6
; CI-NEXT:    s_addk_i32 s5, 0x200
; CI-NEXT:    v_mov_b32_e32 v0, 7
; CI-NEXT:    s_mul_i32 s4, s5, 9
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], s32 offset:260
; CI-NEXT:    s_waitcnt vmcnt(0)
; CI-NEXT:    v_mov_b32_e32 v0, s4
; CI-NEXT:    s_mov_b32 m0, -1
; CI-NEXT:    ds_write_b32 v0, v0
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: func_other_fi_user_non_inline_imm_offset_i32:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_lshr_b32 s5, s32, 6
; GFX9-MUBUF-NEXT:    s_addk_i32 s5, 0x200
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v0, 7
; GFX9-MUBUF-NEXT:    s_mul_i32 s4, s5, 9
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], s32 offset:260
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0)
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v0, s4
; GFX9-MUBUF-NEXT:    ds_write_b32 v0, v0
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: func_other_fi_user_non_inline_imm_offset_i32:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_add_i32 s1, s32, 0x200
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v0, 7
; GFX9-FLATSCR-NEXT:    s_mul_i32 s0, s1, 9
; GFX9-FLATSCR-NEXT:    scratch_store_dword off, v0, s32 offset:260
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-FLATSCR-NEXT:    ds_write_b32 v0, v0
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: func_other_fi_user_non_inline_imm_offset_i32:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_add_i32 s1, s32, 0x200
; GFX11-TRUE16-NEXT:    s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
; GFX11-TRUE16-NEXT:    s_mul_i32 s0, s1, 9
; GFX11-TRUE16-NEXT:    v_dual_mov_b32 v0, 7 :: v_dual_mov_b32 v1, s0
; GFX11-TRUE16-NEXT:    scratch_store_b32 off, v0, s32 offset:260 dlc
; GFX11-TRUE16-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11-TRUE16-NEXT:    ds_store_b32 v0, v1
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: func_other_fi_user_non_inline_imm_offset_i32:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_add_i32 s1, s32, 0x200
; GFX11-FAKE16-NEXT:    s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
; GFX11-FAKE16-NEXT:    s_mul_i32 s0, s1, 9
; GFX11-FAKE16-NEXT:    v_dual_mov_b32 v0, 7 :: v_dual_mov_b32 v1, s0
; GFX11-FAKE16-NEXT:    scratch_store_b32 off, v0, s32 offset:260 dlc
; GFX11-FAKE16-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11-FAKE16-NEXT:    ds_store_b32 v0, v1
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
  %alloca0 = alloca [128 x i32], align 4, addrspace(5)
  %alloca1 = alloca [8 x i32], align 4, addrspace(5)
  %gep0 = getelementptr inbounds [128 x i32], ptr addrspace(5) %alloca0, i32 0, i32 65
  store volatile i32 7, ptr addrspace(5) %gep0
  %ptrtoint = ptrtoint ptr addrspace(5) %alloca1 to i32
  %mul = mul i32 %ptrtoint, 9
  store volatile i32 %mul, ptr addrspace(3) poison
  ret void
}

; GCN-LABEL: {{^}}func_other_fi_user_non_inline_imm_offset_i32_vcc_live:

; MUBUF: s_lshr_b32 [[SCALED:s[0-9]+]], s32, 6
; MUBUF: s_addk_i32 [[SCALED]], 0x200
; MUBUF: s_mul_i32 [[Z:s[0-9]+]], [[SCALED]], 9
; MUBUF: v_mov_b32_e32 [[VZ:v[0-9]+]], [[Z]]

; GFX9-FLATSCR: s_add_i32 [[SZ:[^,]+]], s32, 0x200
; GFX9-FLATSCR: s_mul_i32 [[MUL:s[0-9]+]], [[SZ]], 9
; GFX9-FLATSCR: v_mov_b32_e32 [[VZ:v[0-9]+]], [[MUL]]

; GCN: ds_write_b32 v0, [[VZ]]
define void @func_other_fi_user_non_inline_imm_offset_i32_vcc_live() #0 {
; CI-LABEL: func_other_fi_user_non_inline_imm_offset_i32_vcc_live:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_lshr_b32 s5, s32, 6
; CI-NEXT:    s_addk_i32 s5, 0x200
; CI-NEXT:    v_mov_b32_e32 v0, 7
; CI-NEXT:    s_mul_i32 s4, s5, 9
; CI-NEXT:    ;;#ASMSTART
; CI-NEXT:    ; def vcc
; CI-NEXT:    ;;#ASMEND
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], s32 offset:260
; CI-NEXT:    s_waitcnt vmcnt(0)
; CI-NEXT:    v_mov_b32_e32 v0, s4
; CI-NEXT:    s_mov_b32 m0, -1
; CI-NEXT:    ;;#ASMSTART
; CI-NEXT:    ; use vcc
; CI-NEXT:    ;;#ASMEND
; CI-NEXT:    ds_write_b32 v0, v0
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: func_other_fi_user_non_inline_imm_offset_i32_vcc_live:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_lshr_b32 s5, s32, 6
; GFX9-MUBUF-NEXT:    s_addk_i32 s5, 0x200
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v0, 7
; GFX9-MUBUF-NEXT:    s_mul_i32 s4, s5, 9
; GFX9-MUBUF-NEXT:    ;;#ASMSTART
; GFX9-MUBUF-NEXT:    ; def vcc
; GFX9-MUBUF-NEXT:    ;;#ASMEND
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], s32 offset:260
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0)
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v0, s4
; GFX9-MUBUF-NEXT:    ;;#ASMSTART
; GFX9-MUBUF-NEXT:    ; use vcc
; GFX9-MUBUF-NEXT:    ;;#ASMEND
; GFX9-MUBUF-NEXT:    ds_write_b32 v0, v0
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: func_other_fi_user_non_inline_imm_offset_i32_vcc_live:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_add_i32 s1, s32, 0x200
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v0, 7
; GFX9-FLATSCR-NEXT:    s_mul_i32 s0, s1, 9
; GFX9-FLATSCR-NEXT:    ;;#ASMSTART
; GFX9-FLATSCR-NEXT:    ; def vcc
; GFX9-FLATSCR-NEXT:    ;;#ASMEND
; GFX9-FLATSCR-NEXT:    scratch_store_dword off, v0, s32 offset:260
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-FLATSCR-NEXT:    ;;#ASMSTART
; GFX9-FLATSCR-NEXT:    ; use vcc
; GFX9-FLATSCR-NEXT:    ;;#ASMEND
; GFX9-FLATSCR-NEXT:    ds_write_b32 v0, v0
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: func_other_fi_user_non_inline_imm_offset_i32_vcc_live:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_add_i32 s1, s32, 0x200
; GFX11-TRUE16-NEXT:    ;;#ASMSTART
; GFX11-TRUE16-NEXT:    ; def vcc
; GFX11-TRUE16-NEXT:    ;;#ASMEND
; GFX11-TRUE16-NEXT:    s_mul_i32 s0, s1, 9
; GFX11-TRUE16-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-TRUE16-NEXT:    v_dual_mov_b32 v0, 7 :: v_dual_mov_b32 v1, s0
; GFX11-TRUE16-NEXT:    scratch_store_b32 off, v0, s32 offset:260 dlc
; GFX11-TRUE16-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11-TRUE16-NEXT:    ;;#ASMSTART
; GFX11-TRUE16-NEXT:    ; use vcc
; GFX11-TRUE16-NEXT:    ;;#ASMEND
; GFX11-TRUE16-NEXT:    ds_store_b32 v0, v1
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: func_other_fi_user_non_inline_imm_offset_i32_vcc_live:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_add_i32 s1, s32, 0x200
; GFX11-FAKE16-NEXT:    ;;#ASMSTART
; GFX11-FAKE16-NEXT:    ; def vcc
; GFX11-FAKE16-NEXT:    ;;#ASMEND
; GFX11-FAKE16-NEXT:    s_mul_i32 s0, s1, 9
; GFX11-FAKE16-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-FAKE16-NEXT:    v_dual_mov_b32 v0, 7 :: v_dual_mov_b32 v1, s0
; GFX11-FAKE16-NEXT:    scratch_store_b32 off, v0, s32 offset:260 dlc
; GFX11-FAKE16-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11-FAKE16-NEXT:    ;;#ASMSTART
; GFX11-FAKE16-NEXT:    ; use vcc
; GFX11-FAKE16-NEXT:    ;;#ASMEND
; GFX11-FAKE16-NEXT:    ds_store_b32 v0, v1
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
  %alloca0 = alloca [128 x i32], align 4, addrspace(5)
  %alloca1 = alloca [8 x i32], align 4, addrspace(5)
  %vcc = call i64 asm sideeffect "; def $0", "={vcc}"()
  %gep0 = getelementptr inbounds [128 x i32], ptr addrspace(5) %alloca0, i32 0, i32 65
  store volatile i32 7, ptr addrspace(5) %gep0
  call void asm sideeffect "; use $0", "{vcc}"(i64 %vcc)
  %ptrtoint = ptrtoint ptr addrspace(5) %alloca1 to i32
  %mul = mul i32 %ptrtoint, 9
  store volatile i32 %mul, ptr addrspace(3) poison
  ret void
}

declare void @func(ptr addrspace(5) nocapture) #0

; undef flag not preserved in eliminateFrameIndex when handling the
; stores in the middle block.

; GCN-LABEL: {{^}}undefined_stack_store_reg:
; GCN: s_and_saveexec_b64
; MUBUF: buffer_store_dword v0, off, s[0:3], s33 offset:
; MUBUF: buffer_store_dword v0, off, s[0:3], s33 offset:
; MUBUF: buffer_store_dword v0, off, s[0:3], s33 offset:
; MUBUF: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s33 offset:
; FLATSCR: scratch_store_dword v0, off, s33 offset:
; FLATSCR: scratch_store_dword v0, off, s33 offset:
; FLATSCR: scratch_store_dword v0, off, s33 offset:
; FLATSCR: scratch_store_dword v{{[0-9]+}}, off, s33 offset:
define void @undefined_stack_store_reg(float %arg, i32 %arg1) #0 {
; CI-LABEL: undefined_stack_store_reg:
; CI:       ; %bb.0: ; %bb
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_mov_b32 s16, s33
; CI-NEXT:    s_mov_b32 s33, s32
; CI-NEXT:    s_or_saveexec_b64 s[18:19], -1
; CI-NEXT:    buffer_store_dword v42, off, s[0:3], s33 offset:32 ; 4-byte Folded Spill
; CI-NEXT:    s_mov_b64 exec, s[18:19]
; CI-NEXT:    v_writelane_b32 v42, s16, 18
; CI-NEXT:    v_writelane_b32 v42, s30, 0
; CI-NEXT:    v_writelane_b32 v42, s31, 1
; CI-NEXT:    v_writelane_b32 v42, s34, 2
; CI-NEXT:    v_writelane_b32 v42, s35, 3
; CI-NEXT:    v_writelane_b32 v42, s36, 4
; CI-NEXT:    v_writelane_b32 v42, s37, 5
; CI-NEXT:    v_writelane_b32 v42, s38, 6
; CI-NEXT:    v_writelane_b32 v42, s39, 7
; CI-NEXT:    v_writelane_b32 v42, s48, 8
; CI-NEXT:    v_writelane_b32 v42, s49, 9
; CI-NEXT:    v_writelane_b32 v42, s50, 10
; CI-NEXT:    v_writelane_b32 v42, s51, 11
; CI-NEXT:    v_writelane_b32 v42, s52, 12
; CI-NEXT:    v_writelane_b32 v42, s53, 13
; CI-NEXT:    v_writelane_b32 v42, s54, 14
; CI-NEXT:    v_writelane_b32 v42, s55, 15
; CI-NEXT:    buffer_store_dword v40, off, s[0:3], s33 offset:4 ; 4-byte Folded Spill
; CI-NEXT:    buffer_store_dword v41, off, s[0:3], s33 ; 4-byte Folded Spill
; CI-NEXT:    v_writelane_b32 v42, s64, 16
; CI-NEXT:    v_mov_b32_e32 v40, v0
; CI-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v1
; CI-NEXT:    s_addk_i32 s32, 0xc00
; CI-NEXT:    v_writelane_b32 v42, s65, 17
; CI-NEXT:    buffer_store_dword v0, v0, s[0:3], 0 offen
; CI-NEXT:    s_and_saveexec_b64 s[54:55], vcc
; CI-NEXT:    s_cbranch_execz .LBB11_2
; CI-NEXT:  ; %bb.1: ; %bb4
; CI-NEXT:    s_getpc_b64 s[16:17]
; CI-NEXT:    s_add_u32 s16, s16, func@gotpcrel32@lo+4
; CI-NEXT:    s_addc_u32 s17, s17, func@gotpcrel32@hi+12
; CI-NEXT:    s_load_dwordx2 s[64:65], s[16:17], 0x0
; CI-NEXT:    s_mov_b64 s[34:35], s[4:5]
; CI-NEXT:    s_mov_b64 s[36:37], s[6:7]
; CI-NEXT:    s_mov_b64 s[38:39], s[8:9]
; CI-NEXT:    s_mov_b64 s[48:49], s[10:11]
; CI-NEXT:    s_mov_b32 s50, s12
; CI-NEXT:    s_mov_b32 s51, s13
; CI-NEXT:    s_mov_b32 s52, s14
; CI-NEXT:    s_mov_b32 s53, s15
; CI-NEXT:    v_mov_b32_e32 v41, v31
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_swappc_b64 s[30:31], s[64:65]
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:28
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:24
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:20
; CI-NEXT:    buffer_store_dword v40, off, s[0:3], s33 offset:16
; CI-NEXT:    v_lshr_b32_e64 v0, s33, 6
; CI-NEXT:    s_mov_b64 s[4:5], s[34:35]
; CI-NEXT:    s_mov_b64 s[6:7], s[36:37]
; CI-NEXT:    s_mov_b64 s[8:9], s[38:39]
; CI-NEXT:    s_mov_b64 s[10:11], s[48:49]
; CI-NEXT:    s_mov_b32 s12, s50
; CI-NEXT:    s_mov_b32 s13, s51
; CI-NEXT:    s_mov_b32 s14, s52
; CI-NEXT:    s_mov_b32 s15, s53
; CI-NEXT:    v_mov_b32_e32 v31, v41
; CI-NEXT:    v_add_i32_e32 v0, vcc, 16, v0
; CI-NEXT:    s_swappc_b64 s[30:31], s[64:65]
; CI-NEXT:  .LBB11_2: ; %bb5
; CI-NEXT:    s_or_b64 exec, exec, s[54:55]
; CI-NEXT:    buffer_load_dword v41, off, s[0:3], s33 ; 4-byte Folded Reload
; CI-NEXT:    buffer_load_dword v40, off, s[0:3], s33 offset:4 ; 4-byte Folded Reload
; CI-NEXT:    v_readlane_b32 s65, v42, 17
; CI-NEXT:    v_readlane_b32 s64, v42, 16
; CI-NEXT:    v_readlane_b32 s55, v42, 15
; CI-NEXT:    v_readlane_b32 s54, v42, 14
; CI-NEXT:    v_readlane_b32 s53, v42, 13
; CI-NEXT:    v_readlane_b32 s52, v42, 12
; CI-NEXT:    v_readlane_b32 s51, v42, 11
; CI-NEXT:    v_readlane_b32 s50, v42, 10
; CI-NEXT:    v_readlane_b32 s49, v42, 9
; CI-NEXT:    v_readlane_b32 s48, v42, 8
; CI-NEXT:    v_readlane_b32 s39, v42, 7
; CI-NEXT:    v_readlane_b32 s38, v42, 6
; CI-NEXT:    v_readlane_b32 s37, v42, 5
; CI-NEXT:    v_readlane_b32 s36, v42, 4
; CI-NEXT:    v_readlane_b32 s35, v42, 3
; CI-NEXT:    v_readlane_b32 s34, v42, 2
; CI-NEXT:    v_readlane_b32 s31, v42, 1
; CI-NEXT:    v_readlane_b32 s30, v42, 0
; CI-NEXT:    s_mov_b32 s32, s33
; CI-NEXT:    v_readlane_b32 s4, v42, 18
; CI-NEXT:    s_or_saveexec_b64 s[6:7], -1
; CI-NEXT:    buffer_load_dword v42, off, s[0:3], s33 offset:32 ; 4-byte Folded Reload
; CI-NEXT:    s_mov_b64 exec, s[6:7]
; CI-NEXT:    s_mov_b32 s33, s4
; CI-NEXT:    s_waitcnt vmcnt(0)
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: undefined_stack_store_reg:
; GFX9-MUBUF:       ; %bb.0: ; %bb
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_mov_b32 s16, s33
; GFX9-MUBUF-NEXT:    s_mov_b32 s33, s32
; GFX9-MUBUF-NEXT:    s_or_saveexec_b64 s[18:19], -1
; GFX9-MUBUF-NEXT:    buffer_store_dword v42, off, s[0:3], s33 offset:32 ; 4-byte Folded Spill
; GFX9-MUBUF-NEXT:    s_mov_b64 exec, s[18:19]
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s16, 18
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s30, 0
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s31, 1
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s34, 2
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s35, 3
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s36, 4
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s37, 5
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s38, 6
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s39, 7
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s48, 8
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s49, 9
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s50, 10
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s51, 11
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s52, 12
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s53, 13
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s54, 14
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s55, 15
; GFX9-MUBUF-NEXT:    buffer_store_dword v40, off, s[0:3], s33 offset:4 ; 4-byte Folded Spill
; GFX9-MUBUF-NEXT:    buffer_store_dword v41, off, s[0:3], s33 ; 4-byte Folded Spill
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s64, 16
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v40, v0
; GFX9-MUBUF-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v1
; GFX9-MUBUF-NEXT:    s_addk_i32 s32, 0xc00
; GFX9-MUBUF-NEXT:    v_writelane_b32 v42, s65, 17
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, v0, s[0:3], 0 offen
; GFX9-MUBUF-NEXT:    s_and_saveexec_b64 s[54:55], vcc
; GFX9-MUBUF-NEXT:    s_cbranch_execz .LBB11_2
; GFX9-MUBUF-NEXT:  ; %bb.1: ; %bb4
; GFX9-MUBUF-NEXT:    s_getpc_b64 s[16:17]
; GFX9-MUBUF-NEXT:    s_add_u32 s16, s16, func@gotpcrel32@lo+4
; GFX9-MUBUF-NEXT:    s_addc_u32 s17, s17, func@gotpcrel32@hi+12
; GFX9-MUBUF-NEXT:    s_load_dwordx2 s[64:65], s[16:17], 0x0
; GFX9-MUBUF-NEXT:    s_mov_b64 s[34:35], s[4:5]
; GFX9-MUBUF-NEXT:    s_mov_b64 s[36:37], s[6:7]
; GFX9-MUBUF-NEXT:    s_mov_b64 s[38:39], s[8:9]
; GFX9-MUBUF-NEXT:    s_mov_b64 s[48:49], s[10:11]
; GFX9-MUBUF-NEXT:    s_mov_b32 s50, s12
; GFX9-MUBUF-NEXT:    s_mov_b32 s51, s13
; GFX9-MUBUF-NEXT:    s_mov_b32 s52, s14
; GFX9-MUBUF-NEXT:    s_mov_b32 s53, s15
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v41, v31
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_swappc_b64 s[30:31], s[64:65]
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:28
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:24
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:20
; GFX9-MUBUF-NEXT:    buffer_store_dword v40, off, s[0:3], s33 offset:16
; GFX9-MUBUF-NEXT:    v_lshrrev_b32_e64 v0, 6, s33
; GFX9-MUBUF-NEXT:    s_mov_b64 s[4:5], s[34:35]
; GFX9-MUBUF-NEXT:    s_mov_b64 s[6:7], s[36:37]
; GFX9-MUBUF-NEXT:    s_mov_b64 s[8:9], s[38:39]
; GFX9-MUBUF-NEXT:    s_mov_b64 s[10:11], s[48:49]
; GFX9-MUBUF-NEXT:    s_mov_b32 s12, s50
; GFX9-MUBUF-NEXT:    s_mov_b32 s13, s51
; GFX9-MUBUF-NEXT:    s_mov_b32 s14, s52
; GFX9-MUBUF-NEXT:    s_mov_b32 s15, s53
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v31, v41
; GFX9-MUBUF-NEXT:    v_add_u32_e32 v0, 16, v0
; GFX9-MUBUF-NEXT:    s_swappc_b64 s[30:31], s[64:65]
; GFX9-MUBUF-NEXT:  .LBB11_2: ; %bb5
; GFX9-MUBUF-NEXT:    s_or_b64 exec, exec, s[54:55]
; GFX9-MUBUF-NEXT:    buffer_load_dword v41, off, s[0:3], s33 ; 4-byte Folded Reload
; GFX9-MUBUF-NEXT:    buffer_load_dword v40, off, s[0:3], s33 offset:4 ; 4-byte Folded Reload
; GFX9-MUBUF-NEXT:    v_readlane_b32 s65, v42, 17
; GFX9-MUBUF-NEXT:    v_readlane_b32 s64, v42, 16
; GFX9-MUBUF-NEXT:    v_readlane_b32 s55, v42, 15
; GFX9-MUBUF-NEXT:    v_readlane_b32 s54, v42, 14
; GFX9-MUBUF-NEXT:    v_readlane_b32 s53, v42, 13
; GFX9-MUBUF-NEXT:    v_readlane_b32 s52, v42, 12
; GFX9-MUBUF-NEXT:    v_readlane_b32 s51, v42, 11
; GFX9-MUBUF-NEXT:    v_readlane_b32 s50, v42, 10
; GFX9-MUBUF-NEXT:    v_readlane_b32 s49, v42, 9
; GFX9-MUBUF-NEXT:    v_readlane_b32 s48, v42, 8
; GFX9-MUBUF-NEXT:    v_readlane_b32 s39, v42, 7
; GFX9-MUBUF-NEXT:    v_readlane_b32 s38, v42, 6
; GFX9-MUBUF-NEXT:    v_readlane_b32 s37, v42, 5
; GFX9-MUBUF-NEXT:    v_readlane_b32 s36, v42, 4
; GFX9-MUBUF-NEXT:    v_readlane_b32 s35, v42, 3
; GFX9-MUBUF-NEXT:    v_readlane_b32 s34, v42, 2
; GFX9-MUBUF-NEXT:    v_readlane_b32 s31, v42, 1
; GFX9-MUBUF-NEXT:    v_readlane_b32 s30, v42, 0
; GFX9-MUBUF-NEXT:    s_mov_b32 s32, s33
; GFX9-MUBUF-NEXT:    v_readlane_b32 s4, v42, 18
; GFX9-MUBUF-NEXT:    s_or_saveexec_b64 s[6:7], -1
; GFX9-MUBUF-NEXT:    buffer_load_dword v42, off, s[0:3], s33 offset:32 ; 4-byte Folded Reload
; GFX9-MUBUF-NEXT:    s_mov_b64 exec, s[6:7]
; GFX9-MUBUF-NEXT:    s_mov_b32 s33, s4
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0)
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: undefined_stack_store_reg:
; GFX9-FLATSCR:       ; %bb.0: ; %bb
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_mov_b32 s0, s33
; GFX9-FLATSCR-NEXT:    s_mov_b32 s33, s32
; GFX9-FLATSCR-NEXT:    s_or_saveexec_b64 s[2:3], -1
; GFX9-FLATSCR-NEXT:    scratch_store_dword off, v44, s33 offset:32 ; 4-byte Folded Spill
; GFX9-FLATSCR-NEXT:    s_mov_b64 exec, s[2:3]
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s0, 18
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s30, 0
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s31, 1
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s34, 2
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s35, 3
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s36, 4
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s37, 5
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s38, 6
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s39, 7
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s48, 8
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s49, 9
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s50, 10
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s51, 11
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s52, 12
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s53, 13
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s54, 14
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s55, 15
; GFX9-FLATSCR-NEXT:    scratch_store_dword off, v40, s33 offset:4 ; 4-byte Folded Spill
; GFX9-FLATSCR-NEXT:    scratch_store_dword off, v41, s33 ; 4-byte Folded Spill
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s64, 16
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v40, v0
; GFX9-FLATSCR-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v1
; GFX9-FLATSCR-NEXT:    s_add_i32 s32, s32, 48
; GFX9-FLATSCR-NEXT:    v_writelane_b32 v44, s65, 17
; GFX9-FLATSCR-NEXT:    scratch_store_dwordx4 off, v[40:43], s0
; GFX9-FLATSCR-NEXT:    s_and_saveexec_b64 s[54:55], vcc
; GFX9-FLATSCR-NEXT:    s_cbranch_execz .LBB11_2
; GFX9-FLATSCR-NEXT:  ; %bb.1: ; %bb4
; GFX9-FLATSCR-NEXT:    s_getpc_b64 s[0:1]
; GFX9-FLATSCR-NEXT:    s_add_u32 s0, s0, func@gotpcrel32@lo+4
; GFX9-FLATSCR-NEXT:    s_addc_u32 s1, s1, func@gotpcrel32@hi+12
; GFX9-FLATSCR-NEXT:    s_load_dwordx2 s[64:65], s[0:1], 0x0
; GFX9-FLATSCR-NEXT:    s_mov_b64 s[34:35], s[4:5]
; GFX9-FLATSCR-NEXT:    s_mov_b64 s[36:37], s[6:7]
; GFX9-FLATSCR-NEXT:    s_mov_b64 s[38:39], s[8:9]
; GFX9-FLATSCR-NEXT:    s_mov_b64 s[48:49], s[10:11]
; GFX9-FLATSCR-NEXT:    s_mov_b32 s50, s12
; GFX9-FLATSCR-NEXT:    s_mov_b32 s51, s13
; GFX9-FLATSCR-NEXT:    s_mov_b32 s52, s14
; GFX9-FLATSCR-NEXT:    s_mov_b32 s53, s15
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v41, v31
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_swappc_b64 s[30:31], s[64:65]
; GFX9-FLATSCR-NEXT:    s_add_i32 s0, s33, 16
; GFX9-FLATSCR-NEXT:    s_mov_b64 s[4:5], s[34:35]
; GFX9-FLATSCR-NEXT:    s_mov_b64 s[6:7], s[36:37]
; GFX9-FLATSCR-NEXT:    s_mov_b64 s[8:9], s[38:39]
; GFX9-FLATSCR-NEXT:    s_mov_b64 s[10:11], s[48:49]
; GFX9-FLATSCR-NEXT:    s_mov_b32 s12, s50
; GFX9-FLATSCR-NEXT:    s_mov_b32 s13, s51
; GFX9-FLATSCR-NEXT:    s_mov_b32 s14, s52
; GFX9-FLATSCR-NEXT:    s_mov_b32 s15, s53
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v31, v41
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-FLATSCR-NEXT:    scratch_store_dwordx4 off, v[40:43], s33 offset:16
; GFX9-FLATSCR-NEXT:    s_swappc_b64 s[30:31], s[64:65]
; GFX9-FLATSCR-NEXT:  .LBB11_2: ; %bb5
; GFX9-FLATSCR-NEXT:    s_or_b64 exec, exec, s[54:55]
; GFX9-FLATSCR-NEXT:    scratch_load_dword v41, off, s33 ; 4-byte Folded Reload
; GFX9-FLATSCR-NEXT:    scratch_load_dword v40, off, s33 offset:4 ; 4-byte Folded Reload
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s65, v44, 17
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s64, v44, 16
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s55, v44, 15
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s54, v44, 14
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s53, v44, 13
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s52, v44, 12
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s51, v44, 11
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s50, v44, 10
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s49, v44, 9
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s48, v44, 8
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s39, v44, 7
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s38, v44, 6
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s37, v44, 5
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s36, v44, 4
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s35, v44, 3
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s34, v44, 2
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s31, v44, 1
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s30, v44, 0
; GFX9-FLATSCR-NEXT:    s_mov_b32 s32, s33
; GFX9-FLATSCR-NEXT:    v_readlane_b32 s0, v44, 18
; GFX9-FLATSCR-NEXT:    s_or_saveexec_b64 s[2:3], -1
; GFX9-FLATSCR-NEXT:    scratch_load_dword v44, off, s33 offset:32 ; 4-byte Folded Reload
; GFX9-FLATSCR-NEXT:    s_mov_b64 exec, s[2:3]
; GFX9-FLATSCR-NEXT:    s_mov_b32 s33, s0
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: undefined_stack_store_reg:
; GFX11-TRUE16:       ; %bb.0: ; %bb
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_mov_b32 s0, s33
; GFX11-TRUE16-NEXT:    s_mov_b32 s33, s32
; GFX11-TRUE16-NEXT:    s_or_saveexec_b32 s1, -1
; GFX11-TRUE16-NEXT:    scratch_store_b32 off, v44, s33 offset:32 ; 4-byte Folded Spill
; GFX11-TRUE16-NEXT:    s_mov_b32 exec_lo, s1
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s0, 17
; GFX11-TRUE16-NEXT:    s_clause 0x1
; GFX11-TRUE16-NEXT:    scratch_store_b32 off, v40, s33 offset:4
; GFX11-TRUE16-NEXT:    scratch_store_b32 off, v41, s33
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v40, v0
; GFX11-TRUE16-NEXT:    s_add_i32 s32, s32, 48
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s30, 0
; GFX11-TRUE16-NEXT:    scratch_store_b128 off, v[40:43], s0
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s31, 1
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s34, 2
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s35, 3
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s36, 4
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s37, 5
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s38, 6
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s39, 7
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s48, 8
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s49, 9
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s50, 10
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s51, 11
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s52, 12
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s53, 13
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s54, 14
; GFX11-TRUE16-NEXT:    s_mov_b32 s54, exec_lo
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s64, 15
; GFX11-TRUE16-NEXT:    v_writelane_b32 v44, s65, 16
; GFX11-TRUE16-NEXT:    v_cmpx_eq_u32_e32 0, v1
; GFX11-TRUE16-NEXT:    s_cbranch_execz .LBB11_2
; GFX11-TRUE16-NEXT:  ; %bb.1: ; %bb4
; GFX11-TRUE16-NEXT:    s_getpc_b64 s[0:1]
; GFX11-TRUE16-NEXT:    s_add_u32 s0, s0, func@gotpcrel32@lo+4
; GFX11-TRUE16-NEXT:    s_addc_u32 s1, s1, func@gotpcrel32@hi+12
; GFX11-TRUE16-NEXT:    s_mov_b64 s[34:35], s[4:5]
; GFX11-TRUE16-NEXT:    s_load_b64 s[64:65], s[0:1], 0x0
; GFX11-TRUE16-NEXT:    s_mov_b64 s[36:37], s[6:7]
; GFX11-TRUE16-NEXT:    s_mov_b64 s[38:39], s[8:9]
; GFX11-TRUE16-NEXT:    s_mov_b64 s[48:49], s[10:11]
; GFX11-TRUE16-NEXT:    s_mov_b32 s50, s12
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v41, v31
; GFX11-TRUE16-NEXT:    s_mov_b32 s51, s13
; GFX11-TRUE16-NEXT:    s_mov_b32 s52, s14
; GFX11-TRUE16-NEXT:    s_mov_b32 s53, s15
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_swappc_b64 s[30:31], s[64:65]
; GFX11-TRUE16-NEXT:    s_add_i32 s0, s33, 16
; GFX11-TRUE16-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-TRUE16-NEXT:    v_dual_mov_b32 v31, v41 :: v_dual_mov_b32 v0, s0
; GFX11-TRUE16-NEXT:    s_mov_b64 s[4:5], s[34:35]
; GFX11-TRUE16-NEXT:    s_mov_b64 s[6:7], s[36:37]
; GFX11-TRUE16-NEXT:    s_mov_b64 s[8:9], s[38:39]
; GFX11-TRUE16-NEXT:    s_mov_b64 s[10:11], s[48:49]
; GFX11-TRUE16-NEXT:    s_mov_b32 s12, s50
; GFX11-TRUE16-NEXT:    s_mov_b32 s13, s51
; GFX11-TRUE16-NEXT:    s_mov_b32 s14, s52
; GFX11-TRUE16-NEXT:    s_mov_b32 s15, s53
; GFX11-TRUE16-NEXT:    scratch_store_b128 off, v[40:43], s33 offset:16
; GFX11-TRUE16-NEXT:    s_swappc_b64 s[30:31], s[64:65]
; GFX11-TRUE16-NEXT:  .LBB11_2: ; %bb5
; GFX11-TRUE16-NEXT:    s_or_b32 exec_lo, exec_lo, s54
; GFX11-TRUE16-NEXT:    s_clause 0x1
; GFX11-TRUE16-NEXT:    scratch_load_b32 v41, off, s33
; GFX11-TRUE16-NEXT:    scratch_load_b32 v40, off, s33 offset:4
; GFX11-TRUE16-NEXT:    v_readlane_b32 s65, v44, 16
; GFX11-TRUE16-NEXT:    v_readlane_b32 s64, v44, 15
; GFX11-TRUE16-NEXT:    v_readlane_b32 s54, v44, 14
; GFX11-TRUE16-NEXT:    v_readlane_b32 s53, v44, 13
; GFX11-TRUE16-NEXT:    v_readlane_b32 s52, v44, 12
; GFX11-TRUE16-NEXT:    v_readlane_b32 s51, v44, 11
; GFX11-TRUE16-NEXT:    v_readlane_b32 s50, v44, 10
; GFX11-TRUE16-NEXT:    v_readlane_b32 s49, v44, 9
; GFX11-TRUE16-NEXT:    v_readlane_b32 s48, v44, 8
; GFX11-TRUE16-NEXT:    v_readlane_b32 s39, v44, 7
; GFX11-TRUE16-NEXT:    v_readlane_b32 s38, v44, 6
; GFX11-TRUE16-NEXT:    v_readlane_b32 s37, v44, 5
; GFX11-TRUE16-NEXT:    v_readlane_b32 s36, v44, 4
; GFX11-TRUE16-NEXT:    v_readlane_b32 s35, v44, 3
; GFX11-TRUE16-NEXT:    v_readlane_b32 s34, v44, 2
; GFX11-TRUE16-NEXT:    v_readlane_b32 s31, v44, 1
; GFX11-TRUE16-NEXT:    v_readlane_b32 s30, v44, 0
; GFX11-TRUE16-NEXT:    s_mov_b32 s32, s33
; GFX11-TRUE16-NEXT:    v_readlane_b32 s0, v44, 17
; GFX11-TRUE16-NEXT:    s_or_saveexec_b32 s1, -1
; GFX11-TRUE16-NEXT:    scratch_load_b32 v44, off, s33 offset:32 ; 4-byte Folded Reload
; GFX11-TRUE16-NEXT:    s_mov_b32 exec_lo, s1
; GFX11-TRUE16-NEXT:    s_mov_b32 s33, s0
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: undefined_stack_store_reg:
; GFX11-FAKE16:       ; %bb.0: ; %bb
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_mov_b32 s0, s33
; GFX11-FAKE16-NEXT:    s_mov_b32 s33, s32
; GFX11-FAKE16-NEXT:    s_or_saveexec_b32 s1, -1
; GFX11-FAKE16-NEXT:    scratch_store_b32 off, v44, s33 offset:32 ; 4-byte Folded Spill
; GFX11-FAKE16-NEXT:    s_mov_b32 exec_lo, s1
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s0, 17
; GFX11-FAKE16-NEXT:    s_clause 0x1
; GFX11-FAKE16-NEXT:    scratch_store_b32 off, v40, s33 offset:4
; GFX11-FAKE16-NEXT:    scratch_store_b32 off, v41, s33
; GFX11-FAKE16-NEXT:    v_mov_b32_e32 v40, v0
; GFX11-FAKE16-NEXT:    s_add_i32 s32, s32, 48
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s30, 0
; GFX11-FAKE16-NEXT:    scratch_store_b128 off, v[40:43], s0
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s31, 1
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s34, 2
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s35, 3
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s36, 4
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s37, 5
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s38, 6
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s39, 7
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s48, 8
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s49, 9
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s50, 10
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s51, 11
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s52, 12
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s53, 13
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s54, 14
; GFX11-FAKE16-NEXT:    s_mov_b32 s54, exec_lo
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s64, 15
; GFX11-FAKE16-NEXT:    v_writelane_b32 v44, s65, 16
; GFX11-FAKE16-NEXT:    v_cmpx_eq_u32_e32 0, v1
; GFX11-FAKE16-NEXT:    s_cbranch_execz .LBB11_2
; GFX11-FAKE16-NEXT:  ; %bb.1: ; %bb4
; GFX11-FAKE16-NEXT:    s_getpc_b64 s[0:1]
; GFX11-FAKE16-NEXT:    s_add_u32 s0, s0, func@gotpcrel32@lo+4
; GFX11-FAKE16-NEXT:    s_addc_u32 s1, s1, func@gotpcrel32@hi+12
; GFX11-FAKE16-NEXT:    s_mov_b64 s[34:35], s[4:5]
; GFX11-FAKE16-NEXT:    s_load_b64 s[64:65], s[0:1], 0x0
; GFX11-FAKE16-NEXT:    s_mov_b64 s[36:37], s[6:7]
; GFX11-FAKE16-NEXT:    s_mov_b64 s[38:39], s[8:9]
; GFX11-FAKE16-NEXT:    s_mov_b64 s[48:49], s[10:11]
; GFX11-FAKE16-NEXT:    s_mov_b32 s50, s12
; GFX11-FAKE16-NEXT:    v_mov_b32_e32 v41, v31
; GFX11-FAKE16-NEXT:    s_mov_b32 s51, s13
; GFX11-FAKE16-NEXT:    s_mov_b32 s52, s14
; GFX11-FAKE16-NEXT:    s_mov_b32 s53, s15
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_swappc_b64 s[30:31], s[64:65]
; GFX11-FAKE16-NEXT:    s_add_i32 s0, s33, 16
; GFX11-FAKE16-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-FAKE16-NEXT:    v_dual_mov_b32 v31, v41 :: v_dual_mov_b32 v0, s0
; GFX11-FAKE16-NEXT:    s_mov_b64 s[4:5], s[34:35]
; GFX11-FAKE16-NEXT:    s_mov_b64 s[6:7], s[36:37]
; GFX11-FAKE16-NEXT:    s_mov_b64 s[8:9], s[38:39]
; GFX11-FAKE16-NEXT:    s_mov_b64 s[10:11], s[48:49]
; GFX11-FAKE16-NEXT:    s_mov_b32 s12, s50
; GFX11-FAKE16-NEXT:    s_mov_b32 s13, s51
; GFX11-FAKE16-NEXT:    s_mov_b32 s14, s52
; GFX11-FAKE16-NEXT:    s_mov_b32 s15, s53
; GFX11-FAKE16-NEXT:    scratch_store_b128 off, v[40:43], s33 offset:16
; GFX11-FAKE16-NEXT:    s_swappc_b64 s[30:31], s[64:65]
; GFX11-FAKE16-NEXT:  .LBB11_2: ; %bb5
; GFX11-FAKE16-NEXT:    s_or_b32 exec_lo, exec_lo, s54
; GFX11-FAKE16-NEXT:    s_clause 0x1
; GFX11-FAKE16-NEXT:    scratch_load_b32 v41, off, s33
; GFX11-FAKE16-NEXT:    scratch_load_b32 v40, off, s33 offset:4
; GFX11-FAKE16-NEXT:    v_readlane_b32 s65, v44, 16
; GFX11-FAKE16-NEXT:    v_readlane_b32 s64, v44, 15
; GFX11-FAKE16-NEXT:    v_readlane_b32 s54, v44, 14
; GFX11-FAKE16-NEXT:    v_readlane_b32 s53, v44, 13
; GFX11-FAKE16-NEXT:    v_readlane_b32 s52, v44, 12
; GFX11-FAKE16-NEXT:    v_readlane_b32 s51, v44, 11
; GFX11-FAKE16-NEXT:    v_readlane_b32 s50, v44, 10
; GFX11-FAKE16-NEXT:    v_readlane_b32 s49, v44, 9
; GFX11-FAKE16-NEXT:    v_readlane_b32 s48, v44, 8
; GFX11-FAKE16-NEXT:    v_readlane_b32 s39, v44, 7
; GFX11-FAKE16-NEXT:    v_readlane_b32 s38, v44, 6
; GFX11-FAKE16-NEXT:    v_readlane_b32 s37, v44, 5
; GFX11-FAKE16-NEXT:    v_readlane_b32 s36, v44, 4
; GFX11-FAKE16-NEXT:    v_readlane_b32 s35, v44, 3
; GFX11-FAKE16-NEXT:    v_readlane_b32 s34, v44, 2
; GFX11-FAKE16-NEXT:    v_readlane_b32 s31, v44, 1
; GFX11-FAKE16-NEXT:    v_readlane_b32 s30, v44, 0
; GFX11-FAKE16-NEXT:    s_mov_b32 s32, s33
; GFX11-FAKE16-NEXT:    v_readlane_b32 s0, v44, 17
; GFX11-FAKE16-NEXT:    s_or_saveexec_b32 s1, -1
; GFX11-FAKE16-NEXT:    scratch_load_b32 v44, off, s33 offset:32 ; 4-byte Folded Reload
; GFX11-FAKE16-NEXT:    s_mov_b32 exec_lo, s1
; GFX11-FAKE16-NEXT:    s_mov_b32 s33, s0
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
bb:
  %tmp = alloca <4 x float>, align 16, addrspace(5)
  %tmp2 = insertelement <4 x float> poison, float %arg, i32 0
  store <4 x float> %tmp2, ptr addrspace(5) poison
  %tmp3 = icmp eq i32 %arg1, 0
  br i1 %tmp3, label %bb4, label %bb5

bb4:
  call void @func(ptr addrspace(5) nonnull undef)
  store <4 x float> %tmp2, ptr addrspace(5) %tmp, align 16
  call void @func(ptr addrspace(5) nonnull %tmp)
  br label %bb5

bb5:
  ret void
}

; GCN-LABEL: {{^}}alloca_ptr_nonentry_block:
; GCN: s_and_saveexec_b64
; MUBUF:   buffer_load_dword v{{[0-9]+}}, off, s[0:3], s32 offset:4
; FLATSCR: scratch_load_dword v{{[0-9]+}}, off, s32 offset:4

; CI: v_lshr_b32_e64 [[SHIFT:v[0-9]+]], s32, 6
; CI-NEXT: v_or_b32_e32 [[PTR:v[0-9]+]], 4, [[SHIFT]]

; GFX9-MUBUF: v_lshrrev_b32_e64 [[SHIFT:v[0-9]+]], 6, s32
; GFX9-MUBUF-NEXT: v_or_b32_e32 [[PTR:v[0-9]+]], 4, [[SHIFT]]

; GFX9-FLATSCR: v_or_b32_e64 [[PTR:v[0-9]+]], s32, 4

; GCN: ds_write_b32 v{{[0-9]+}}, [[PTR]]
define void @alloca_ptr_nonentry_block(i32 %arg0) #0 {
; CI-LABEL: alloca_ptr_nonentry_block:
; CI:       ; %bb.0:
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; CI-NEXT:    s_and_saveexec_b64 s[4:5], vcc
; CI-NEXT:    s_cbranch_execz .LBB12_2
; CI-NEXT:  ; %bb.1: ; %bb
; CI-NEXT:    buffer_load_dword v0, off, s[0:3], s32 offset:4 glc
; CI-NEXT:    s_waitcnt vmcnt(0)
; CI-NEXT:    v_lshr_b32_e64 v1, s32, 6
; CI-NEXT:    v_or_b32_e32 v0, 4, v1
; CI-NEXT:    s_mov_b32 m0, -1
; CI-NEXT:    ds_write_b32 v0, v0
; CI-NEXT:  .LBB12_2: ; %ret
; CI-NEXT:    s_or_b64 exec, exec, s[4:5]
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: alloca_ptr_nonentry_block:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; GFX9-MUBUF-NEXT:    s_and_saveexec_b64 s[4:5], vcc
; GFX9-MUBUF-NEXT:    s_cbranch_execz .LBB12_2
; GFX9-MUBUF-NEXT:  ; %bb.1: ; %bb
; GFX9-MUBUF-NEXT:    buffer_load_dword v0, off, s[0:3], s32 offset:4 glc
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0)
; GFX9-MUBUF-NEXT:    v_lshrrev_b32_e64 v1, 6, s32
; GFX9-MUBUF-NEXT:    v_or_b32_e32 v0, 4, v1
; GFX9-MUBUF-NEXT:    ds_write_b32 v0, v0
; GFX9-MUBUF-NEXT:  .LBB12_2: ; %ret
; GFX9-MUBUF-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: alloca_ptr_nonentry_block:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; GFX9-FLATSCR-NEXT:    s_and_saveexec_b64 s[0:1], vcc
; GFX9-FLATSCR-NEXT:    s_cbranch_execz .LBB12_2
; GFX9-FLATSCR-NEXT:  ; %bb.1: ; %bb
; GFX9-FLATSCR-NEXT:    scratch_load_dword v0, off, s32 offset:4 glc
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT:    v_or_b32_e64 v0, s32, 4
; GFX9-FLATSCR-NEXT:    ds_write_b32 v0, v0
; GFX9-FLATSCR-NEXT:  .LBB12_2: ; %ret
; GFX9-FLATSCR-NEXT:    s_or_b64 exec, exec, s[0:1]
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: alloca_ptr_nonentry_block:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_mov_b32 s0, exec_lo
; GFX11-TRUE16-NEXT:    v_cmpx_eq_u32_e32 0, v0
; GFX11-TRUE16-NEXT:    s_cbranch_execz .LBB12_2
; GFX11-TRUE16-NEXT:  ; %bb.1: ; %bb
; GFX11-TRUE16-NEXT:    scratch_load_b32 v0, off, s32 offset:4 glc dlc
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-TRUE16-NEXT:    v_or_b32_e64 v0, s32, 4
; GFX11-TRUE16-NEXT:    ds_store_b32 v0, v0
; GFX11-TRUE16-NEXT:  .LBB12_2: ; %ret
; GFX11-TRUE16-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: alloca_ptr_nonentry_block:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_mov_b32 s0, exec_lo
; GFX11-FAKE16-NEXT:    v_cmpx_eq_u32_e32 0, v0
; GFX11-FAKE16-NEXT:    s_cbranch_execz .LBB12_2
; GFX11-FAKE16-NEXT:  ; %bb.1: ; %bb
; GFX11-FAKE16-NEXT:    scratch_load_b32 v0, off, s32 offset:4 glc dlc
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-FAKE16-NEXT:    v_or_b32_e64 v0, s32, 4
; GFX11-FAKE16-NEXT:    ds_store_b32 v0, v0
; GFX11-FAKE16-NEXT:  .LBB12_2: ; %ret
; GFX11-FAKE16-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
  %alloca0 = alloca { i8, i32 }, align 8, addrspace(5)
  %cmp = icmp eq i32 %arg0, 0
  br i1 %cmp, label %bb, label %ret

bb:
  %gep0 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %alloca0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, ptr addrspace(5) %alloca0, i32 0, i32 1
  %load1 = load volatile i32, ptr addrspace(5) %gep1
  store volatile ptr addrspace(5) %gep1, ptr addrspace(3) poison
  br label %ret

ret:
  ret void
}

%struct0 = type { [4224 x %type.i16] }
%type.i16 = type { i16 }
@_ZZN0 = external hidden addrspace(3) global %struct0, align 8

; GFX11-TRUE16-LABEL: tied_operand_test:
; GFX11-TRUE16:       ; %bb.0: ; %entry
; GFX11-TRUE16:     scratch_load_d16_b16 [[LDRESULT:v[0-9]+]], off, off
; GFX11-TRUE16:     v_mov_b16_e32 [[C:v[0-9]]].{{(l|h)}}, 0x7b
; GFX11-TRUE16-DAG:     ds_store_b16 v{{[0-9]+}}, [[LDRESULT]]  offset:10
; GFX11-TRUE16-NEXT:    s_endpgm
;
; GFX11-FAKE16-LABEL: tied_operand_test:
; GFX11-FAKE16:       ; %bb.0: ; %entry
; GFX11-FAKE16:     scratch_load_u16 [[LDRESULT:v[0-9]+]], off, off
; GFX11-FAKE16:     v_dual_mov_b32 [[C:v[0-9]+]], 0x7b :: v_dual_mov_b32 v{{[0-9]+}}, s{{[0-9]+}}
; GFX11-FAKE16-DAG:     ds_store_b16 v{{[0-9]+}}, [[LDRESULT]]  offset:10
; GFX11-FAKE16-DAG:     ds_store_b16 v{{[0-9]+}}, [[C]]  offset:8
; GFX11-FAKE16-NEXT:    s_endpgm
define protected amdgpu_kernel void @tied_operand_test(i1 %c1, i1 %c2, i32 %val) {
; CI-LABEL: tied_operand_test:
; CI:       ; %bb.0: ; %entry
; CI-NEXT:    s_add_u32 s0, s0, s17
; CI-NEXT:    s_addc_u32 s1, s1, 0
; CI-NEXT:    buffer_load_ushort v0, off, s[0:3], 0
; CI-NEXT:    s_load_dword s4, s[8:9], 0x1
; CI-NEXT:    v_mov_b32_e32 v1, 0x7b
; CI-NEXT:    s_mov_b32 m0, -1
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_lshl_b32 s4, s4, 1
; CI-NEXT:    v_mov_b32_e32 v2, s4
; CI-NEXT:    ds_write_b16 v2, v1 offset:8
; CI-NEXT:    s_waitcnt vmcnt(0)
; CI-NEXT:    ds_write_b16 v2, v0 offset:10
; CI-NEXT:    s_endpgm
;
; GFX9-MUBUF-LABEL: tied_operand_test:
; GFX9-MUBUF:       ; %bb.0: ; %entry
; GFX9-MUBUF-NEXT:    s_add_u32 s0, s0, s17
; GFX9-MUBUF-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-MUBUF-NEXT:    buffer_load_ushort v0, off, s[0:3], 0
; GFX9-MUBUF-NEXT:    s_load_dword s4, s[8:9], 0x4
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v1, 0x7b
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_lshl_b32 s4, s4, 1
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v2, s4
; GFX9-MUBUF-NEXT:    ds_write_b16 v2, v1 offset:8
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0)
; GFX9-MUBUF-NEXT:    ds_write_b16 v2, v0 offset:10
; GFX9-MUBUF-NEXT:    s_endpgm
;
; GFX9-FLATSCR-LABEL: tied_operand_test:
; GFX9-FLATSCR:       ; %bb.0: ; %entry
; GFX9-FLATSCR-NEXT:    s_add_u32 flat_scratch_lo, s8, s13
; GFX9-FLATSCR-NEXT:    s_addc_u32 flat_scratch_hi, s9, 0
; GFX9-FLATSCR-NEXT:    s_mov_b32 s0, 0
; GFX9-FLATSCR-NEXT:    scratch_load_ushort v0, off, s0
; GFX9-FLATSCR-NEXT:    s_load_dword s0, s[4:5], 0x4
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v1, 0x7b
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_lshl_b32 s0, s0, 1
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v2, s0
; GFX9-FLATSCR-NEXT:    ds_write_b16 v2, v1 offset:8
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT:    ds_write_b16 v2, v0 offset:10
; GFX9-FLATSCR-NEXT:    s_endpgm
;
; GFX11-TRUE16-LABEL: tied_operand_test:
; GFX11-TRUE16:       ; %bb.0: ; %entry
; GFX11-TRUE16-NEXT:    scratch_load_d16_b16 v0, off, off
; GFX11-TRUE16-NEXT:    s_load_b32 s0, s[4:5], 0x4
; GFX11-TRUE16-NEXT:    v_mov_b16_e32 v0.h, 0x7b
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_lshl_b32 s0, s0, 1
; GFX11-TRUE16-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v1, s0
; GFX11-TRUE16-NEXT:    ds_store_b16_d16_hi v1, v0 offset:8
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-TRUE16-NEXT:    ds_store_b16 v1, v0 offset:10
; GFX11-TRUE16-NEXT:    s_endpgm
;
; GFX11-FAKE16-LABEL: tied_operand_test:
; GFX11-FAKE16:       ; %bb.0: ; %entry
; GFX11-FAKE16-NEXT:    scratch_load_u16 v0, off, off
; GFX11-FAKE16-NEXT:    s_load_b32 s0, s[4:5], 0x4
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_lshl_b32 s0, s0, 1
; GFX11-FAKE16-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-FAKE16-NEXT:    v_dual_mov_b32 v1, 0x7b :: v_dual_mov_b32 v2, s0
; GFX11-FAKE16-NEXT:    ds_store_b16 v2, v1 offset:8
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-FAKE16-NEXT:    ds_store_b16 v2, v0 offset:10
; GFX11-FAKE16-NEXT:    s_endpgm
entry:
  %scratch0 = alloca i16, align 4, addrspace(5)
  %scratch1 = alloca i16, align 4, addrspace(5)
  %first = select i1 %c1, ptr addrspace(5) %scratch0, ptr addrspace(5) %scratch1
  %spec.select = select i1 %c2, ptr addrspace(5) %first, ptr addrspace(5) %scratch0
  %dead.load = load i16, ptr addrspace(5) %spec.select, align 2
  %scratch0.load = load i16, ptr addrspace(5) %scratch0, align 4
  %add4 = add nuw nsw i32 %val, 4
  %addr0 = getelementptr inbounds %struct0, ptr addrspace(3) @_ZZN0, i32 0, i32 0, i32 %add4, i32 0
  store i16 123, ptr addrspace(3) %addr0, align 2
  %add5 = add nuw nsw i32 %val, 5
  %addr1 = getelementptr inbounds %struct0, ptr addrspace(3) @_ZZN0, i32 0, i32 0, i32 %add5, i32 0
  store i16 %scratch0.load, ptr addrspace(3) %addr1, align 2
  ret void
}

; GCN-LABEL: {{^}}fi_vop3_literal_error:
; CI: v_lshr_b32_e64 [[SCALED_FP:v[0-9]+]], s33, 6
; CI: s_movk_i32 vcc_lo, 0x3000
; CI-NEXT: v_add_i32_e32 [[SCALED_FP]], vcc, vcc_lo, [[SCALED_FP]]
; CI-NEXT: v_add_i32_e32 v0, vcc, 64, [[SCALED_FP]]

; GFX9-MUBUF: v_lshrrev_b32_e64 [[SCALED_FP:v[0-9]+]], 6, s33
; GFX9-MUBUF-NEXT: v_add_u32_e32 [[SCALED_FP]], 0x3000, [[SCALED_FP]]
; GFX9-MUBUF-NEXT: v_add_u32_e32 v0, 64, [[SCALED_FP]]
define void @fi_vop3_literal_error() {
; CI-LABEL: fi_vop3_literal_error:
; CI:       ; %bb.0: ; %entry
; CI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CI-NEXT:    s_mov_b32 s4, s33
; CI-NEXT:    s_add_i32 s33, s32, 0x7ffc0
; CI-NEXT:    s_and_b32 s33, s33, 0xfff80000
; CI-NEXT:    v_lshr_b32_e64 v1, s33, 6
; CI-NEXT:    s_movk_i32 vcc_lo, 0x3000
; CI-NEXT:    v_add_i32_e32 v1, vcc, vcc_lo, v1
; CI-NEXT:    v_add_i32_e32 v0, vcc, 64, v1
; CI-NEXT:    v_mov_b32_e32 v1, 0
; CI-NEXT:    v_mov_b32_e32 v2, 0x2000
; CI-NEXT:    buffer_store_dword v1, v2, s[0:3], s33 offen
; CI-NEXT:    buffer_load_dword v1, v0, s[0:3], 0 offen glc
; CI-NEXT:    s_waitcnt vmcnt(0)
; CI-NEXT:    buffer_load_dword v0, v0, s[0:3], 0 offen offset:4 glc
; CI-NEXT:    s_waitcnt vmcnt(0)
; CI-NEXT:    s_mov_b32 s5, s34
; CI-NEXT:    s_mov_b32 s34, s32
; CI-NEXT:    s_add_i32 s32, s32, 0x200000
; CI-NEXT:    s_mov_b32 s32, s34
; CI-NEXT:    s_mov_b32 s34, s5
; CI-NEXT:    s_mov_b32 s33, s4
; CI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-MUBUF-LABEL: fi_vop3_literal_error:
; GFX9-MUBUF:       ; %bb.0: ; %entry
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_mov_b32 s4, s33
; GFX9-MUBUF-NEXT:    s_add_i32 s33, s32, 0x7ffc0
; GFX9-MUBUF-NEXT:    s_and_b32 s33, s33, 0xfff80000
; GFX9-MUBUF-NEXT:    v_lshrrev_b32_e64 v1, 6, s33
; GFX9-MUBUF-NEXT:    v_add_u32_e32 v1, 0x3000, v1
; GFX9-MUBUF-NEXT:    v_add_u32_e32 v0, 64, v1
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v1, 0
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v2, 0x2000
; GFX9-MUBUF-NEXT:    buffer_store_dword v1, v2, s[0:3], s33 offen
; GFX9-MUBUF-NEXT:    buffer_load_dword v1, v0, s[0:3], 0 offen glc
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0)
; GFX9-MUBUF-NEXT:    s_mov_b32 s5, s34
; GFX9-MUBUF-NEXT:    buffer_load_dword v1, v0, s[0:3], 0 offen offset:4 glc
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(0)
; GFX9-MUBUF-NEXT:    s_mov_b32 s34, s32
; GFX9-MUBUF-NEXT:    s_add_i32 s32, s32, 0x200000
; GFX9-MUBUF-NEXT:    ; kill: killed $vgpr0
; GFX9-MUBUF-NEXT:    s_mov_b32 s32, s34
; GFX9-MUBUF-NEXT:    s_mov_b32 s34, s5
; GFX9-MUBUF-NEXT:    s_mov_b32 s33, s4
; GFX9-MUBUF-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-FLATSCR-LABEL: fi_vop3_literal_error:
; GFX9-FLATSCR:       ; %bb.0: ; %entry
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_mov_b32 s0, s33
; GFX9-FLATSCR-NEXT:    s_add_i32 s33, s32, 0x1fff
; GFX9-FLATSCR-NEXT:    s_and_b32 s33, s33, 0xffffe000
; GFX9-FLATSCR-NEXT:    s_mov_b32 s1, s34
; GFX9-FLATSCR-NEXT:    s_mov_b32 s34, s32
; GFX9-FLATSCR-NEXT:    s_add_i32 s32, s32, 0x8000
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v0, 0
; GFX9-FLATSCR-NEXT:    s_add_i32 s2, s33, 0x2000
; GFX9-FLATSCR-NEXT:    scratch_store_dword off, v0, s2
; GFX9-FLATSCR-NEXT:    s_add_i32 s2, s33, 0x3000
; GFX9-FLATSCR-NEXT:    scratch_load_dwordx2 v[0:1], off, s2 offset:64 glc
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT:    s_mov_b32 s32, s34
; GFX9-FLATSCR-NEXT:    s_mov_b32 s34, s1
; GFX9-FLATSCR-NEXT:    s_mov_b32 s33, s0
; GFX9-FLATSCR-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-TRUE16-LABEL: fi_vop3_literal_error:
; GFX11-TRUE16:       ; %bb.0: ; %entry
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_mov_b32 s0, s33
; GFX11-TRUE16-NEXT:    s_add_i32 s33, s32, 0x1fff
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v0, 0
; GFX11-TRUE16-NEXT:    s_and_b32 s33, s33, 0xffffe000
; GFX11-TRUE16-NEXT:    s_mov_b32 s1, s34
; GFX11-TRUE16-NEXT:    s_mov_b32 s34, s32
; GFX11-TRUE16-NEXT:    s_add_i32 s32, s32, 0x8000
; GFX11-TRUE16-NEXT:    s_add_i32 s2, s33, 0x2000
; GFX11-TRUE16-NEXT:    s_mov_b32 s32, s34
; GFX11-TRUE16-NEXT:    scratch_store_b32 off, v0, s2
; GFX11-TRUE16-NEXT:    s_add_i32 s2, s33, 0x3000
; GFX11-TRUE16-NEXT:    s_mov_b32 s34, s1
; GFX11-TRUE16-NEXT:    scratch_load_b64 v[0:1], off, s2 offset:64 glc dlc
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-TRUE16-NEXT:    s_mov_b32 s33, s0
; GFX11-TRUE16-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-FAKE16-LABEL: fi_vop3_literal_error:
; GFX11-FAKE16:       ; %bb.0: ; %entry
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_mov_b32 s0, s33
; GFX11-FAKE16-NEXT:    s_add_i32 s33, s32, 0x1fff
; GFX11-FAKE16-NEXT:    v_mov_b32_e32 v0, 0
; GFX11-FAKE16-NEXT:    s_and_b32 s33, s33, 0xffffe000
; GFX11-FAKE16-NEXT:    s_mov_b32 s1, s34
; GFX11-FAKE16-NEXT:    s_mov_b32 s34, s32
; GFX11-FAKE16-NEXT:    s_add_i32 s32, s32, 0x8000
; GFX11-FAKE16-NEXT:    s_add_i32 s2, s33, 0x2000
; GFX11-FAKE16-NEXT:    s_mov_b32 s32, s34
; GFX11-FAKE16-NEXT:    scratch_store_b32 off, v0, s2
; GFX11-FAKE16-NEXT:    s_add_i32 s2, s33, 0x3000
; GFX11-FAKE16-NEXT:    s_mov_b32 s34, s1
; GFX11-FAKE16-NEXT:    scratch_load_b64 v[0:1], off, s2 offset:64 glc dlc
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-FAKE16-NEXT:    s_mov_b32 s33, s0
; GFX11-FAKE16-NEXT:    s_setpc_b64 s[30:31]
entry:
  %pin.low = alloca i32, align 8192, addrspace(5)
  %local.area = alloca [1060 x i64], align 4096, addrspace(5)
  store i32 0, ptr addrspace(5) %pin.low, align 4
  %gep.small.offset = getelementptr i8, ptr addrspace(5) %local.area, i64 64
  %load1 = load volatile i64, ptr addrspace(5) %gep.small.offset, align 4
  ret void
}

; Check for "SOP2/SOPC instruction requires too many immediate
; constants" verifier error.  Frame index would fold into low half of
; the lowered flat pointer add, and use s_add_u32 instead of
; s_add_i32.

; GCN-LABEL: {{^}}fi_sop2_s_add_u32_literal_error:
; GCN: s_add_u32 [[ADD_LO:s[0-9]+]], 0, 0x2010
; GCN: s_addc_u32 [[ADD_HI:s[0-9]+]], s{{[0-9]+}}, 0
define amdgpu_kernel void @fi_sop2_s_add_u32_literal_error() #0 {
; CI-LABEL: fi_sop2_s_add_u32_literal_error:
; CI:       ; %bb.0: ; %entry
; CI-NEXT:    s_load_dword s5, s[8:9], 0x30
; CI-NEXT:    s_add_u32 s0, s0, s17
; CI-NEXT:    s_addc_u32 s1, s1, 0
; CI-NEXT:    s_add_u32 s4, 0, 0x2010
; CI-NEXT:    v_mov_b32_e32 v0, 0
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_addc_u32 s5, s5, 0
; CI-NEXT:    v_cmp_lt_u64_e64 s[4:5], s[4:5], 2
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; CI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; CI-NEXT:    v_cmp_ne_u32_e64 s[4:5], 1, v0
; CI-NEXT:  .LBB15_1: ; %.shuffle.then.i.i.i.i
; CI-NEXT:    ; =>This Inner Loop Header: Depth=1
; CI-NEXT:    s_and_b64 vcc, exec, s[4:5]
; CI-NEXT:    s_cbranch_vccnz .LBB15_1
; CI-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; CI-NEXT:    buffer_load_dword v0, off, s[0:3], 0 offset:8
; CI-NEXT:    buffer_load_dword v1, off, s[0:3], 0 offset:4
; CI-NEXT:    s_waitcnt vmcnt(1)
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; CI-NEXT:    s_waitcnt vmcnt(1)
; CI-NEXT:    buffer_store_dword v1, off, s[0:3], 0
; CI-NEXT:    s_endpgm
;
; GFX9-MUBUF-LABEL: fi_sop2_s_add_u32_literal_error:
; GFX9-MUBUF:       ; %bb.0: ; %entry
; GFX9-MUBUF-NEXT:    s_add_u32 s0, s0, s17
; GFX9-MUBUF-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-MUBUF-NEXT:    s_mov_b64 s[4:5], src_private_base
; GFX9-MUBUF-NEXT:    s_add_u32 s4, 0, 0x2010
; GFX9-MUBUF-NEXT:    s_addc_u32 s5, s5, 0
; GFX9-MUBUF-NEXT:    v_cmp_lt_u64_e64 s[4:5], s[4:5], 2
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v0, 0
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GFX9-MUBUF-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9-MUBUF-NEXT:    v_cmp_ne_u32_e64 s[4:5], 1, v0
; GFX9-MUBUF-NEXT:  .LBB15_1: ; %.shuffle.then.i.i.i.i
; GFX9-MUBUF-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX9-MUBUF-NEXT:    s_and_b64 vcc, exec, s[4:5]
; GFX9-MUBUF-NEXT:    s_cbranch_vccnz .LBB15_1
; GFX9-MUBUF-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; GFX9-MUBUF-NEXT:    buffer_load_dword v0, off, s[0:3], 0 offset:8
; GFX9-MUBUF-NEXT:    buffer_load_dword v1, off, s[0:3], 0 offset:4
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(1)
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(1)
; GFX9-MUBUF-NEXT:    buffer_store_dword v1, off, s[0:3], 0
; GFX9-MUBUF-NEXT:    s_endpgm
;
; GFX9-FLATSCR-LABEL: fi_sop2_s_add_u32_literal_error:
; GFX9-FLATSCR:       ; %bb.0: ; %entry
; GFX9-FLATSCR-NEXT:    s_add_u32 flat_scratch_lo, s8, s13
; GFX9-FLATSCR-NEXT:    s_addc_u32 flat_scratch_hi, s9, 0
; GFX9-FLATSCR-NEXT:    s_mov_b64 s[0:1], src_private_base
; GFX9-FLATSCR-NEXT:    s_add_u32 s0, 0, 0x2010
; GFX9-FLATSCR-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v0, 0
; GFX9-FLATSCR-NEXT:    v_cmp_lt_u64_e64 s[0:1], s[0:1], 2
; GFX9-FLATSCR-NEXT:    s_mov_b32 s2, 0
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-FLATSCR-NEXT:    scratch_store_dwordx2 off, v[0:1], s2
; GFX9-FLATSCR-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; GFX9-FLATSCR-NEXT:    v_cmp_ne_u32_e64 s[0:1], 1, v0
; GFX9-FLATSCR-NEXT:  .LBB15_1: ; %.shuffle.then.i.i.i.i
; GFX9-FLATSCR-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX9-FLATSCR-NEXT:    s_and_b64 vcc, exec, s[0:1]
; GFX9-FLATSCR-NEXT:    s_cbranch_vccnz .LBB15_1
; GFX9-FLATSCR-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; GFX9-FLATSCR-NEXT:    s_mov_b32 s0, 0
; GFX9-FLATSCR-NEXT:    s_nop 1
; GFX9-FLATSCR-NEXT:    scratch_load_dwordx2 v[0:1], off, s0 offset:4
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT:    scratch_store_dwordx2 off, v[0:1], s0
; GFX9-FLATSCR-NEXT:    s_endpgm
;
; GFX11-TRUE16-LABEL: fi_sop2_s_add_u32_literal_error:
; GFX11-TRUE16:       ; %bb.0: ; %entry
; GFX11-TRUE16-NEXT:    s_mov_b64 s[0:1], src_private_base
; GFX11-TRUE16-NEXT:    s_add_u32 s0, 0, 0x2010
; GFX11-TRUE16-NEXT:    s_addc_u32 s1, s1, 0
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v0, 0
; GFX11-TRUE16-NEXT:    v_cmp_lt_u64_e64 s0, s[0:1], 2
; GFX11-TRUE16-NEXT:    s_mov_b32 s1, 0
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-TRUE16-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11-TRUE16-NEXT:    scratch_store_b64 off, v[0:1], s1
; GFX11-TRUE16-NEXT:    v_cmp_ne_u32_e64 s0, 1, v2
; GFX11-TRUE16-NEXT:  .LBB15_1: ; %.shuffle.then.i.i.i.i
; GFX11-TRUE16-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX11-TRUE16-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-TRUE16-NEXT:    s_and_b32 vcc_lo, exec_lo, s0
; GFX11-TRUE16-NEXT:    s_cbranch_vccnz .LBB15_1
; GFX11-TRUE16-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; GFX11-TRUE16-NEXT:    scratch_load_b64 v[0:1], off, off offset:4
; GFX11-TRUE16-NEXT:    s_mov_b32 s0, 0
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-TRUE16-NEXT:    scratch_store_b64 off, v[0:1], s0
; GFX11-TRUE16-NEXT:    s_endpgm
;
; GFX11-FAKE16-LABEL: fi_sop2_s_add_u32_literal_error:
; GFX11-FAKE16:       ; %bb.0: ; %entry
; GFX11-FAKE16-NEXT:    s_mov_b64 s[0:1], src_private_base
; GFX11-FAKE16-NEXT:    s_add_u32 s0, 0, 0x2010
; GFX11-FAKE16-NEXT:    s_addc_u32 s1, s1, 0
; GFX11-FAKE16-NEXT:    v_mov_b32_e32 v0, 0
; GFX11-FAKE16-NEXT:    v_cmp_lt_u64_e64 s0, s[0:1], 2
; GFX11-FAKE16-NEXT:    s_mov_b32 s1, 0
; GFX11-FAKE16-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-FAKE16-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11-FAKE16-NEXT:    scratch_store_b64 off, v[0:1], s1
; GFX11-FAKE16-NEXT:    v_cmp_ne_u32_e64 s0, 1, v2
; GFX11-FAKE16-NEXT:  .LBB15_1: ; %.shuffle.then.i.i.i.i
; GFX11-FAKE16-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX11-FAKE16-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-FAKE16-NEXT:    s_and_b32 vcc_lo, exec_lo, s0
; GFX11-FAKE16-NEXT:    s_cbranch_vccnz .LBB15_1
; GFX11-FAKE16-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; GFX11-FAKE16-NEXT:    scratch_load_b64 v[0:1], off, off offset:4
; GFX11-FAKE16-NEXT:    s_mov_b32 s0, 0
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-FAKE16-NEXT:    scratch_store_b64 off, v[0:1], s0
; GFX11-FAKE16-NEXT:    s_endpgm
entry:
  %.omp.reduction.element.i.i.i.i = alloca [1024 x i32], align 4, addrspace(5)
  %Total3.i.i = alloca [1024 x i32], align 16, addrspace(5)
  %Total3.ascast.i.i = addrspacecast ptr addrspace(5) %Total3.i.i to ptr
  %gep = getelementptr i8, ptr %Total3.ascast.i.i, i64 4096
  %p2i = ptrtoint ptr %gep to i64
  br label %.shuffle.then.i.i.i.i

.shuffle.then.i.i.i.i:                            ; preds = %.shuffle.then.i.i.i.i, %entry
  store i64 0, ptr addrspace(5) null, align 4
  %icmp = icmp ugt i64 %p2i, 1
  br i1 %icmp, label %.shuffle.then.i.i.i.i, label %vector.body.i.i.i.i

vector.body.i.i.i.i:                              ; preds = %.shuffle.then.i.i.i.i
  %wide.load9.i.i.i.i = load <2 x i32>, ptr addrspace(5) %.omp.reduction.element.i.i.i.i, align 4
  store <2 x i32> %wide.load9.i.i.i.i, ptr addrspace(5) null, align 4
  ret void
}

; GCN-LABEL: {{^}}fi_sop2_and_literal_error:
; GCN: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0x1fe00
define amdgpu_kernel void @fi_sop2_and_literal_error() #0 {
; CI-LABEL: fi_sop2_and_literal_error:
; CI:       ; %bb.0: ; %entry
; CI-NEXT:    s_add_u32 s0, s0, s17
; CI-NEXT:    s_addc_u32 s1, s1, 0
; CI-NEXT:    v_mov_b32_e32 v0, 0
; CI-NEXT:    s_mov_b64 s[4:5], -1
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; CI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; CI-NEXT:    v_cmp_ne_u32_e64 s[4:5], 1, v0
; CI-NEXT:  .LBB16_1: ; %.shuffle.then.i.i.i.i
; CI-NEXT:    ; =>This Inner Loop Header: Depth=1
; CI-NEXT:    s_and_b64 vcc, exec, s[4:5]
; CI-NEXT:    s_cbranch_vccnz .LBB16_1
; CI-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; CI-NEXT:    buffer_load_dword v0, off, s[0:3], 0 offset:8
; CI-NEXT:    buffer_load_dword v1, off, s[0:3], 0 offset:4
; CI-NEXT:    s_waitcnt vmcnt(1)
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; CI-NEXT:    s_waitcnt vmcnt(1)
; CI-NEXT:    buffer_store_dword v1, off, s[0:3], 0
; CI-NEXT:    s_endpgm
;
; GFX9-MUBUF-LABEL: fi_sop2_and_literal_error:
; GFX9-MUBUF:       ; %bb.0: ; %entry
; GFX9-MUBUF-NEXT:    s_add_u32 s0, s0, s17
; GFX9-MUBUF-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v0, 0
; GFX9-MUBUF-NEXT:    s_mov_b64 s[4:5], -1
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GFX9-MUBUF-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9-MUBUF-NEXT:    v_cmp_ne_u32_e64 s[4:5], 1, v0
; GFX9-MUBUF-NEXT:  .LBB16_1: ; %.shuffle.then.i.i.i.i
; GFX9-MUBUF-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX9-MUBUF-NEXT:    s_and_b64 vcc, exec, s[4:5]
; GFX9-MUBUF-NEXT:    s_cbranch_vccnz .LBB16_1
; GFX9-MUBUF-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; GFX9-MUBUF-NEXT:    buffer_load_dword v0, off, s[0:3], 0 offset:8
; GFX9-MUBUF-NEXT:    buffer_load_dword v1, off, s[0:3], 0 offset:4
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(1)
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(1)
; GFX9-MUBUF-NEXT:    buffer_store_dword v1, off, s[0:3], 0
; GFX9-MUBUF-NEXT:    s_endpgm
;
; GFX9-FLATSCR-LABEL: fi_sop2_and_literal_error:
; GFX9-FLATSCR:       ; %bb.0: ; %entry
; GFX9-FLATSCR-NEXT:    s_add_u32 flat_scratch_lo, s8, s13
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v0, 0
; GFX9-FLATSCR-NEXT:    s_addc_u32 flat_scratch_hi, s9, 0
; GFX9-FLATSCR-NEXT:    s_mov_b32 s0, 0
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-FLATSCR-NEXT:    scratch_store_dwordx2 off, v[0:1], s0
; GFX9-FLATSCR-NEXT:    s_mov_b64 s[0:1], -1
; GFX9-FLATSCR-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; GFX9-FLATSCR-NEXT:    v_cmp_ne_u32_e64 s[0:1], 1, v0
; GFX9-FLATSCR-NEXT:  .LBB16_1: ; %.shuffle.then.i.i.i.i
; GFX9-FLATSCR-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX9-FLATSCR-NEXT:    s_and_b64 vcc, exec, s[0:1]
; GFX9-FLATSCR-NEXT:    s_cbranch_vccnz .LBB16_1
; GFX9-FLATSCR-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; GFX9-FLATSCR-NEXT:    s_mov_b32 s0, 0
; GFX9-FLATSCR-NEXT:    s_nop 1
; GFX9-FLATSCR-NEXT:    scratch_load_dwordx2 v[0:1], off, s0 offset:4
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT:    scratch_store_dwordx2 off, v[0:1], s0
; GFX9-FLATSCR-NEXT:    s_endpgm
;
; GFX11-TRUE16-LABEL: fi_sop2_and_literal_error:
; GFX11-TRUE16:       ; %bb.0: ; %entry
; GFX11-TRUE16-NEXT:    s_mov_b32 s0, -1
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v0, 0
; GFX11-TRUE16-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11-TRUE16-NEXT:    s_mov_b32 s1, 0
; GFX11-TRUE16-NEXT:    s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-TRUE16-NEXT:    v_cmp_ne_u32_e64 s0, 1, v2
; GFX11-TRUE16-NEXT:    scratch_store_b64 off, v[0:1], s1
; GFX11-TRUE16-NEXT:  .LBB16_1: ; %.shuffle.then.i.i.i.i
; GFX11-TRUE16-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX11-TRUE16-NEXT:    s_and_b32 vcc_lo, exec_lo, s0
; GFX11-TRUE16-NEXT:    s_cbranch_vccnz .LBB16_1
; GFX11-TRUE16-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; GFX11-TRUE16-NEXT:    scratch_load_b64 v[0:1], off, off offset:4
; GFX11-TRUE16-NEXT:    s_mov_b32 s0, 0
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-TRUE16-NEXT:    scratch_store_b64 off, v[0:1], s0
; GFX11-TRUE16-NEXT:    s_endpgm
;
; GFX11-FAKE16-LABEL: fi_sop2_and_literal_error:
; GFX11-FAKE16:       ; %bb.0: ; %entry
; GFX11-FAKE16-NEXT:    s_mov_b32 s0, -1
; GFX11-FAKE16-NEXT:    v_mov_b32_e32 v0, 0
; GFX11-FAKE16-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11-FAKE16-NEXT:    s_mov_b32 s1, 0
; GFX11-FAKE16-NEXT:    s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX11-FAKE16-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-FAKE16-NEXT:    v_cmp_ne_u32_e64 s0, 1, v2
; GFX11-FAKE16-NEXT:    scratch_store_b64 off, v[0:1], s1
; GFX11-FAKE16-NEXT:  .LBB16_1: ; %.shuffle.then.i.i.i.i
; GFX11-FAKE16-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX11-FAKE16-NEXT:    s_and_b32 vcc_lo, exec_lo, s0
; GFX11-FAKE16-NEXT:    s_cbranch_vccnz .LBB16_1
; GFX11-FAKE16-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; GFX11-FAKE16-NEXT:    scratch_load_b64 v[0:1], off, off offset:4
; GFX11-FAKE16-NEXT:    s_mov_b32 s0, 0
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-FAKE16-NEXT:    scratch_store_b64 off, v[0:1], s0
; GFX11-FAKE16-NEXT:    s_endpgm
entry:
  %.omp.reduction.element.i.i.i.i = alloca [1024 x i32], align 4, addrspace(5)
  %Total3.i.i = alloca [1024 x i32], align 16, addrspace(5)
  %p2i = ptrtoint ptr addrspace(5) %Total3.i.i to i32
  br label %.shuffle.then.i.i.i.i

.shuffle.then.i.i.i.i:                            ; preds = %.shuffle.then.i.i.i.i, %entry
  store i64 0, ptr addrspace(5) null, align 4
  %or = and i32 %p2i, -512
  %icmp = icmp ugt i32 %or, 9999999
  br i1 %icmp, label %.shuffle.then.i.i.i.i, label %vector.body.i.i.i.i

vector.body.i.i.i.i:                              ; preds = %.shuffle.then.i.i.i.i
  %wide.load9.i.i.i.i = load <2 x i32>, ptr addrspace(5) %.omp.reduction.element.i.i.i.i, align 4
  store <2 x i32> %wide.load9.i.i.i.i, ptr addrspace(5) null, align 4
  ret void
}

; GCN-LABEL: {{^}}fi_sop2_or_literal_error:
; GCN: s_or_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0x3039
define amdgpu_kernel void @fi_sop2_or_literal_error() #0 {
; CI-LABEL: fi_sop2_or_literal_error:
; CI:       ; %bb.0: ; %entry
; CI-NEXT:    s_add_u32 s0, s0, s17
; CI-NEXT:    s_addc_u32 s1, s1, 0
; CI-NEXT:    v_mov_b32_e32 v0, 0
; CI-NEXT:    s_mov_b64 s[4:5], -1
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; CI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; CI-NEXT:    v_cmp_ne_u32_e64 s[4:5], 1, v0
; CI-NEXT:  .LBB17_1: ; %.shuffle.then.i.i.i.i
; CI-NEXT:    ; =>This Inner Loop Header: Depth=1
; CI-NEXT:    s_and_b64 vcc, exec, s[4:5]
; CI-NEXT:    s_cbranch_vccnz .LBB17_1
; CI-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; CI-NEXT:    buffer_load_dword v0, off, s[0:3], 0 offset:8
; CI-NEXT:    buffer_load_dword v1, off, s[0:3], 0 offset:4
; CI-NEXT:    s_waitcnt vmcnt(1)
; CI-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; CI-NEXT:    s_waitcnt vmcnt(1)
; CI-NEXT:    buffer_store_dword v1, off, s[0:3], 0
; CI-NEXT:    s_endpgm
;
; GFX9-MUBUF-LABEL: fi_sop2_or_literal_error:
; GFX9-MUBUF:       ; %bb.0: ; %entry
; GFX9-MUBUF-NEXT:    s_add_u32 s0, s0, s17
; GFX9-MUBUF-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v0, 0
; GFX9-MUBUF-NEXT:    s_mov_b64 s[4:5], -1
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GFX9-MUBUF-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9-MUBUF-NEXT:    v_cmp_ne_u32_e64 s[4:5], 1, v0
; GFX9-MUBUF-NEXT:  .LBB17_1: ; %.shuffle.then.i.i.i.i
; GFX9-MUBUF-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX9-MUBUF-NEXT:    s_and_b64 vcc, exec, s[4:5]
; GFX9-MUBUF-NEXT:    s_cbranch_vccnz .LBB17_1
; GFX9-MUBUF-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; GFX9-MUBUF-NEXT:    buffer_load_dword v0, off, s[0:3], 0 offset:8
; GFX9-MUBUF-NEXT:    buffer_load_dword v1, off, s[0:3], 0 offset:4
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(1)
; GFX9-MUBUF-NEXT:    buffer_store_dword v0, off, s[0:3], 0 offset:4
; GFX9-MUBUF-NEXT:    s_waitcnt vmcnt(1)
; GFX9-MUBUF-NEXT:    buffer_store_dword v1, off, s[0:3], 0
; GFX9-MUBUF-NEXT:    s_endpgm
;
; GFX9-FLATSCR-LABEL: fi_sop2_or_literal_error:
; GFX9-FLATSCR:       ; %bb.0: ; %entry
; GFX9-FLATSCR-NEXT:    s_add_u32 flat_scratch_lo, s8, s13
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v0, 0
; GFX9-FLATSCR-NEXT:    s_addc_u32 flat_scratch_hi, s9, 0
; GFX9-FLATSCR-NEXT:    s_mov_b32 s0, 0
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-FLATSCR-NEXT:    scratch_store_dwordx2 off, v[0:1], s0
; GFX9-FLATSCR-NEXT:    s_mov_b64 s[0:1], -1
; GFX9-FLATSCR-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; GFX9-FLATSCR-NEXT:    v_cmp_ne_u32_e64 s[0:1], 1, v0
; GFX9-FLATSCR-NEXT:  .LBB17_1: ; %.shuffle.then.i.i.i.i
; GFX9-FLATSCR-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX9-FLATSCR-NEXT:    s_and_b64 vcc, exec, s[0:1]
; GFX9-FLATSCR-NEXT:    s_cbranch_vccnz .LBB17_1
; GFX9-FLATSCR-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; GFX9-FLATSCR-NEXT:    s_mov_b32 s0, 0
; GFX9-FLATSCR-NEXT:    s_nop 1
; GFX9-FLATSCR-NEXT:    scratch_load_dwordx2 v[0:1], off, s0 offset:4
; GFX9-FLATSCR-NEXT:    s_waitcnt vmcnt(0)
; GFX9-FLATSCR-NEXT:    scratch_store_dwordx2 off, v[0:1], s0
; GFX9-FLATSCR-NEXT:    s_endpgm
;
; GFX11-TRUE16-LABEL: fi_sop2_or_literal_error:
; GFX11-TRUE16:       ; %bb.0: ; %entry
; GFX11-TRUE16-NEXT:    s_mov_b32 s0, -1
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v0, 0
; GFX11-TRUE16-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11-TRUE16-NEXT:    s_mov_b32 s1, 0
; GFX11-TRUE16-NEXT:    s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-TRUE16-NEXT:    v_cmp_ne_u32_e64 s0, 1, v2
; GFX11-TRUE16-NEXT:    scratch_store_b64 off, v[0:1], s1
; GFX11-TRUE16-NEXT:  .LBB17_1: ; %.shuffle.then.i.i.i.i
; GFX11-TRUE16-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX11-TRUE16-NEXT:    s_and_b32 vcc_lo, exec_lo, s0
; GFX11-TRUE16-NEXT:    s_cbranch_vccnz .LBB17_1
; GFX11-TRUE16-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; GFX11-TRUE16-NEXT:    scratch_load_b64 v[0:1], off, off offset:4
; GFX11-TRUE16-NEXT:    s_mov_b32 s0, 0
; GFX11-TRUE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-TRUE16-NEXT:    scratch_store_b64 off, v[0:1], s0
; GFX11-TRUE16-NEXT:    s_endpgm
;
; GFX11-FAKE16-LABEL: fi_sop2_or_literal_error:
; GFX11-FAKE16:       ; %bb.0: ; %entry
; GFX11-FAKE16-NEXT:    s_mov_b32 s0, -1
; GFX11-FAKE16-NEXT:    v_mov_b32_e32 v0, 0
; GFX11-FAKE16-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11-FAKE16-NEXT:    s_mov_b32 s1, 0
; GFX11-FAKE16-NEXT:    s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX11-FAKE16-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-FAKE16-NEXT:    v_cmp_ne_u32_e64 s0, 1, v2
; GFX11-FAKE16-NEXT:    scratch_store_b64 off, v[0:1], s1
; GFX11-FAKE16-NEXT:  .LBB17_1: ; %.shuffle.then.i.i.i.i
; GFX11-FAKE16-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX11-FAKE16-NEXT:    s_and_b32 vcc_lo, exec_lo, s0
; GFX11-FAKE16-NEXT:    s_cbranch_vccnz .LBB17_1
; GFX11-FAKE16-NEXT:  ; %bb.2: ; %vector.body.i.i.i.i
; GFX11-FAKE16-NEXT:    scratch_load_b64 v[0:1], off, off offset:4
; GFX11-FAKE16-NEXT:    s_mov_b32 s0, 0
; GFX11-FAKE16-NEXT:    s_waitcnt vmcnt(0)
; GFX11-FAKE16-NEXT:    scratch_store_b64 off, v[0:1], s0
; GFX11-FAKE16-NEXT:    s_endpgm
entry:
  %.omp.reduction.element.i.i.i.i = alloca [1024 x i32], align 4, addrspace(5)
  %Total3.i.i = alloca [1024 x i32], align 16, addrspace(5)
  %p2i = ptrtoint ptr addrspace(5) %Total3.i.i to i32
  br label %.shuffle.then.i.i.i.i

.shuffle.then.i.i.i.i:                            ; preds = %.shuffle.then.i.i.i.i, %entry
  store i64 0, ptr addrspace(5) null, align 4
  %or = or i32 %p2i, 12345
  %icmp = icmp ugt i32 %or, 9999999
  br i1 %icmp, label %.shuffle.then.i.i.i.i, label %vector.body.i.i.i.i

vector.body.i.i.i.i:                              ; preds = %.shuffle.then.i.i.i.i
  %wide.load9.i.i.i.i = load <2 x i32>, ptr addrspace(5) %.omp.reduction.element.i.i.i.i, align 4
  store <2 x i32> %wide.load9.i.i.i.i, ptr addrspace(5) null, align 4
  ret void
}

; Check that we do not produce a verifier error after prolog
; epilog. alloca1 and alloca2 will lower to literals.

; GCN-LABEL: {{^}}s_multiple_frame_indexes_literal_offsets:
; GCN: s_load_dword [[ARG0:s[0-9]+]]
; GCN: s_movk_i32 [[ALLOCA1:s[0-9]+]], 0x44
; GCN: s_cmp_eq_u32 [[ARG0]], 0
; GCN: s_cselect_b32 [[SELECT:s[0-9]+]], [[ALLOCA1]], 0x48
; GCN: s_mov_b32 [[ALLOCA0:s[0-9]+]], 0
; GCN: ; use [[SELECT]], [[ALLOCA0]]
define amdgpu_kernel void @s_multiple_frame_indexes_literal_offsets(i32 inreg %arg0) #0 {
; CI-LABEL: s_multiple_frame_indexes_literal_offsets:
; CI:       ; %bb.0:
; CI-NEXT:    s_load_dword s4, s[8:9], 0x0
; CI-NEXT:    s_add_u32 s0, s0, s17
; CI-NEXT:    s_addc_u32 s1, s1, 0
; CI-NEXT:    s_movk_i32 s5, 0x44
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_cmp_eq_u32 s4, 0
; CI-NEXT:    s_cselect_b32 s4, s5, 0x48
; CI-NEXT:    s_mov_b32 s5, 0
; CI-NEXT:    ;;#ASMSTART
; CI-NEXT:    ; use s4, s5
; CI-NEXT:    ;;#ASMEND
; CI-NEXT:    s_endpgm
;
; GFX9-MUBUF-LABEL: s_multiple_frame_indexes_literal_offsets:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_load_dword s4, s[8:9], 0x0
; GFX9-MUBUF-NEXT:    s_add_u32 s0, s0, s17
; GFX9-MUBUF-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-MUBUF-NEXT:    s_movk_i32 s5, 0x44
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_cmp_eq_u32 s4, 0
; GFX9-MUBUF-NEXT:    s_cselect_b32 s4, s5, 0x48
; GFX9-MUBUF-NEXT:    s_mov_b32 s5, 0
; GFX9-MUBUF-NEXT:    ;;#ASMSTART
; GFX9-MUBUF-NEXT:    ; use s4, s5
; GFX9-MUBUF-NEXT:    ;;#ASMEND
; GFX9-MUBUF-NEXT:    s_endpgm
;
; GFX9-FLATSCR-LABEL: s_multiple_frame_indexes_literal_offsets:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_load_dword s0, s[4:5], 0x0
; GFX9-FLATSCR-NEXT:    s_add_u32 flat_scratch_lo, s8, s13
; GFX9-FLATSCR-NEXT:    s_addc_u32 flat_scratch_hi, s9, 0
; GFX9-FLATSCR-NEXT:    s_movk_i32 s1, 0x44
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_cmp_eq_u32 s0, 0
; GFX9-FLATSCR-NEXT:    s_cselect_b32 s0, s1, 0x48
; GFX9-FLATSCR-NEXT:    s_mov_b32 s1, 0
; GFX9-FLATSCR-NEXT:    ;;#ASMSTART
; GFX9-FLATSCR-NEXT:    ; use s0, s1
; GFX9-FLATSCR-NEXT:    ;;#ASMEND
; GFX9-FLATSCR-NEXT:    s_endpgm
;
; GFX11-TRUE16-LABEL: s_multiple_frame_indexes_literal_offsets:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_load_b32 s0, s[4:5], 0x0
; GFX11-TRUE16-NEXT:    s_movk_i32 s1, 0x44
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_cmp_eq_u32 s0, 0
; GFX11-TRUE16-NEXT:    s_cselect_b32 s0, s1, 0x48
; GFX11-TRUE16-NEXT:    s_mov_b32 s1, 0
; GFX11-TRUE16-NEXT:    ;;#ASMSTART
; GFX11-TRUE16-NEXT:    ; use s0, s1
; GFX11-TRUE16-NEXT:    ;;#ASMEND
; GFX11-TRUE16-NEXT:    s_endpgm
;
; GFX11-FAKE16-LABEL: s_multiple_frame_indexes_literal_offsets:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_load_b32 s0, s[4:5], 0x0
; GFX11-FAKE16-NEXT:    s_movk_i32 s1, 0x44
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_cmp_eq_u32 s0, 0
; GFX11-FAKE16-NEXT:    s_cselect_b32 s0, s1, 0x48
; GFX11-FAKE16-NEXT:    s_mov_b32 s1, 0
; GFX11-FAKE16-NEXT:    ;;#ASMSTART
; GFX11-FAKE16-NEXT:    ; use s0, s1
; GFX11-FAKE16-NEXT:    ;;#ASMEND
; GFX11-FAKE16-NEXT:    s_endpgm
  %alloca0 = alloca [17 x i32], align 8, addrspace(5)
  %alloca1 = alloca i32, align 4, addrspace(5)
  %alloca2 = alloca i32, align 4, addrspace(5)
  %cmp = icmp eq i32 %arg0, 0
  %select = select i1 %cmp, ptr addrspace(5) %alloca1, ptr addrspace(5) %alloca2
  call void asm sideeffect "; use $0, $1","s,s"(ptr addrspace(5) %select, ptr addrspace(5) %alloca0)
  ret void
}

; %alloca1 or alloca2 will lower to an inline constant, and one will
; be a literal, so we could fold both indexes into the instruction.

; GCN-LABEL: {{^}}s_multiple_frame_indexes_one_imm_one_literal_offset:
; GCN: s_load_dword [[ARG0:s[0-9]+]]
; GCN: s_mov_b32 [[ALLOCA1:s[0-9]+]], 64
; GCN: s_cmp_eq_u32 [[ARG0]], 0
; GCN: s_cselect_b32 [[SELECT:s[0-9]+]], [[ALLOCA1]], 0x44
; GCN: s_mov_b32 [[ALLOCA0:s[0-9]+]], 0
; GCN: ; use [[SELECT]], [[ALLOCA0]]
define amdgpu_kernel void @s_multiple_frame_indexes_one_imm_one_literal_offset(i32 inreg %arg0) #0 {
; CI-LABEL: s_multiple_frame_indexes_one_imm_one_literal_offset:
; CI:       ; %bb.0:
; CI-NEXT:    s_load_dword s4, s[8:9], 0x0
; CI-NEXT:    s_add_u32 s0, s0, s17
; CI-NEXT:    s_addc_u32 s1, s1, 0
; CI-NEXT:    s_mov_b32 s5, 64
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_cmp_eq_u32 s4, 0
; CI-NEXT:    s_cselect_b32 s4, s5, 0x44
; CI-NEXT:    s_mov_b32 s5, 0
; CI-NEXT:    ;;#ASMSTART
; CI-NEXT:    ; use s4, s5
; CI-NEXT:    ;;#ASMEND
; CI-NEXT:    s_endpgm
;
; GFX9-MUBUF-LABEL: s_multiple_frame_indexes_one_imm_one_literal_offset:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_load_dword s4, s[8:9], 0x0
; GFX9-MUBUF-NEXT:    s_add_u32 s0, s0, s17
; GFX9-MUBUF-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-MUBUF-NEXT:    s_mov_b32 s5, 64
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_cmp_eq_u32 s4, 0
; GFX9-MUBUF-NEXT:    s_cselect_b32 s4, s5, 0x44
; GFX9-MUBUF-NEXT:    s_mov_b32 s5, 0
; GFX9-MUBUF-NEXT:    ;;#ASMSTART
; GFX9-MUBUF-NEXT:    ; use s4, s5
; GFX9-MUBUF-NEXT:    ;;#ASMEND
; GFX9-MUBUF-NEXT:    s_endpgm
;
; GFX9-FLATSCR-LABEL: s_multiple_frame_indexes_one_imm_one_literal_offset:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_load_dword s0, s[4:5], 0x0
; GFX9-FLATSCR-NEXT:    s_add_u32 flat_scratch_lo, s8, s13
; GFX9-FLATSCR-NEXT:    s_addc_u32 flat_scratch_hi, s9, 0
; GFX9-FLATSCR-NEXT:    s_mov_b32 s1, 64
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_cmp_eq_u32 s0, 0
; GFX9-FLATSCR-NEXT:    s_cselect_b32 s0, s1, 0x44
; GFX9-FLATSCR-NEXT:    s_mov_b32 s1, 0
; GFX9-FLATSCR-NEXT:    ;;#ASMSTART
; GFX9-FLATSCR-NEXT:    ; use s0, s1
; GFX9-FLATSCR-NEXT:    ;;#ASMEND
; GFX9-FLATSCR-NEXT:    s_endpgm
;
; GFX11-TRUE16-LABEL: s_multiple_frame_indexes_one_imm_one_literal_offset:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_load_b32 s0, s[4:5], 0x0
; GFX11-TRUE16-NEXT:    s_mov_b32 s1, 64
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_cmp_eq_u32 s0, 0
; GFX11-TRUE16-NEXT:    s_cselect_b32 s0, s1, 0x44
; GFX11-TRUE16-NEXT:    s_mov_b32 s1, 0
; GFX11-TRUE16-NEXT:    ;;#ASMSTART
; GFX11-TRUE16-NEXT:    ; use s0, s1
; GFX11-TRUE16-NEXT:    ;;#ASMEND
; GFX11-TRUE16-NEXT:    s_endpgm
;
; GFX11-FAKE16-LABEL: s_multiple_frame_indexes_one_imm_one_literal_offset:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_load_b32 s0, s[4:5], 0x0
; GFX11-FAKE16-NEXT:    s_mov_b32 s1, 64
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_cmp_eq_u32 s0, 0
; GFX11-FAKE16-NEXT:    s_cselect_b32 s0, s1, 0x44
; GFX11-FAKE16-NEXT:    s_mov_b32 s1, 0
; GFX11-FAKE16-NEXT:    ;;#ASMSTART
; GFX11-FAKE16-NEXT:    ; use s0, s1
; GFX11-FAKE16-NEXT:    ;;#ASMEND
; GFX11-FAKE16-NEXT:    s_endpgm
  %alloca0 = alloca [16 x i32], align 8, addrspace(5)
  %alloca1 = alloca i32, align 4, addrspace(5)
  %alloca2 = alloca i32, align 4, addrspace(5)
  %cmp = icmp eq i32 %arg0, 0
  %select = select i1 %cmp, ptr addrspace(5) %alloca1, ptr addrspace(5) %alloca2
  call void asm sideeffect "; use $0, $1","s,s"(ptr addrspace(5) %select, ptr addrspace(5) %alloca0)
  ret void
}

; GCN-LABEL: {{^}}s_multiple_frame_indexes_imm_offsets:
; GCN: s_load_dword [[ARG0:s[0-9]+]]
; GCN: s_mov_b32 [[ALLOCA1:s[0-9]+]], 16
; GCN: s_cmp_eq_u32 [[ARG0]], 0
; GCN: s_cselect_b32 [[SELECT:s[0-9]+]], [[ALLOCA1]], 20
; GCN: s_mov_b32 [[ALLOCA0:s[0-9]+]], 0
; GCN: ; use [[SELECT]], [[ALLOCA0]]
define amdgpu_kernel void @s_multiple_frame_indexes_imm_offsets(i32 inreg %arg0) #0 {
; CI-LABEL: s_multiple_frame_indexes_imm_offsets:
; CI:       ; %bb.0:
; CI-NEXT:    s_load_dword s4, s[8:9], 0x0
; CI-NEXT:    s_add_u32 s0, s0, s17
; CI-NEXT:    s_addc_u32 s1, s1, 0
; CI-NEXT:    s_mov_b32 s5, 16
; CI-NEXT:    s_waitcnt lgkmcnt(0)
; CI-NEXT:    s_cmp_eq_u32 s4, 0
; CI-NEXT:    s_cselect_b32 s4, s5, 20
; CI-NEXT:    s_mov_b32 s5, 0
; CI-NEXT:    ;;#ASMSTART
; CI-NEXT:    ; use s4, s5
; CI-NEXT:    ;;#ASMEND
; CI-NEXT:    s_endpgm
;
; GFX9-MUBUF-LABEL: s_multiple_frame_indexes_imm_offsets:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_load_dword s4, s[8:9], 0x0
; GFX9-MUBUF-NEXT:    s_add_u32 s0, s0, s17
; GFX9-MUBUF-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-MUBUF-NEXT:    s_mov_b32 s5, 16
; GFX9-MUBUF-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-MUBUF-NEXT:    s_cmp_eq_u32 s4, 0
; GFX9-MUBUF-NEXT:    s_cselect_b32 s4, s5, 20
; GFX9-MUBUF-NEXT:    s_mov_b32 s5, 0
; GFX9-MUBUF-NEXT:    ;;#ASMSTART
; GFX9-MUBUF-NEXT:    ; use s4, s5
; GFX9-MUBUF-NEXT:    ;;#ASMEND
; GFX9-MUBUF-NEXT:    s_endpgm
;
; GFX9-FLATSCR-LABEL: s_multiple_frame_indexes_imm_offsets:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_load_dword s0, s[4:5], 0x0
; GFX9-FLATSCR-NEXT:    s_add_u32 flat_scratch_lo, s8, s13
; GFX9-FLATSCR-NEXT:    s_addc_u32 flat_scratch_hi, s9, 0
; GFX9-FLATSCR-NEXT:    s_mov_b32 s1, 16
; GFX9-FLATSCR-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-NEXT:    s_cmp_eq_u32 s0, 0
; GFX9-FLATSCR-NEXT:    s_cselect_b32 s0, s1, 20
; GFX9-FLATSCR-NEXT:    s_mov_b32 s1, 0
; GFX9-FLATSCR-NEXT:    ;;#ASMSTART
; GFX9-FLATSCR-NEXT:    ; use s0, s1
; GFX9-FLATSCR-NEXT:    ;;#ASMEND
; GFX9-FLATSCR-NEXT:    s_endpgm
;
; GFX11-TRUE16-LABEL: s_multiple_frame_indexes_imm_offsets:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    s_load_b32 s0, s[4:5], 0x0
; GFX11-TRUE16-NEXT:    s_mov_b32 s1, 16
; GFX11-TRUE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-TRUE16-NEXT:    s_cmp_eq_u32 s0, 0
; GFX11-TRUE16-NEXT:    s_cselect_b32 s0, s1, 20
; GFX11-TRUE16-NEXT:    s_mov_b32 s1, 0
; GFX11-TRUE16-NEXT:    ;;#ASMSTART
; GFX11-TRUE16-NEXT:    ; use s0, s1
; GFX11-TRUE16-NEXT:    ;;#ASMEND
; GFX11-TRUE16-NEXT:    s_endpgm
;
; GFX11-FAKE16-LABEL: s_multiple_frame_indexes_imm_offsets:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    s_load_b32 s0, s[4:5], 0x0
; GFX11-FAKE16-NEXT:    s_mov_b32 s1, 16
; GFX11-FAKE16-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-FAKE16-NEXT:    s_cmp_eq_u32 s0, 0
; GFX11-FAKE16-NEXT:    s_cselect_b32 s0, s1, 20
; GFX11-FAKE16-NEXT:    s_mov_b32 s1, 0
; GFX11-FAKE16-NEXT:    ;;#ASMSTART
; GFX11-FAKE16-NEXT:    ; use s0, s1
; GFX11-FAKE16-NEXT:    ;;#ASMEND
; GFX11-FAKE16-NEXT:    s_endpgm
  %alloca0 = alloca [4 x i32], align 8, addrspace(5)
  %alloca1 = alloca i32, align 4, addrspace(5)
  %alloca2 = alloca i32, align 4, addrspace(5)
  %cmp = icmp eq i32 %arg0, 0
  %select = select i1 %cmp, ptr addrspace(5) %alloca1, ptr addrspace(5) %alloca2
  call void asm sideeffect "; use $0, $1","s,s"(ptr addrspace(5) %select, ptr addrspace(5) %alloca0)
  ret void
}

; GCN-LABEL: {{^}}v_multiple_frame_indexes_literal_offsets:
; GCN: v_mov_b32_e32 [[ALLOCA1:v[0-9]+]], 0x48
; GCN: v_mov_b32_e32 [[ALLOCA2:v[0-9]+]], 0x44
; GCN: v_cmp_eq_u32_e32 vcc, 0, v0
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[ALLOCA1]], [[ALLOCA2]], vcc
; GCN: v_mov_b32_e32 [[ALLOCA0:v[0-9]+]], 0{{$}}
; GCN: ; use [[SELECT]], [[ALLOCA0]]
define amdgpu_kernel void @v_multiple_frame_indexes_literal_offsets() #0 {
; CI-LABEL: v_multiple_frame_indexes_literal_offsets:
; CI:       ; %bb.0:
; CI-NEXT:    s_add_u32 s0, s0, s17
; CI-NEXT:    v_mov_b32_e32 v1, 0x48
; CI-NEXT:    v_mov_b32_e32 v2, 0x44
; CI-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; CI-NEXT:    s_addc_u32 s1, s1, 0
; CI-NEXT:    v_cndmask_b32_e32 v0, v1, v2, vcc
; CI-NEXT:    v_mov_b32_e32 v1, 0
; CI-NEXT:    ;;#ASMSTART
; CI-NEXT:    ; use v0, v1
; CI-NEXT:    ;;#ASMEND
; CI-NEXT:    s_endpgm
;
; GFX9-MUBUF-LABEL: v_multiple_frame_indexes_literal_offsets:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_add_u32 s0, s0, s17
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v1, 0x48
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v2, 0x44
; GFX9-MUBUF-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; GFX9-MUBUF-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-MUBUF-NEXT:    v_cndmask_b32_e32 v0, v1, v2, vcc
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v1, 0
; GFX9-MUBUF-NEXT:    ;;#ASMSTART
; GFX9-MUBUF-NEXT:    ; use v0, v1
; GFX9-MUBUF-NEXT:    ;;#ASMEND
; GFX9-MUBUF-NEXT:    s_endpgm
;
; GFX9-FLATSCR-LABEL: v_multiple_frame_indexes_literal_offsets:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_add_u32 flat_scratch_lo, s8, s13
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v1, 0x48
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v2, 0x44
; GFX9-FLATSCR-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; GFX9-FLATSCR-NEXT:    s_addc_u32 flat_scratch_hi, s9, 0
; GFX9-FLATSCR-NEXT:    v_cndmask_b32_e32 v0, v1, v2, vcc
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v1, 0
; GFX9-FLATSCR-NEXT:    ;;#ASMSTART
; GFX9-FLATSCR-NEXT:    ; use v0, v1
; GFX9-FLATSCR-NEXT:    ;;#ASMEND
; GFX9-FLATSCR-NEXT:    s_endpgm
;
; GFX11-TRUE16-LABEL: v_multiple_frame_indexes_literal_offsets:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    v_and_b32_e32 v0, 0x3ff, v0
; GFX11-TRUE16-NEXT:    v_mov_b32_e32 v1, 0x44
; GFX11-TRUE16-NEXT:    s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX11-TRUE16-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 0, v0
; GFX11-TRUE16-NEXT:    v_dual_cndmask_b32 v0, 0x48, v1 :: v_dual_mov_b32 v1, 0
; GFX11-TRUE16-NEXT:    ;;#ASMSTART
; GFX11-TRUE16-NEXT:    ; use v0, v1
; GFX11-TRUE16-NEXT:    ;;#ASMEND
; GFX11-TRUE16-NEXT:    s_endpgm
;
; GFX11-FAKE16-LABEL: v_multiple_frame_indexes_literal_offsets:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    v_and_b32_e32 v0, 0x3ff, v0
; GFX11-FAKE16-NEXT:    v_mov_b32_e32 v1, 0x44
; GFX11-FAKE16-NEXT:    s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX11-FAKE16-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 0, v0
; GFX11-FAKE16-NEXT:    v_dual_cndmask_b32 v0, 0x48, v1 :: v_dual_mov_b32 v1, 0
; GFX11-FAKE16-NEXT:    ;;#ASMSTART
; GFX11-FAKE16-NEXT:    ; use v0, v1
; GFX11-FAKE16-NEXT:    ;;#ASMEND
; GFX11-FAKE16-NEXT:    s_endpgm
  %vgpr = call i32 @llvm.amdgcn.workitem.id.x()
  %alloca0 = alloca [17 x i32], align 8, addrspace(5)
  %alloca1 = alloca i32, align 4, addrspace(5)
  %alloca2 = alloca i32, align 4, addrspace(5)
  %cmp = icmp eq i32 %vgpr, 0
  %select = select i1 %cmp, ptr addrspace(5) %alloca1, ptr addrspace(5) %alloca2
  call void asm sideeffect "; use $0, $1","v,v"(ptr addrspace(5) %select, ptr addrspace(5) %alloca0)
  ret void
}

; GCN-LABEL: {{^}}v_multiple_frame_indexes_one_imm_one_literal_offset:
; GCN: v_mov_b32_e32 [[ALLOCA1:v[0-9]+]], 0x44
; GCN: v_mov_b32_e32 [[ALLOCA2:v[0-9]+]], 64
; GCN: v_cmp_eq_u32_e32 vcc, 0, v0
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[ALLOCA1]], [[ALLOCA2]], vcc
; GCN: v_mov_b32_e32 [[ALLOCA0:v[0-9]+]], 0{{$}}
; GCN: ; use [[SELECT]], [[ALLOCA0]]
define amdgpu_kernel void @v_multiple_frame_indexes_one_imm_one_literal_offset() #0 {
; CI-LABEL: v_multiple_frame_indexes_one_imm_one_literal_offset:
; CI:       ; %bb.0:
; CI-NEXT:    s_add_u32 s0, s0, s17
; CI-NEXT:    v_mov_b32_e32 v1, 0x44
; CI-NEXT:    v_mov_b32_e32 v2, 64
; CI-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; CI-NEXT:    s_addc_u32 s1, s1, 0
; CI-NEXT:    v_cndmask_b32_e32 v0, v1, v2, vcc
; CI-NEXT:    v_mov_b32_e32 v1, 0
; CI-NEXT:    ;;#ASMSTART
; CI-NEXT:    ; use v0, v1
; CI-NEXT:    ;;#ASMEND
; CI-NEXT:    s_endpgm
;
; GFX9-MUBUF-LABEL: v_multiple_frame_indexes_one_imm_one_literal_offset:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_add_u32 s0, s0, s17
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v1, 0x44
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v2, 64
; GFX9-MUBUF-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; GFX9-MUBUF-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-MUBUF-NEXT:    v_cndmask_b32_e32 v0, v1, v2, vcc
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v1, 0
; GFX9-MUBUF-NEXT:    ;;#ASMSTART
; GFX9-MUBUF-NEXT:    ; use v0, v1
; GFX9-MUBUF-NEXT:    ;;#ASMEND
; GFX9-MUBUF-NEXT:    s_endpgm
;
; GFX9-FLATSCR-LABEL: v_multiple_frame_indexes_one_imm_one_literal_offset:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_add_u32 flat_scratch_lo, s8, s13
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v1, 0x44
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v2, 64
; GFX9-FLATSCR-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; GFX9-FLATSCR-NEXT:    s_addc_u32 flat_scratch_hi, s9, 0
; GFX9-FLATSCR-NEXT:    v_cndmask_b32_e32 v0, v1, v2, vcc
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v1, 0
; GFX9-FLATSCR-NEXT:    ;;#ASMSTART
; GFX9-FLATSCR-NEXT:    ; use v0, v1
; GFX9-FLATSCR-NEXT:    ;;#ASMEND
; GFX9-FLATSCR-NEXT:    s_endpgm
;
; GFX11-TRUE16-LABEL: v_multiple_frame_indexes_one_imm_one_literal_offset:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    v_dual_mov_b32 v1, 64 :: v_dual_and_b32 v0, 0x3ff, v0
; GFX11-TRUE16-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX11-TRUE16-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 0, v0
; GFX11-TRUE16-NEXT:    v_dual_cndmask_b32 v0, 0x44, v1 :: v_dual_mov_b32 v1, 0
; GFX11-TRUE16-NEXT:    ;;#ASMSTART
; GFX11-TRUE16-NEXT:    ; use v0, v1
; GFX11-TRUE16-NEXT:    ;;#ASMEND
; GFX11-TRUE16-NEXT:    s_endpgm
;
; GFX11-FAKE16-LABEL: v_multiple_frame_indexes_one_imm_one_literal_offset:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    v_dual_mov_b32 v1, 64 :: v_dual_and_b32 v0, 0x3ff, v0
; GFX11-FAKE16-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX11-FAKE16-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 0, v0
; GFX11-FAKE16-NEXT:    v_dual_cndmask_b32 v0, 0x44, v1 :: v_dual_mov_b32 v1, 0
; GFX11-FAKE16-NEXT:    ;;#ASMSTART
; GFX11-FAKE16-NEXT:    ; use v0, v1
; GFX11-FAKE16-NEXT:    ;;#ASMEND
; GFX11-FAKE16-NEXT:    s_endpgm
  %vgpr = call i32 @llvm.amdgcn.workitem.id.x()
  %alloca0 = alloca [16 x i32], align 8, addrspace(5)
  %alloca1 = alloca i32, align 4, addrspace(5)
  %alloca2 = alloca i32, align 4, addrspace(5)
  %cmp = icmp eq i32 %vgpr, 0
  %select = select i1 %cmp, ptr addrspace(5) %alloca1, ptr addrspace(5) %alloca2
  call void asm sideeffect "; use $0, $1","v,v"(ptr addrspace(5) %select, ptr addrspace(5) %alloca0)
  ret void
}

; GCN-LABEL: {{^}}v_multiple_frame_indexes_imm_offsets:
; GCN: v_mov_b32_e32 [[ALLOCA1:v[0-9]+]], 12
; GCN: v_mov_b32_e32 [[ALLOCA2:v[0-9]+]], 8
; GCN: v_cmp_eq_u32_e32 vcc, 0, v0
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[ALLOCA1]], [[ALLOCA2]], vcc
; GCN: v_mov_b32_e32 [[ALLOCA0:v[0-9]+]], 0{{$}}
; GCN: ; use [[SELECT]], [[ALLOCA0]]
define amdgpu_kernel void @v_multiple_frame_indexes_imm_offsets() #0 {
; CI-LABEL: v_multiple_frame_indexes_imm_offsets:
; CI:       ; %bb.0:
; CI-NEXT:    s_add_u32 s0, s0, s17
; CI-NEXT:    v_mov_b32_e32 v1, 12
; CI-NEXT:    v_mov_b32_e32 v2, 8
; CI-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; CI-NEXT:    s_addc_u32 s1, s1, 0
; CI-NEXT:    v_cndmask_b32_e32 v0, v1, v2, vcc
; CI-NEXT:    v_mov_b32_e32 v1, 0
; CI-NEXT:    ;;#ASMSTART
; CI-NEXT:    ; use v0, v1
; CI-NEXT:    ;;#ASMEND
; CI-NEXT:    s_endpgm
;
; GFX9-MUBUF-LABEL: v_multiple_frame_indexes_imm_offsets:
; GFX9-MUBUF:       ; %bb.0:
; GFX9-MUBUF-NEXT:    s_add_u32 s0, s0, s17
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v1, 12
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v2, 8
; GFX9-MUBUF-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; GFX9-MUBUF-NEXT:    s_addc_u32 s1, s1, 0
; GFX9-MUBUF-NEXT:    v_cndmask_b32_e32 v0, v1, v2, vcc
; GFX9-MUBUF-NEXT:    v_mov_b32_e32 v1, 0
; GFX9-MUBUF-NEXT:    ;;#ASMSTART
; GFX9-MUBUF-NEXT:    ; use v0, v1
; GFX9-MUBUF-NEXT:    ;;#ASMEND
; GFX9-MUBUF-NEXT:    s_endpgm
;
; GFX9-FLATSCR-LABEL: v_multiple_frame_indexes_imm_offsets:
; GFX9-FLATSCR:       ; %bb.0:
; GFX9-FLATSCR-NEXT:    s_add_u32 flat_scratch_lo, s8, s13
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v1, 12
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v2, 8
; GFX9-FLATSCR-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; GFX9-FLATSCR-NEXT:    s_addc_u32 flat_scratch_hi, s9, 0
; GFX9-FLATSCR-NEXT:    v_cndmask_b32_e32 v0, v1, v2, vcc
; GFX9-FLATSCR-NEXT:    v_mov_b32_e32 v1, 0
; GFX9-FLATSCR-NEXT:    ;;#ASMSTART
; GFX9-FLATSCR-NEXT:    ; use v0, v1
; GFX9-FLATSCR-NEXT:    ;;#ASMEND
; GFX9-FLATSCR-NEXT:    s_endpgm
;
; GFX11-TRUE16-LABEL: v_multiple_frame_indexes_imm_offsets:
; GFX11-TRUE16:       ; %bb.0:
; GFX11-TRUE16-NEXT:    v_dual_mov_b32 v1, 8 :: v_dual_and_b32 v0, 0x3ff, v0
; GFX11-TRUE16-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX11-TRUE16-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 0, v0
; GFX11-TRUE16-NEXT:    v_dual_cndmask_b32 v0, 12, v1 :: v_dual_mov_b32 v1, 0
; GFX11-TRUE16-NEXT:    ;;#ASMSTART
; GFX11-TRUE16-NEXT:    ; use v0, v1
; GFX11-TRUE16-NEXT:    ;;#ASMEND
; GFX11-TRUE16-NEXT:    s_endpgm
;
; GFX11-FAKE16-LABEL: v_multiple_frame_indexes_imm_offsets:
; GFX11-FAKE16:       ; %bb.0:
; GFX11-FAKE16-NEXT:    v_dual_mov_b32 v1, 8 :: v_dual_and_b32 v0, 0x3ff, v0
; GFX11-FAKE16-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX11-FAKE16-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 0, v0
; GFX11-FAKE16-NEXT:    v_dual_cndmask_b32 v0, 12, v1 :: v_dual_mov_b32 v1, 0
; GFX11-FAKE16-NEXT:    ;;#ASMSTART
; GFX11-FAKE16-NEXT:    ; use v0, v1
; GFX11-FAKE16-NEXT:    ;;#ASMEND
; GFX11-FAKE16-NEXT:    s_endpgm
  %vgpr = call i32 @llvm.amdgcn.workitem.id.x()
  %alloca0 = alloca [2 x i32], align 8, addrspace(5)
  %alloca1 = alloca i32, align 4, addrspace(5)
  %alloca2 = alloca i32, align 4, addrspace(5)
  %cmp = icmp eq i32 %vgpr, 0
  %select = select i1 %cmp, ptr addrspace(5) %alloca1, ptr addrspace(5) %alloca2
  call void asm sideeffect "; use $0, $1","v,v"(ptr addrspace(5) %select, ptr addrspace(5) %alloca0)
  ret void
}

attributes #0 = { nounwind }
;; NOTE: These prefixes are unused and the list is autogenerated. Do not add tests below this line:
; GCN: {{.*}}
; GFX9: {{.*}}
; MUBUF: {{.*}}
