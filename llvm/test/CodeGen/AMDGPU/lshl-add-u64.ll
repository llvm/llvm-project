; RUN: llc -mtriple=amdgcn -mcpu=gfx942 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

define i64 @lshl_add_u64_v1v(i64 %v, i64 %a) {
; GCN-LABEL: lshl_add_u64_v1v:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_lshl_add_u64 v[0:1], v[0:1], 1, v[2:3]
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %shl = shl i64 %v, 1
  %add = add i64 %shl, %a
  ret i64 %add
}

define i64 @lshl_add_u64_v4v(i64 %v, i64 %a) {
; GCN-LABEL: lshl_add_u64_v4v:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_lshl_add_u64 v[0:1], v[0:1], 4, v[2:3]
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %shl = shl i64 %v, 4
  %add = add i64 %shl, %a
  ret i64 %add
}

define i64 @lshl_add_u64_v5v(i64 %v, i64 %a) {
; GCN-LABEL: lshl_add_u64_v5v:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_lshl_add_u64 v[0:1], v[0:1], 5, v[2:3]
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %shl = shl i64 %v, 5
  %add = add i64 %shl, %a
  ret i64 %add
}

define i64 @lshl_add_u64_vvv(i64 %v, i64 %s, i64 %a) {
; GCN-LABEL: lshl_add_u64_vvv:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_lshl_add_u64 v[0:1], v[0:1], v2, v[4:5]
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %shl = shl i64 %v, %s
  %add = add i64 %shl, %a
  ret i64 %add
}

define amdgpu_kernel void @lshl_add_u64_s2v(i64 %v) {
; GCN-LABEL: lshl_add_u64_s2v:
; GCN:       ; %bb.0:
; GCN-NEXT:    flat_load_dwordx2 v[0:1], v[0:1]
; GCN-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; GCN-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_lshl_add_u64 v[0:1], s[0:1], 2, v[0:1]
; GCN-NEXT:    flat_store_dwordx2 v[0:1], v[0:1] sc0 sc1
; GCN-NEXT:    s_endpgm
  %a = load i64, ptr undef
  %shl = shl i64 %v, 2
  %add = add i64 %shl, %a
  store i64 %add, ptr undef
  ret void
}

define amdgpu_kernel void @lshl_add_u64_v2s(i64 %a) {
; GCN-LABEL: lshl_add_u64_v2s:
; GCN:       ; %bb.0:
; GCN-NEXT:    flat_load_dwordx2 v[0:1], v[0:1]
; GCN-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; GCN-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_lshl_add_u64 v[0:1], v[0:1], 2, s[0:1]
; GCN-NEXT:    flat_store_dwordx2 v[0:1], v[0:1] sc0 sc1
; GCN-NEXT:    s_endpgm
  %v = load i64, ptr undef
  %shl = shl i64 %v, 2
  %add = add i64 %shl, %a
  store i64 %add, ptr undef
  ret void
}

define amdgpu_kernel void @lshl_add_u64_s2s(i64 %v, i64 %a) {
; GCN-LABEL: lshl_add_u64_s2s:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_mov_b32_e32 v0, s2
; GCN-NEXT:    v_mov_b32_e32 v1, s3
; GCN-NEXT:    v_lshl_add_u64 v[0:1], s[0:1], 2, v[0:1]
; GCN-NEXT:    flat_store_dwordx2 v[0:1], v[0:1] sc0 sc1
; GCN-NEXT:    s_endpgm
  %shl = shl i64 %v, 2
  %add = add i64 %shl, %a
  store i64 %add, ptr undef
  ret void
}

define i64 @add_u64_vv(i64 %v, i64 %a) {
; GCN-LABEL: add_u64_vv:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %add = add i64 %v, %a
  ret i64 %add
}

define amdgpu_kernel void @add_u64_sv(i64 %v) {
; GCN-LABEL: add_u64_sv:
; GCN:       ; %bb.0:
; GCN-NEXT:    flat_load_dwordx2 v[0:1], v[0:1]
; GCN-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; GCN-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_lshl_add_u64 v[0:1], s[0:1], 0, v[0:1]
; GCN-NEXT:    flat_store_dwordx2 v[0:1], v[0:1] sc0 sc1
; GCN-NEXT:    s_endpgm
  %a = load i64, ptr undef
  %add = add i64 %v, %a
  store i64 %add, ptr undef
  ret void
}

define amdgpu_kernel void @add_u64_vs(i64 %a) {
; GCN-LABEL: add_u64_vs:
; GCN:       ; %bb.0:
; GCN-NEXT:    flat_load_dwordx2 v[0:1], v[0:1]
; GCN-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; GCN-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_lshl_add_u64 v[0:1], v[0:1], 0, s[0:1]
; GCN-NEXT:    flat_store_dwordx2 v[0:1], v[0:1] sc0 sc1
; GCN-NEXT:    s_endpgm
  %v = load i64, ptr undef
  %add = add i64 %v, %a
  store i64 %add, ptr undef
  ret void
}

define amdgpu_kernel void @add_u64_ss(i64 %v, i64 %a) {
; GCN-LABEL: add_u64_ss:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_add_u32 s0, s0, s2
; GCN-NEXT:    s_addc_u32 s1, s1, s3
; GCN-NEXT:    v_mov_b64_e32 v[0:1], s[0:1]
; GCN-NEXT:    flat_store_dwordx2 v[0:1], v[0:1] sc0 sc1
; GCN-NEXT:    s_endpgm
  %add = add i64 %v, %a
  store i64 %add, ptr undef
  ret void
}

define i32 @lshl_add_u64_gep(ptr %p, i64 %a) {
; GCN-LABEL: lshl_add_u64_gep:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_lshl_add_u64 v[0:1], v[2:3], 2, v[0:1]
; GCN-NEXT:    flat_load_dword v0, v[0:1]
; GCN-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %gep = getelementptr inbounds i32, ptr %p, i64 %a
  %v = load i32, ptr %gep
  ret i32 %v
}
