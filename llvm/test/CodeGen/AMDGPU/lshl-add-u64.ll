; RUN: llc -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

define i64 @lshl_add_u64_v1v(i64 %v, i64 %a) {
; GCN-LABEL: lshl_add_u64_v1v:
; GCN: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 1, v[{{[0-9:]+}}]
  %shl = shl i64 %v, 1
  %add = add i64 %shl, %a
  ret i64 %add
}

define i64 @lshl_add_u64_v4v(i64 %v, i64 %a) {
; GCN-LABEL: lshl_add_u64_v4v:
; GCN: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 4, v[{{[0-9:]+}}]
  %shl = shl i64 %v, 4
  %add = add i64 %shl, %a
  ret i64 %add
}

define i64 @lshl_add_u64_v5v(i64 %v, i64 %a) {
; GCN-LABEL: lshl_add_u64_v5v:
; GCN:      v_lshlrev_b64
; GCN-NEXT: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 0, v[{{[0-9:]+}}]
  %shl = shl i64 %v, 5
  %add = add i64 %shl, %a
  ret i64 %add
}

define i64 @lshl_add_u64_vvv(i64 %v, i64 %s, i64 %a) {
; GCN-LABEL: lshl_add_u64_vvv:
; GCN:      v_lshlrev_b64
; GCN-NEXT: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 0, v[{{[0-9:]+}}]
  %shl = shl i64 %v, %s
  %add = add i64 %shl, %a
  ret i64 %add
}

define amdgpu_kernel void @lshl_add_u64_s2v(i64 %v) {
; GCN-LABEL: lshl_add_u64_s2v:
; GCN: v_lshl_add_u64 v[{{[0-9:]+}}], s[{{[0-9:]+}}], 2, v[{{[0-9:]+}}]
  %a = load i64, ptr undef
  %shl = shl i64 %v, 2
  %add = add i64 %shl, %a
  store i64 %add, ptr undef
  ret void
}

define amdgpu_kernel void @lshl_add_u64_v2s(i64 %a) {
; GCN-LABEL: lshl_add_u64_v2s:
; GCN: v_lshl_add_u64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 2, s[{{[0-9:]+}}]
  %v = load i64, ptr undef
  %shl = shl i64 %v, 2
  %add = add i64 %shl, %a
  store i64 %add, ptr undef
  ret void
}

define amdgpu_kernel void @lshl_add_u64_s2s(i64 %v, i64 %a) {
; GCN-LABEL: lshl_add_u64_s2s:
; GCN:    s_lshl_b64
; GCN:    s_add_u32
; GCN:    s_addc_u32
  %shl = shl i64 %v, 2
  %add = add i64 %shl, %a
  store i64 %add, ptr undef
  ret void
}

define i64 @add_u64_vv(i64 %v, i64 %a) {
; GCN-LABEL: add_u64_vv:
; GCN: v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
  %add = add i64 %v, %a
  ret i64 %add
}

define amdgpu_kernel void @add_u64_sv(i64 %v) {
; GCN-LABEL: add_u64_sv:
; GCN: v_lshl_add_u64 v[0:1], s[0:1], 0, v[0:1]
  %a = load i64, ptr undef
  %add = add i64 %v, %a
  store i64 %add, ptr undef
  ret void
}

define amdgpu_kernel void @add_u64_vs(i64 %a) {
; GCN-LABEL: add_u64_vs:
; GCN: v_lshl_add_u64 v[0:1], v[0:1], 0, s[0:1]
  %v = load i64, ptr undef
  %add = add i64 %v, %a
  store i64 %add, ptr undef
  ret void
}

define amdgpu_kernel void @add_u64_ss(i64 %v, i64 %a) {
; GCN-LABEL: add_u64_ss:
; GCN: s_add_u32
; GCN: s_addc_u32 s1, s1, s3
  %add = add i64 %v, %a
  store i64 %add, ptr undef
  ret void
}

define i32 @lshl_add_u64_gep(ptr %p, i64 %a) {
; GCN-LABEL: lshl_add_u64_gep:
; GCN: v_lshl_add_u64 v[0:1], v[2:3], 2, v[0:1]
  %gep = getelementptr inbounds i32, ptr %p, i64 %a
  %v = load i32, ptr %gep
  ret i32 %v
}
