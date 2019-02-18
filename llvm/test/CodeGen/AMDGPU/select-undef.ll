; RUN: llc -mtriple=amdgcn-amd-amdhsa < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}select_undef_lhs:
; GCN: s_waitcnt
; GCN-NOT: v_cmp
; GCN-NOT: v_cndmask
; GCN-NEXT: s_setpc_b64
define float @select_undef_lhs(float %val, i1 %cond) {
  %undef = call float @llvm.amdgcn.rcp.f32(float undef)
  %sel = select i1 %cond, float %undef, float %val
  ret float %sel
}

; GCN-LABEL: {{^}}select_undef_rhs:
; GCN: s_waitcnt
; GCN-NOT: v_cmp
; GCN-NOT: v_cndmask
; GCN-NEXT: s_setpc_b64
define float @select_undef_rhs(float %val, i1 %cond) {
  %undef = call float @llvm.amdgcn.rcp.f32(float undef)
  %sel = select i1 %cond, float %val, float %undef
  ret float %sel
}

; GCN-LABEL: {{^}}select_undef_n1:
; GCN: v_mov_b32_e32 [[RES:v[0-9]+]], 1.0
; GCN: store_dword {{[^,]+}}, [[RES]]
define void @select_undef_n1(float addrspace(1)* %a, i32 %c) {
  %cc = icmp eq i32 %c, 0
  %sel = select i1 %cc, float 1.000000e+00, float undef
  store float %sel, float addrspace(1)* %a
  ret void
}

; GCN-LABEL: {{^}}select_undef_n2:
; GCN: v_mov_b32_e32 [[RES:v[0-9]+]], 1.0
; GCN: store_dword {{[^,]+}}, [[RES]]
define void @select_undef_n2(float addrspace(1)* %a, i32 %c) {
  %cc = icmp eq i32 %c, 0
  %sel = select i1 %cc, float undef, float 1.000000e+00
  store float %sel, float addrspace(1)* %a
  ret void
}

declare float @llvm.amdgcn.rcp.f32(float)
