; RUN: llc < %s -march=amdgcn -mcpu=tahiti -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: {{^}}select0:
; i64 select should be split into two i32 selects, and we shouldn't need
; to use a shfit to extract the hi dword of the input.
; CHECK-NOT: s_lshr_b64
; CHECK: v_cndmask
; CHECK: v_cndmask
define amdgpu_kernel void @select0(i64 addrspace(1)* %out, i32 %cond, i64 %in) {
entry:
  %0 = icmp ugt i32 %cond, 5
  %1 = select i1 %0, i64 0, i64 %in
  store i64 %1, i64 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}select_trunc_i64:
; CHECK: s_cselect_b32
; CHECK-NOT: s_cselect_b32
define amdgpu_kernel void @select_trunc_i64(i32 addrspace(1)* %out, i32 %cond, i64 %in) nounwind {
  %cmp = icmp ugt i32 %cond, 5
  %sel = select i1 %cmp, i64 0, i64 %in
  %trunc = trunc i64 %sel to i32
  store i32 %trunc, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}select_trunc_i64_2:
; CHECK: s_cselect_b32
; CHECK-NOT: s_cselect_b32
define amdgpu_kernel void @select_trunc_i64_2(i32 addrspace(1)* %out, i32 %cond, i64 %a, i64 %b) nounwind {
  %cmp = icmp ugt i32 %cond, 5
  %sel = select i1 %cmp, i64 %a, i64 %b
  %trunc = trunc i64 %sel to i32
  store i32 %trunc, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}v_select_trunc_i64_2:
; CHECK: s_cselect_b32
; CHECK-NOT: s_cselect_b32
define amdgpu_kernel void @v_select_trunc_i64_2(i32 addrspace(1)* %out, i32 %cond, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) nounwind {
  %cmp = icmp ugt i32 %cond, 5
  %a = load i64, i64 addrspace(1)* %aptr, align 8
  %b = load i64, i64 addrspace(1)* %bptr, align 8
  %sel = select i1 %cmp, i64 %a, i64 %b
  %trunc = trunc i64 %sel to i32
  store i32 %trunc, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}v_select_i64_split_imm:
; CHECK-DAG: v_cndmask_b32_e32 {{v[0-9]+}}, 0, {{v[0-9]+}}
; CHECK-DAG: v_cndmask_b32_e32 {{v[0-9]+}}, 63, {{v[0-9]+}}
; CHECK: s_endpgm
define amdgpu_kernel void @v_select_i64_split_imm(i64 addrspace(1)* %out, i32 %cond, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) nounwind {
  %cmp = icmp ugt i32 %cond, 5
  %a = load i64, i64 addrspace(1)* %aptr, align 8
  %b = load i64, i64 addrspace(1)* %bptr, align 8
  %sel = select i1 %cmp, i64 %a, i64 270582939648 ; 63 << 32
  store i64 %sel, i64 addrspace(1)* %out, align 8
  ret void
}
