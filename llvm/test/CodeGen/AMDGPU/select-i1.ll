; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; FIXME: This should go in existing select.ll test, except the current testcase there is broken on GCN

; GCN-LABEL: {{^}}select_i1:
; GCN: v_cndmask_b32
; GCN-NOT: v_cndmask_b32
define amdgpu_kernel void @select_i1(i1 addrspace(1)* %out, i32 %cond, i1 %a, i1 %b) nounwind {
  %cmp = icmp ugt i32 %cond, 5
  %sel = select i1 %cmp, i1 %a, i1 %b
  store i1 %sel, i1 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}s_minmax_i1:
; GCN: s_load_dword [[LOAD:s[0-9]+]],
; GCN-DAG: s_lshr_b32 [[A:s[0-9]+]], [[LOAD]], 8
; GCN-DAG: s_lshr_b32 [[B:s[0-9]+]], [[LOAD]], 16
; GCN-DAG: s_and_b32 [[COND:s[0-9]+]], 1, [[LOAD]]
; GCN-DAG: v_mov_b32_e32 [[V_A:v[0-9]+]], [[A]]
; GCN-DAG: v_mov_b32_e32 [[V_B:v[0-9]+]], [[B]]
; GCN: v_cmp_eq_u32_e64 vcc, [[COND]], 1
; GCN: v_cndmask_b32_e32 [[SEL:v[0-9]+]], [[V_B]], [[V_A]]
; GCN: v_and_b32_e32 v{{[0-9]+}}, 1, [[SEL]]
define amdgpu_kernel void @s_minmax_i1(i1 addrspace(1)* %out, i1 zeroext %cond, i1 zeroext %a, i1 zeroext %b) nounwind {
  %cmp = icmp slt i1 %cond, false
  %sel = select i1 %cmp, i1 %a, i1 %b
  store i1 %sel, i1 addrspace(1)* %out, align 4
  ret void
}
