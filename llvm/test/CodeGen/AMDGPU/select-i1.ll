; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; FIXME: This should go in existing select.ll test, except the current testcase there is broken on GCN

; GCN-LABEL: {{^}}select_i1:
; GCN: s_cselect_b32
; GCN-NOT: v_cndmask_b32
define amdgpu_kernel void @select_i1(ptr addrspace(1) %out, i32 %cond, i1 %a, i1 %b) nounwind {
  %cmp = icmp ugt i32 %cond, 5
  %sel = select i1 %cmp, i1 %a, i1 %b
  store i1 %sel, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}s_minmax_i1:
; GCN: s_load_dword [[LOAD:s[0-9]+]],
; GCN: s_bitcmp1_b32 [[LOAD]], 0
; GCN: s_cselect_b32 [[SHIFTVAL:s[0-9]+]], 8, 16
; GCN: s_lshr_b32 [[LOAD]], [[LOAD]], [[SHIFTVAL]]
; GCN: s_and_b32  [[LOAD]], [[LOAD]], 1
define amdgpu_kernel void @s_minmax_i1(ptr addrspace(1) %out, [8 x i32], i1 zeroext %cond, i1 zeroext %a, i1 zeroext %b) nounwind {
  %cmp = icmp slt i1 %cond, false
  %sel = select i1 %cmp, i1 %a, i1 %b
  store i1 %sel, ptr addrspace(1) %out, align 4
  ret void
}
