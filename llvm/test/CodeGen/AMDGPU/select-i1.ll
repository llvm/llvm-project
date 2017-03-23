; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FIXME: This should go in existing select.ll test, except the current testcase there is broken on SI

; FUNC-LABEL: {{^}}select_i1:
; SI: v_cndmask_b32
; SI-NOT: v_cndmask_b32
define amdgpu_kernel void @select_i1(i1 addrspace(1)* %out, i32 %cond, i1 %a, i1 %b) nounwind {
  %cmp = icmp ugt i32 %cond, 5
  %sel = select i1 %cmp, i1 %a, i1 %b
  store i1 %sel, i1 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_minmax_i1:
; SI-DAG: buffer_load_ubyte [[COND:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:44
; SI-DAG: buffer_load_ubyte [[A:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:45
; SI-DAG: buffer_load_ubyte [[B:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:46
; SI: v_cmp_eq_u32_e32 vcc, 1, [[COND]]
; SI: v_cndmask_b32_e32 v{{[0-9]+}}, [[B]], [[A]]
define amdgpu_kernel void @s_minmax_i1(i1 addrspace(1)* %out, i1 zeroext %cond, i1 zeroext %a, i1 zeroext %b) nounwind {
  %cmp = icmp slt i1 %cond, false
  %sel = select i1 %cmp, i1 %a, i1 %b
  store i1 %sel, i1 addrspace(1)* %out, align 4
  ret void
}
