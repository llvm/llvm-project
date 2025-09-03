; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; XFAIL: *

; ExpandFp now expands frem before it reaches dagcombine.
; TODO Implement this optimization in/before ExpandFP

; GCN-LABEL: {{^}}frem_constant_sel_constants:
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 2.0, 1.0,
define amdgpu_kernel void @frem_constant_sel_constants(ptr addrspace(1) %p, i1 %cond) {
  %sel = select i1 %cond, float -4.0, float 3.0
  %bo = frem float 5.0, %sel
  store float %bo, ptr addrspace(1) %p, align 4
  ret void
}
