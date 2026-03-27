; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -o - %s | FileCheck %s

; CHECK-LABEL: {{^}}t0:
; CHECK: s_load_dwordx2 s[[[PTR_LO:[0-9]+]]:[[PTR_HI:[0-9]+]]], s[8:9], 0x0
; CHECK: v_mov_b32_e32 v{{[0-9]+}}, s[[PTR_HI]]
; There should be no redundant copies from PTR_HI.
; CHECK-NOT: v_mov_b32_e32 v{{[0-9]+}}, s[[PTR_HI]]
define protected amdgpu_kernel void @t0(ptr addrspace(1) %p, i32 %i0, i32 %j0, i32 %k0) {
entry:
  %0 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %i = add i32 %0, %i0
  %j = add i32 %0, %j0
  %k = add i32 %0, %k0
  %pi = getelementptr float, ptr addrspace(1) %p, i32 %i
  %vi = load float, ptr addrspace(1) %pi
  %pj = getelementptr float, ptr addrspace(1) %p, i32 %j
  %vj = load float, ptr addrspace(1) %pj
  %sum = fadd float %vi, %vj
  %pk = getelementptr float, ptr addrspace(1) %p, i32 %k
  store float %sum, ptr addrspace(1) %pk
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
