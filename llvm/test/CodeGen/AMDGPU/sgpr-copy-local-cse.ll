; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -verify-machineinstrs -o - %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-ni:7"
target triple = "amdgcn-amd-amdhsa"

; CHECK-LABEL: {{^}}t0:
; CHECK: s_load_dwordx2 s{{\[}}[[PTR_LO:[0-9]+]]:[[PTR_HI:[0-9]+]]], s[4:5], 0x0
; CHECK: v_mov_b32_e32 v{{[0-9]+}}, s[[PTR_HI]]
; There should be no redundant copies from PTR_HI.
; CHECK-NOT: v_mov_b32_e32 v{{[0-9]+}}, s[[PTR_HI]]
define protected amdgpu_kernel void @t0(float addrspace(1)* %p, i32 %i0, i32 %j0, i32 %k0) {
entry:
  %0 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %i = add i32 %0, %i0
  %j = add i32 %0, %j0
  %k = add i32 %0, %k0
  %pi = getelementptr float, float addrspace(1)* %p, i32 %i
  %vi = load float, float addrspace(1)* %pi
  %pj = getelementptr float, float addrspace(1)* %p, i32 %j
  %vj = load float, float addrspace(1)* %pj
  %sum = fadd float %vi, %vj
  %pk = getelementptr float, float addrspace(1)* %p, i32 %k
  store float %sum, float addrspace(1)* %pk
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
