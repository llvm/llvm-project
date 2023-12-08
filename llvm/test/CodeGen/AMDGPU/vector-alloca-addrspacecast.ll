; RUN: opt -S -mtriple=amdgcn-- -data-layout=A5 -passes='amdgpu-promote-alloca,sroa,instcombine' < %s | FileCheck -check-prefix=OPT %s

; Should give up promoting alloca to vector with an addrspacecast.

; OPT-LABEL: @vector_addrspacecast(
; OPT: alloca [3 x i32]
; OPT: store i32 0, ptr addrspace(5) %alloca, align 4
; OPT: store i32 1, ptr addrspace(5) %a1, align 4
; OPT: store i32 2, ptr addrspace(5) %a2, align 4
; OPT: %tmp = getelementptr [3 x i32], ptr addrspace(5) %alloca, i64 0, i64 %index
; OPT: %ac = addrspacecast ptr addrspace(5) %tmp to ptr
; OPT: %data = load i32, ptr %ac, align 4
define amdgpu_kernel void @vector_addrspacecast(ptr addrspace(1) %out, i64 %index) {
entry:
  %alloca = alloca [3 x i32], addrspace(5)
  %a1 = getelementptr [3 x i32], ptr addrspace(5) %alloca, i32 0, i32 1
  %a2 = getelementptr [3 x i32], ptr addrspace(5) %alloca, i32 0, i32 2
  store i32 0, ptr addrspace(5) %alloca
  store i32 1, ptr addrspace(5) %a1
  store i32 2, ptr addrspace(5) %a2
  %tmp = getelementptr [3 x i32], ptr addrspace(5) %alloca, i64 0, i64 %index
  %ac = addrspacecast ptr addrspace(5) %tmp to ptr
  %data = load i32, ptr %ac
  store i32 %data, ptr addrspace(1) %out
  ret void
}
