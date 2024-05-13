; RUN: opt -S -mtriple=amdgcn-- -data-layout=A5 -passes='amdgpu-promote-alloca,sroa,instcombine' < %s | FileCheck -check-prefix=OPT %s

; Show that what the alloca promotion pass will do for non-atomic load/store.

; OPT-LABEL: @vector_alloca_not_atomic(
;
; OPT: extractelement <3 x i32> <i32 0, i32 1, i32 2>, i64 %index
define amdgpu_kernel void @vector_alloca_not_atomic(ptr addrspace(1) %out, i64 %index) {
entry:
  %alloca = alloca [3 x i32], addrspace(5)
  %a1 = getelementptr [3 x i32], ptr addrspace(5) %alloca, i32 0, i32 1
  %a2 = getelementptr [3 x i32], ptr addrspace(5) %alloca, i32 0, i32 2
  store i32 0, ptr addrspace(5) %alloca
  store i32 1, ptr addrspace(5) %a1
  store i32 2, ptr addrspace(5) %a2
  %tmp = getelementptr [3 x i32], ptr addrspace(5) %alloca, i64 0, i64 %index
  %data = load i32, ptr addrspace(5) %tmp
  store i32 %data, ptr addrspace(1) %out
  ret void
}

; OPT-LABEL: @vector_alloca_atomic_read(
;
; OPT: alloca [3 x i32]
; OPT: store i32 0, ptr addrspace(5)
; OPT: store i32 1, ptr addrspace(5)
; OPT: store i32 2, ptr addrspace(5)
; OPT: load atomic i32, ptr addrspace(5)
define amdgpu_kernel void @vector_alloca_atomic_read(ptr addrspace(1) %out, i64 %index) {
entry:
  %alloca = alloca [3 x i32], addrspace(5)
  %a1 = getelementptr [3 x i32], ptr addrspace(5) %alloca, i32 0, i32 1
  %a2 = getelementptr [3 x i32], ptr addrspace(5) %alloca, i32 0, i32 2
  store i32 0, ptr addrspace(5) %alloca
  store i32 1, ptr addrspace(5) %a1
  store i32 2, ptr addrspace(5) %a2
  %tmp = getelementptr [3 x i32], ptr addrspace(5) %alloca, i64 0, i64 %index
  %data = load atomic i32, ptr addrspace(5) %tmp acquire, align 4
  store i32 %data, ptr addrspace(1) %out
  ret void
}

; OPT-LABEL: @vector_alloca_atomic_write(
;
; OPT: alloca [3 x i32]
; OPT: store atomic i32 0, ptr addrspace(5)
; OPT: store atomic i32 1, ptr addrspace(5)
; OPT: store atomic i32 2, ptr addrspace(5)
; OPT: load i32, ptr addrspace(5)
define amdgpu_kernel void @vector_alloca_atomic_write(ptr addrspace(1) %out, i64 %index) {
entry:
  %alloca = alloca [3 x i32], addrspace(5)
  %a1 = getelementptr [3 x i32], ptr addrspace(5) %alloca, i32 0, i32 1
  %a2 = getelementptr [3 x i32], ptr addrspace(5) %alloca, i32 0, i32 2
  store atomic i32 0, ptr addrspace(5) %alloca release, align 4
  store atomic i32 1, ptr addrspace(5) %a1 release, align 4
  store atomic i32 2, ptr addrspace(5) %a2  release, align 4
  %tmp = getelementptr [3 x i32], ptr addrspace(5) %alloca, i64 0, i64 %index
  %data = load i32, ptr addrspace(5) %tmp
  store i32 %data, ptr addrspace(1) %out
  ret void
}
