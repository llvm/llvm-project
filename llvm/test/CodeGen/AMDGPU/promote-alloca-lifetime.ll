; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -passes=amdgpu-promote-alloca %s | FileCheck -check-prefix=OPT %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

declare void @llvm.lifetime.start.p5(i64, ptr addrspace(5) nocapture) #0
declare void @llvm.lifetime.end.p5(i64, ptr addrspace(5) nocapture) #0

; OPT-LABEL: @use_lifetime_promotable_lds(
; OPT-NOT: alloca i32
; OPT-NOT: llvm.lifetime
; OPT: store i32 %tmp3, ptr addrspace(3)
define amdgpu_kernel void @use_lifetime_promotable_lds(ptr addrspace(1) %arg) #2 {
bb:
  %tmp = alloca i32, align 4, addrspace(5)
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %tmp)
  %tmp2 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 1
  %tmp3 = load i32, ptr addrspace(1) %tmp2
  store i32 %tmp3, ptr addrspace(5) %tmp
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %tmp)
  ret void
}

; After handling the alloca, the lifetime was erased. This was the
; next iterator to be checked as an alloca, crashing.

; OPT-LABEL: @iterator_erased_lifetime(
; OPT-NOT: alloca i8
define amdgpu_kernel void @iterator_erased_lifetime() {
entry:
  %alloca = alloca i8, align 1, addrspace(5)
  call void @llvm.lifetime.start.p5(i64 1, ptr addrspace(5) %alloca)
  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
