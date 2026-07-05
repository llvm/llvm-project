; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -passes=amdgpu-promote-alloca < %s | FileCheck %s

; This is just an arbitrary intrinisic that shouldn't be
; handled to ensure it doesn't crash.

declare void @llvm.stackrestore.p5(ptr addrspace(5)) #2

; CHECK-LABEL: @try_promote_unhandled_intrinsic(
; CHECK: alloca
; CHECK: call void @llvm.stackrestore.p5(ptr addrspace(5) %tmp)
define amdgpu_kernel void @try_promote_unhandled_intrinsic(ptr addrspace(1) %arg) #2 {
bb:
  %tmp = alloca i32, addrspace(5)
  %tmp2 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 1
  %tmp3 = load i32, ptr addrspace(1) %tmp2
  store i32 %tmp3, ptr addrspace(5) %tmp
  call void @llvm.stackrestore.p5(ptr addrspace(5) %tmp)
  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
