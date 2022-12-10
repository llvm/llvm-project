; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -passes=amdgpu-promote-alloca < %s | FileCheck %s

; This is just an arbitrary intrinisic that shouldn't ever need to be
; handled to ensure it doesn't crash.

declare void @llvm.stackrestore(ptr) #2

; CHECK-LABEL: @try_promote_unhandled_intrinsic(
; CHECK: alloca
; CHECK: call void @llvm.stackrestore(ptr %tmp)
define amdgpu_kernel void @try_promote_unhandled_intrinsic(ptr addrspace(1) %arg) #2 {
bb:
  %tmp = alloca i32, align 4
  %tmp2 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 1
  %tmp3 = load i32, ptr addrspace(1) %tmp2
  store i32 %tmp3, ptr %tmp
  call void @llvm.stackrestore(ptr %tmp)
  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
