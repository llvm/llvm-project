; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -amdgpu-promote-alloca < %s | FileCheck %s

; This is just an arbitrary intrinisic that shouldn't ever need to be
; handled to ensure it doesn't crash.

declare void @llvm.stackrestore(i8*) #2

; CHECK-LABEL: @try_promote_unhandled_intrinsic(
; CHECK: alloca
; CHECK: call void @llvm.stackrestore(i8* %tmp1)
define amdgpu_kernel void @try_promote_unhandled_intrinsic(i32 addrspace(1)* %arg) #2 {
bb:
  %tmp = alloca i32, align 4
  %tmp1 = bitcast i32* %tmp to i8*
  %tmp2 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 1
  %tmp3 = load i32, i32 addrspace(1)* %tmp2
  store i32 %tmp3, i32* %tmp
  call void @llvm.stackrestore(i8* %tmp1)
  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
