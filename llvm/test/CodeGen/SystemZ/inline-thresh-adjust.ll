; RUN: opt < %s -mtriple=systemz-unknown -mcpu=z15 -inline -disable-output \
; RUN:   -debug-only=inline,systemztti 2>&1 | FileCheck %s
; REQUIRES: asserts
;
; Check that the inlining threshold is incremented for a function using an
; argument only as a memcpy source.

; CHECK: Inlining calls in: root_function
; CHECK:     Inlining {{.*}} Call:   call void @leaf_function_A(ptr %Dst)
; CHECK:     ++ SZTTI Adding inlining bonus: 150
; CHECK:     Inlining {{.*}} Call:   call void @leaf_function_B(ptr %Dst, ptr %Src)

define void @leaf_function_A(ptr %Dst)  {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %Dst, ptr undef, i64 16, i1 false)
  ret void
}

define void @leaf_function_B(ptr %Dst, ptr %Src)  {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %Dst, ptr %Src, i64 16, i1 false)
  ret void
}

define void @root_function(ptr %Dst, ptr %Src) {
entry:
  call void @leaf_function_A(ptr %Dst)
  call void @leaf_function_B(ptr %Dst, ptr %Src)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
