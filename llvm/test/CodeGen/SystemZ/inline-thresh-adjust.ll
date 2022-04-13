; RUN: opt < %s -mtriple=systemz-unknown -mcpu=z15 -inline -disable-output \
; RUN:   -debug-only=inline,systemztti 2>&1 | FileCheck %s
; REQUIRES: asserts
;
; Check that the inlining threshold is incremented for a function using an
; argument only as a memcpy source.

; CHECK: Inlining calls in: root_function
; CHECK:     Inlining {{.*}} Call:   call void @leaf_function_A(i8* %Dst)
; CHECK:     ++ SZTTI Adding inlining bonus: 150
; CHECK:     Inlining {{.*}} Call:   call void @leaf_function_B(i8* %Dst, i8* %Src)

define void @leaf_function_A(i8* %Dst)  {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %Dst, i8* undef, i64 16, i1 false)
  ret void
}

define void @leaf_function_B(i8* %Dst, i8* %Src)  {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %Dst, i8* %Src, i64 16, i1 false)
  ret void
}

define void @root_function(i8* %Dst, i8* %Src) {
entry:
  call void @leaf_function_A(i8* %Dst)
  call void @leaf_function_B(i8* %Dst, i8* %Src)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)
