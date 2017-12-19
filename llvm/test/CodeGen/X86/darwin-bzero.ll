; RUN: llc < %s -mtriple=i386-apple-darwin10 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 | FileCheck %s

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

; CHECK-LABEL: foo:
; CHECK: {{calll|callq}} ___bzero
define void @foo(i8* %p, i32 %len) {
  call void @llvm.memset.p0i8.i32(i8* %p, i8 0, i32 %len, i32 1, i1 false)
  ret void
}
