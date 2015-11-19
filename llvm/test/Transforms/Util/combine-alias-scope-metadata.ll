; RUN: opt < %s -S -basicaa -memcpyopt | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @test(i8* noalias dereferenceable(1) %in, i8* noalias dereferenceable(1) %out) {
  %tmp = alloca i8
  %tmp2 = alloca i8
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %out, i8* %in, i64 1, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp, i8* %in, i64 1, i1 false), !alias.scope !4
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp2, i8* %tmp, i64 1, i1 false), !alias.scope !5

  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %out, i8* %tmp2, i64 1, i1 false), !noalias !6

  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)

!0 = !{!0}
!1 = distinct !{!1, !0, !"in"}
!2 = distinct !{!2, !0, !"tmp"}
!3 = distinct !{!3, !0, !"tmp2"}
!4 = distinct !{!1, !2}
!5 = distinct !{!2, !3}
!6 = distinct !{!1, !2}
