; RUN: opt < %s -S -basic-aa -memcpyopt | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @test(ptr noalias dereferenceable(1) %in, ptr noalias dereferenceable(1) %out) {
  %tmp = alloca i8
  %tmp2 = alloca i8
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %out, ptr align 8 %in, i64 1, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %tmp, ptr align 8 %in, i64 1, i1 false), !alias.scope !4
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %tmp2, ptr align 8 %tmp, i64 1, i1 false), !alias.scope !5

  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %out, ptr align 8 %tmp2, i64 1, i1 false), !noalias !6

  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

!0 = !{!0}
!1 = distinct !{!1, !0, !"in"}
!2 = distinct !{!2, !0, !"tmp"}
!3 = distinct !{!3, !0, !"tmp2"}
!4 = distinct !{!1, !2}
!5 = distinct !{!2, !3}
!6 = distinct !{!1, !2}
