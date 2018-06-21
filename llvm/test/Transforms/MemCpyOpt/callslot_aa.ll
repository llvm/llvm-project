; RUN: opt < %s -S -basicaa -memcpyopt | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%T = type { i64, i64 }

define void @test(i8* %src) {
  %tmp = alloca i8
  %dst = alloca i8
; CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 1, i32 8, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp, i8* %src, i64 1, i32 8, i1 false), !noalias !2
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %tmp, i64 1, i32 8, i1 false)

  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i32, i1)

; Check that the noalias for "dst" was removed by checking that the metadata is gone
; CHECK-NOT: "dst"
!0 = !{!0}
!1 = distinct !{!1, !0, !"dst"}
!2 = distinct !{!1}
