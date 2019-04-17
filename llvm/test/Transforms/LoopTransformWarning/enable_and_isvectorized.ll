; RUN: opt -transform-warning -disable-output < %s 2>&1 | FileCheck -allow-empty %s
;
; llvm.org/PR40546
; Do not warn about about leftover llvm.loop.vectorize.enable for already
; vectorized loops.

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"

define void @test(i32 %n) {
entry:
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %simd.if.end, label %omp.inner.for.body.preheader

omp.inner.for.body.preheader:
  %wide.trip.count = zext i32 %n to i64
  br label %omp.inner.for.body

omp.inner.for.body:
  %indvars.iv = phi i64 [ 0, %omp.inner.for.body.preheader ], [ %indvars.iv.next, %omp.inner.for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %simd.if.end, label %omp.inner.for.body, !llvm.loop !0

simd.if.end:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
!2 = !{!"llvm.loop.isvectorized"}


; CHECK-NOT: loop not vectorized
