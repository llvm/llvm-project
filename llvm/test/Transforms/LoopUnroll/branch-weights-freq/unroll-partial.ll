; Test branch weight metadata, estimated trip count metadata, and block
; frequencies after partial loop unrolling without -unroll-runtime.

; ------------------------------------------------------------------------------
; RUN: opt < %s -S -passes='print<block-freq>' 2>&1 | \
; RUN:   FileCheck -check-prefix=CHECK %s
;
; Verify that the test code produces the original loop body frequency we expect.
; CHECK: - do.body: float = 10.0,

; ------------------------------------------------------------------------------
; RUN: opt < %s -S -passes='loop-unroll,print<block-freq>' \
; RUN:     -unroll-count=4 2>&1 | \
; RUN:   FileCheck %s -check-prefix=CHECK-UR \
; RUN:       -implicit-check-not='llvm.loop.estimated_trip_count' \
; RUN:       -implicit-check-not='!prof' \
; RUN:       -implicit-check-not='branch_weights' \
; RUN:       -implicit-check-not='call void @f'
;
; The sum is approximately the original loop body frequency, 10.
; CHECK-UR: - do.body: float = 2.9078,
; CHECK-UR: - do.body.1: float = 2.617,
; CHECK-UR: - do.body.2: float = 2.3553,
; CHECK-UR: - do.body.3: float = 2.1198,
;
; The original branch weights are preserved across all unrolled iterations and
; the remaining loop, and there is one original loop body (represented by a call
; to @f) within each unrolled iteration.
; CHECK-UR: call void @f
; CHECK-UR: br i1 %{{.*}}, label %do.end, label %do.body.1, !prof !0
; CHECK-UR: call void @f
; CHECK-UR: br i1 %{{.*}}, label %do.end, label %do.body.2, !prof !0
; CHECK-UR: call void @f
; CHECK-UR: br i1 %{{.*}}, label %do.end, label %do.body.3, !prof !0
; CHECK-UR: call void @f
; CHECK-UR: br i1 %{{.*}}, label %do.end, label %do.body, !prof !0, !llvm.loop !1
; CHECK-UR: !0 = !{!"branch_weights", i32 1, i32 9}
;
; llvm.loop.estimated_trip_count is the original estimated trip count (which is
; the original loop body frequency, 10, because there is no prior
; llvm.loop.estimated_trip_count) divided by the unroll count, 4, rounded up to
; the next integer.
; CHECK-UR: !1 = distinct !{!1, !2, !3}
; CHECK-UR: !2 = !{!"llvm.loop.estimated_trip_count", i32 3}
; CHECK-UR: !3 = !{!"llvm.loop.unroll.disable"}

declare void @f(i32)

define void @test(i32 %n) {

entry:
  br label %do.body

do.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %inc = add i32 %i, 1
  call void @f(i32 %i)
  %c = icmp sge i32 %inc, %n
  br i1 %c, label %do.end, label %do.body, !prof !0

do.end:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 9}
