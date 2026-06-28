; RUN: opt < %s -passes=loop-vectorize -disable-output

; Regression test for issue #182646:
; https://github.com/llvm/llvm-project/issues/182646
;
; This reduced test case has a symbolic induction step and loop vectorization
; hints. It previously hit the "VPlan cost model and legacy cost model
; disagreed" assertion in computeBestVF().

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external global [0 x [3 x [3 x i8]]], align 1
@b = external global [0 x [3 x i8]], align 1

define void @test(i32 %d, i1 %e, ptr %f) {
entry:
  %assume.cmp = icmp eq i32 %d, -1430399074
  call void @llvm.assume(i1 %assume.cmp)
  %step = add nsw i32 %d, 1430399077
  br label %for.body

for.body:
  %i = phi i32 [ 1, %entry ], [ %next, %for.body ]
  %idx = sext i32 %i to i64
  %a.gep = getelementptr inbounds [3 x i8], ptr @a, i64 0, i64 %idx
  store i8 0, ptr %a.gep, align 1
  %f.gep = getelementptr inbounds [3 x [3 x i8]], ptr %f, i64 %idx, i64 %idx
  %f.val = load i8, ptr %f.gep, align 1
  %f.bool = trunc i8 %f.val to i1
  %stored.bool = or i1 %f.bool, %e
  %stored.val = zext i1 %stored.bool to i8
  store i8 %stored.val, ptr @b, align 1
  %sum = add i32 %step, %i
  %cmp = icmp slt i32 %sum, 10
  %next = select i1 %cmp, i32 %sum, i32 1
  br label %for.body, !llvm.loop !0
}

declare void @llvm.assume(i1)

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}
