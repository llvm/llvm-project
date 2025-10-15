; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v -S -debug %s 2>&1 | FileCheck %s

; REQUIRES: asserts

; For %for.1, we are fine initially, because the previous value %for.1.next dominates the
; user of %for.1. But for %for.2, we have to sink the user (%for.1.next) past the previous
; value %for.2.next. This however breaks the condition we have for %for.1. We cannot fix
; both first order recurrences and cannot vectorize the loop.
;
; Make sure we don't compute costs if there are no vector VPlans.

; CHECK-NOT: LV: Vector loop of width {{.+}} costs:
;
; CHECK: define i32 @test(
; CHECK-NOT: vector.body
;
define i32 @test(i32 %N) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %iv  = phi i32 [ %inc, %for.body ], [ 10, %entry ]
  %for.1 = phi i32 [ %for.1.next, %for.body ], [ 20, %entry ]
  %for.2 = phi i32 [ %for.2.next, %for.body ], [ 11, %entry ]
  %for.1.next = add nsw i32 %for.2, 1
  %for.2.next = shl i32 %for.1, 24
  %inc = add nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond1.for.end_crit_edge, label %for.body

for.cond1.for.end_crit_edge:                      ; preds = %for.body
  %add.lcssa = phi i32 [ %for.1.next, %for.body ]
  %sext.lcssa = phi i32 [ %for.2.next, %for.body ]
  %res = add i32 %add.lcssa, %sext.lcssa
  ret i32 %res
}
