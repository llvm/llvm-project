; RUN: opt -passes=loop-vectorize -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -S -debug %s 2>&1 | FileCheck %s

; REQUIRES: asserts

; Make sure we do not vectorize a loop with a widened int induction.
define void @test_wide_integer_induction(ptr noalias %a, i64 %N) {
; CHECK-NOT: LV: Vector loop of width {{.+}} costs:
;
; CHECK: define void @test_wide_integer_induction(
; CHECK-NOT: vector.body
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  store i64 %iv, ptr %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}
