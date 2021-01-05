; RUN: opt -debug-only=branch-prob -jump-threading -S %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Make sure that we set edge probabilities for bb2 as we
; call DuplicateCondBranchOnPHIIntoPred(bb3, {bb2}).
;
; CHECK-LABEL: ---- Branch Probability Info : foo
; CHECK:      set edge bb2 -> 0 successor probability to 0x80000000 / 0x80000000 = 100.00%
; CHECK-NEXT: set edge bb2 -> 1 successor probability to 0x00000000 / 0x80000000 = 0.00%
define void @foo(i1 %f0, i1 %f1, i1 %f2) !prof !{!"function_entry_count", i64 0} {
; CHECK-LABEL: @foo(
bb1:
  br i1 %f0, label %bb3, label %bb2

bb2:
; CHECK:      bb2:
; CHECK-NEXT:   br i1 %f2, label %exit1, label %unreach
  br label %bb3

bb3:
  %ph = phi i1 [ %f1, %bb1 ], [ %f2, %bb2 ]
  br i1 %ph, label %exit1, label %unreach

exit1:
  ret void

unreach:
  unreachable
}
