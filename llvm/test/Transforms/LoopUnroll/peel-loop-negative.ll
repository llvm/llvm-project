; RUN: opt < %s -S -loop-unroll -unroll-threshold=800 -unroll-peel-max-count=0 | FileCheck %s

; We should not peel this loop even though we can, because the max count is set
; to zero.
define i32 @invariant_backedge_neg_1(i32 %a, i32 %b) {
; CHECK-LABEL: @invariant_backedge_neg_1
; CHECK-NOT:   loop.peel{{.*}}:
; CHECK:       loop:
; CHECK:         %i = phi
; CHECK:         %sum = phi
; CHECK:         %plus = phi
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %incsum, %loop ]
  %plus = phi i32 [ %a, %entry ], [ %b, %loop ]

  %incsum = add i32 %sum, %plus
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %i, 1000

  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum
}
