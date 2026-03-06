; RUN: opt < %s -passes=indvars,loop-deletion -S |FileCheck %s

; Check IndVarSimplify should replace exit value even if the expansion cost
; is high because the loop can be deleted after the exit value rewrite.
;
; CHECK-LABEL: @_Z3fooPKcjj(
; CHECK: udiv
; CHECK: [[LABEL:^[a-zA-Z0-9_.]+]]:
; CHECK-NOT: br {{.*}} [[LABEL]]

define i32 @_Z3fooPKcjj(ptr nocapture readnone %s, i32 %len, i32 %c) #0 {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %klen.0 = phi i32 [ %len, %entry ], [ %sub, %while.cond ]
  %cmp = icmp ugt i32 %klen.0, 11
  %sub = add i32 %klen.0, -12
  br i1 %cmp, label %while.cond, label %while.end

while.end:                                        ; preds = %while.cond
  %klen.0.lcssa = phi i32 [ %klen.0, %while.cond ]
  ret i32 %klen.0.lcssa
}
