; RUN: opt -indvars -scalar-evolution -analyze %s
; This test checks if the SCEV analysis is printed out at all.
; It failed once as the RequiredTransitive option was not implemented
; correctly.

define i32 @main() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvar1 = phi i64 [ %indvar.next2, %for.inc ], [ 0, %entry ] ; <i64> [#uses=3]
  %exitcond = icmp ne i64 %indvar1, 1024          ; <i1> [#uses=1]
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvar.next2 = add i64 %indvar1, 1             ; <i64> [#uses=1]
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret i32 0
}
