; RUN: opt < %s -loop-unroll -unroll-count=2 -S | FileCheck %s

; LoopUnroll should unroll this loop into one big basic block.

; CHECK: for.body:
; CHECK: %i.013 = phi i64 [ 0, %entry ], [ %tmp16.1, %for.body ]
; CHECK: br i1 %exitcond.1, label %for.end, label %for.body

define void @foo(double* nocapture %p, i64 %n) nounwind {
entry:
  %mul10 = shl i64 %n, 1                          ; <i64> [#uses=2]
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.013 = phi i64 [ %tmp16, %for.body ], [ 0, %entry ] ; <i64> [#uses=2]
  %arrayidx7 = getelementptr double* %p, i64 %i.013 ; <double*> [#uses=2]
  %tmp16 = add i64 %i.013, 1                      ; <i64> [#uses=3]
  %arrayidx = getelementptr double* %p, i64 %tmp16 ; <double*> [#uses=1]
  %tmp4 = load double* %arrayidx                  ; <double> [#uses=1]
  %tmp8 = load double* %arrayidx7                 ; <double> [#uses=1]
  %mul9 = fmul double %tmp8, %tmp4                ; <double> [#uses=1]
  store double %mul9, double* %arrayidx7
  %exitcond = icmp eq i64 %tmp16, %mul10          ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}
