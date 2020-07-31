; RUN: opt -analyze -enable-new-pm=0 -scalar-evolution < %s | FileCheck %s
; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

; ScalarEvolution should be able to use nsw information to prove that
; this loop has a finite trip count.

; CHECK: @le
; CHECK: Loop %for.body: backedge-taken count is %n
; CHECK: Loop %for.body: max backedge-taken count is 9223372036854775807

define void @le(i64 %n, double* nocapture %p) nounwind {
entry:
  %cmp6 = icmp slt i64 %n, 0                      ; <i1> [#uses=1]
  br i1 %cmp6, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ] ; <i64> [#uses=2]
  %arrayidx = getelementptr double, double* %p, i64 %i    ; <double*> [#uses=2]
  %t4 = load double, double* %arrayidx                    ; <double> [#uses=1]
  %mul = fmul double %t4, 2.200000e+00            ; <double> [#uses=1]
  store double %mul, double* %arrayidx
  %i.next = add nsw i64 %i, 1                     ; <i64> [#uses=2]
  %cmp = icmp sgt i64 %i.next, %n                 ; <i1> [#uses=1]
  br i1 %cmp, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}
