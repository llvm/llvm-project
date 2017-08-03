; RUN: llc < %s -march=x86-64 | grep cvtsi2sd

; LSR previously eliminated the sitofp by introducing an induction
; variable which stepped by a bogus ((double)UINT32_C(-1)). It's theoretically
; possible to eliminate the sitofp using a proper -1.0 step though; this
; test should be changed if that is done.

define void @foo(i32 %N) nounwind {
entry:
  %0 = icmp slt i32 %N, 0                         ; <i1> [#uses=1]
  br i1 %0, label %bb, label %return

bb:                                               ; preds = %bb, %entry
  %i.03 = phi i32 [ 0, %entry ], [ %2, %bb ]      ; <i32> [#uses=2]
  %1 = sitofp i32 %i.03 to double                  ; <double> [#uses=1]
  tail call void @bar(double %1) nounwind
  %2 = add nsw i32 %i.03, -1                       ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %2, %N                  ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}

declare void @bar(double)
