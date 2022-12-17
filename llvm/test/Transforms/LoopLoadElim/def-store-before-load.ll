; RUN: opt -passes=loop-load-elim -S < %s | FileCheck %s

;  No loop-carried forwarding: The intervening store to A[i] kills the stored
;  value from the previous iteration.
;
;   for (unsigned i = 0; i < 100; i++) {
;     A[i] = 1;
;     A[i+1] = A[i] + B[i];
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i64 %N) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
; CHECK-NOT: %store_forwarded
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 1, i32* %arrayidx, align 4
  %a = load i32, i32* %arrayidx, align 4
  %arrayidxB = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %b = load i32, i32* %arrayidxB, align 4
; CHECK: %add = add i32 %b, %a
  %add = add i32 %b, %a
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx_next = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  store i32 %add, i32* %arrayidx_next, align 4
  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
