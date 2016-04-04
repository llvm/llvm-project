; RUN: opt < %s -S -unroll-threshold=20 -loop-unroll -unroll-allow-partial | FileCheck %s

; The Loop TripCount is 9. However unroll factors 3 or 9 exceed given threshold.
; The test checks that we choose a smaller, power-of-two, unroll count and do not give up on unrolling.

; CHECK: for.body:
; CHECK: store
; CHECK: for.body.1:
; CHECK: store

define void @foo(i32* nocapture %a, i32* nocapture readonly %b) nounwind uwtable {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 1, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %ld = load i32, i32* %arrayidx, align 4
  %idxprom1 = sext i32 %ld to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %idxprom1
  %st = trunc i64 %indvars.iv to i32
  store i32 %st, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
