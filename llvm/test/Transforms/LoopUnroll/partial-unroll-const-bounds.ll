; RUN: opt < %s -S -unroll-partial-threshold=20 -unroll-threshold=20 -passes=loop-unroll -unroll-allow-partial -unroll-runtime -unroll-allow-remainder -unroll-max-percent-threshold-boost=100 | FileCheck %s
; RUN: opt < %s -S -passes='require<opt-remark-emit>,loop-unroll' -unroll-partial-threshold=20 -unroll-threshold=20 -unroll-allow-partial -unroll-runtime -unroll-allow-remainder -unroll-max-percent-threshold-boost=100 | FileCheck %s
;
; Also check that the simple unroller doesn't allow the partial unrolling.
; RUN: opt < %s -S -passes='require<opt-remark-emit>,loop(loop-unroll-full)' -unroll-partial-threshold=20 -unroll-threshold=20 -unroll-allow-partial -unroll-runtime -unroll-allow-remainder -unroll-max-percent-threshold-boost=100 | FileCheck %s --check-prefix=CHECK-NO-UNROLL

; The Loop TripCount is 9. However unroll factors 3 or 9 exceed given threshold.
; The test checks that we choose a smaller, power-of-two, unroll count and do not give up on unrolling.

; CHECK: for.body:
; CHECK: store
; CHECK: for.body.1:
; CHECK: store

; CHECK-NO-UNROLL: for.body:
; CHECK-NO-UNROLL: store
; CHECK-NO-UNROLL-NOT: store

define void @foo(ptr nocapture %a, ptr nocapture readonly %b) nounwind uwtable {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 1, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %ld = load i32, ptr %arrayidx, align 4
  %idxprom1 = sext i32 %ld to i64
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 %idxprom1
  %st = trunc i64 %indvars.iv to i32
  store i32 %st, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
