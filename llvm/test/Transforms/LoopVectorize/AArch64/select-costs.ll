; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize -debug-only=loop-vectorize -S 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

define void @selects_1(i32* nocapture %dst, i32 %A, i32 %B, i32 %C, i32 %N) {
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %cond = select i1 %cmp1, i32 10, i32 %and
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %cond6 = select i1 %cmp2, i32 30, i32 %and
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %cond11 = select i1 %cmp7, i32 %cond, i32 %cond6

; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %cond = select i1 %cmp1, i32 10, i32 %and
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %cond6 = select i1 %cmp2, i32 30, i32 %and
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %cond11 = select i1 %cmp7, i32 %cond, i32 %cond6

; CHECK-LABEL: define void @selects_1(
; CHECK:       vector.body:
; CHECK:         select <4 x i1>

entry:
  %cmp26 = icmp sgt i32 %N, 0
  br i1 %cmp26, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %dst, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %and = and i32 %0, 2047
  %cmp1 = icmp eq i32 %and, %A
  %cond = select i1 %cmp1, i32 10, i32 %and
  %cmp2 = icmp eq i32 %and, %B
  %cond6 = select i1 %cmp2, i32 30, i32 %and
  %cmp7 = icmp ugt i32 %cond, %C
  %cond11 = select i1 %cmp7, i32 %cond, i32 %cond6
  store i32 %cond11, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}
