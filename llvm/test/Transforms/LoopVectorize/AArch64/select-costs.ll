; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize -debug-only=loop-vectorize -disable-output -S 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

define void @selects_1(ptr nocapture %dst, i32 %A, i32 %B, i32 %C, i32 %N) {
; CHECK: LV: Checking a loop in 'selects_1'

; CHECK: Cost of 1 for VF 2: WIDEN-SELECT ir<%cond> = select ir<%cmp1>, ir<10>, ir<%and>
; CHECK: Cost of 1 for VF 2: WIDEN-SELECT ir<%cond6> = select ir<%cmp2>, ir<30>, ir<%and>
; CHECK: Cost of 1 for VF 2: WIDEN-SELECT ir<%cond11> = select ir<%cmp7>, ir<%cond>, ir<%cond6>

; CHECK: Cost of 1 for VF 4: WIDEN-SELECT ir<%cond> = select ir<%cmp1>, ir<10>, ir<%and>
; CHECK: Cost of 1 for VF 4: WIDEN-SELECT ir<%cond6> = select ir<%cmp2>, ir<30>, ir<%and>
; CHECK: Cost of 1 for VF 4: WIDEN-SELECT ir<%cond11> = select ir<%cmp7>, ir<%cond>, ir<%cond6>

; CHECK: LV: Selecting VF: 4

entry:
  %cmp26 = icmp sgt i32 %N, 0
  br i1 %cmp26, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %n = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %dst, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %and = and i32 %0, 2047
  %cmp1 = icmp eq i32 %and, %A
  %cond = select i1 %cmp1, i32 10, i32 %and
  %cmp2 = icmp eq i32 %and, %B
  %cond6 = select i1 %cmp2, i32 30, i32 %and
  %cmp7 = icmp ugt i32 %cond, %C
  %cond11 = select i1 %cmp7, i32 %cond, i32 %cond6
  store i32 %cond11, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}

define i32 @multi_user_cmp(ptr readonly %a, i64 noundef %n) {
; CHECK: LV: Checking a loop in 'multi_user_cmp'
; CHECK: Cost of 4 for VF 16: WIDEN ir<%cmp1> = fcmp olt ir<%load1>, ir<0.000000e+00>
; CHECK: LV: Selecting VF: 16.
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %all.off.next = phi i1 [ true, %entry ], [ %all.off, %for.body ]
  %any.0.off09 = phi i1 [ false, %entry ], [ %.any.0.off0, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %load1 = load float, ptr %arrayidx, align 4
  %cmp1 = fcmp olt float %load1, 0.000000e+00
  %.any.0.off0 = select i1 %cmp1, i1 true, i1 %any.0.off09
  %all.off = select i1 %cmp1, i1 %all.off.next, i1 false
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  %0 = select i1 %.any.0.off0, i32 2, i32 3
  %1 = select i1 %all.off, i32 1, i32 %0
  ret i32 %1
}
