; RUN: opt < %s -passes=loop-vectorize -enable-vplan-native-path -vplan-build-stress-test -vplan-verify-hcfg -debug-only=loop-vectorize -disable-output 2>&1 | FileCheck %s -check-prefix=VERIFIER
; RUN: opt < %s -passes=loop-vectorize -enable-vplan-native-path -vplan-build-stress-test -debug-only=loop-vectorize -disable-output 2>&1 | FileCheck %s -check-prefix=NO-VERIFIER -allow-empty
; REQUIRES: asserts

; Verify that the stress testing flag for the VPlan H-CFG builder works as
; expected with and without enabling the VPlan H-CFG Verifier.

; VERIFIER: Verifying VPlan H-CFG.
; NO-VERIFIER-NOT: Verifying VPlan H-CFG.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(ptr nocapture %a, ptr nocapture readonly %b, i32 %N, i32 %M) {
entry:
  %cmp32 = icmp sgt i32 %N, 0
  br i1 %cmp32, label %outer.ph, label %for.end15

outer.ph:
  %cmp230 = icmp sgt i32 %M, 0
  %0 = sext i32 %M to i64
  %wide.trip.count = zext i32 %M to i64
  %wide.trip.count38 = zext i32 %N to i64
  br label %outer.body

outer.body:
  %indvars.iv35 = phi i64 [ 0, %outer.ph ], [ %indvars.iv.next36, %outer.inc ]
  br i1 %cmp230, label %inner.ph, label %outer.inc

inner.ph:
  %1 = mul nsw i64 %indvars.iv35, %0
  br label %inner.body

inner.body:
  %indvars.iv = phi i64 [ 0, %inner.ph ], [ %indvars.iv.next, %inner.body ]
  %2 = add nsw i64 %indvars.iv, %1
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %2
  %3 = load i32, ptr %arrayidx, align 4
  %arrayidx12 = getelementptr inbounds i32, ptr %a, i64 %2
  store i32 %3, ptr %arrayidx12, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %outer.inc, label %inner.body

outer.inc:
  %indvars.iv.next36 = add nuw nsw i64 %indvars.iv35, 1
  %exitcond39 = icmp eq i64 %indvars.iv.next36, %wide.trip.count38
  br i1 %exitcond39, label %for.end15, label %outer.body

for.end15:
  ret void
}
