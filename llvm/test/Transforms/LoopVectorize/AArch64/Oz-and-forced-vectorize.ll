; RUN: opt -passes='default<Oz>' -S < %s | FileCheck %s

; Forcing vectorization should allow for more aggressive loop-rotation with
; -Oz, because LV requires rotated loops. Make sure the loop in @foo is
; vectorized with -Oz.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

define void @foo(ptr noalias nocapture %ptrA, ptr noalias nocapture readonly %ptrB, i64 %size) {
; CHECK-LABEL: @foo(
; CHECK: fmul <4 x float>
;
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %exitcond = icmp eq i64 %indvars.iv, %size
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds float, ptr %ptrB, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %ptrA, i64 %indvars.iv
  %1 = load float, ptr %arrayidx2, align 4
  %mul3 = fmul float %0, %1
  store float %mul3, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.cond
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
