; RUN: opt -vector-library=Darwin_libsystem_m -passes=inject-tli-mappings,loop-vectorize -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

declare float @atan2f(float, float)

define void @foo(ptr noalias nocapture %ptrA,
                 ptr noalias nocapture readonly %ptrB,
                 ptr noalias nocapture readonly %ptrC,
                 i64 %size) {
; CHECK-LABEL: @foo(
; CHECK: call <4 x float> @_simd_atan2_f4(<4 x float>
;
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %exitcond = icmp eq i64 %indvars.iv, %size
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds float, ptr %ptrB, i64 %indvars.iv
  %src1 = load float, ptr %arrayidx, align 4

  %arrayidx2 = getelementptr inbounds float, ptr %ptrC, i64 %indvars.iv
  %src2 = load float, ptr %arrayidx, align 4

  %arrayidx3 = getelementptr inbounds float, ptr %ptrA, i64 %indvars.iv

  %phase = call float @atan2f(float %src1, float %src2)

  store float %phase, ptr %arrayidx3, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.cond
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
