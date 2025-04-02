; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize -mtriple riscv64 -riscv-v-vector-bits-min=128 -mattr="+v" -debug-only=loop-vectorize --disable-output -S 2>&1 | FileCheck %s

; CHECK: LV: Loop hints: force=enabled
; CHECK: LV: Scalar loop costs: 4.
; ChosenFactor.Cost is 4, but the real cost will be divided by the width, which is 2.
; CHECK: Cost for VF 2: 4 (Estimated cost per lane: 2.0)
; Regardless of force vectorization or not, this loop will eventually be vectorized because of the cost model.
; Therefore, the following message does not need to be printed even if vectorization is explicitly forced in the metadata.
; CHECK-NOT: LV: Vectorization seems to be not beneficial, but was forced by a user.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown"

define i64 @foo(ptr nocapture noundef readonly %a, i64 noundef %N, i64 noundef %init) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.06 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rd.05 = phi i64 [ %add, %for.body ], [ %init, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %i.06
  %0 = load i64, ptr %arrayidx, align 8
  %add = add nsw i64 %0, %rd.05
  %inc = add nuw i64 %i.06, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body
  ret i64 %add
}

!0 = !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
