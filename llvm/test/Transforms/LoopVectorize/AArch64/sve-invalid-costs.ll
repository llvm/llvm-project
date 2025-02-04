; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize -mattr=+sve -force-vector-width=4 -debug-only=loop-vectorize \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue -S -disable-output 2>&1 | FileCheck %s
target triple = "aarch64-linux-gnu"

define dso_local void @loop_sve_i1(ptr nocapture %ptr, i64 %N) {
; CHECK-LABEL: LV: Checking a loop in 'loop_sve_i1'
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 4 For instruction:   %0 = load i1, ptr %arrayidx, align 16
; CHECK: LV: Found an estimated cost of Invalid for VF vscale x 4 For instruction:   store i1 %add, ptr %arrayidx, align 16
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i1, ptr %ptr, i64 %iv
  %0 = load i1, ptr %arrayidx, align 16
  %add = add nsw i1 %0, 1
  store i1 %add, ptr %arrayidx, align 16
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
