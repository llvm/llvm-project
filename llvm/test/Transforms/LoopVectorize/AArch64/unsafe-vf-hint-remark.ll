; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple=arm64-apple-iphoneos -pass-remarks-analysis=loop-vectorize -debug-only=loop-vectorize -S < %s 2>&1 | FileCheck %s

; Specify a large unsafe vectorization factor of 32 that gets clamped to 16,
; then test an even smaller VF of 2 is selected based on the cost-model.

; CHECK: LV: User VF=32 is unsafe, clamping to max safe VF=16.
; CHECK: remark: <unknown>:0:0: User-specified vectorization factor 32 is unsafe, clamping to maximum safe vectorization factor 16
; CHECK: LV: Selecting VF: 2.
; CHECK-LABEL: @test
; CHECK: <2 x i64>
define void @test(i64* nocapture %a, i64* nocapture readonly %b) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 %iv
  %0 = load i64, i64* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i64, i64* %b, i64 %iv
  %1 = load i64, i64* %arrayidx2, align 4
  %add = add nsw i64 %1, %0
  %2 = add nuw nsw i64 %iv, 16
  %arrayidx5 = getelementptr inbounds i64, i64* %a, i64 %2
  %c = icmp eq i64 %1, 120
  br i1 %c, label %then, label %latch

then:
  store i64 %add, i64* %arrayidx5, align 4
  br label %latch

latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop.header, !llvm.loop !0

exit:
  ret void
}

!0 = !{!0, !1, !2}
!1 = !{!"llvm.loop.vectorize.width", i64 32}
!2 = !{!"llvm.loop.vectorize.enable", i1 true}
