; REQUIRES: asserts
; RUN: opt -loop-vectorize -pass-remarks-analysis=loop-vectorize -debug-only=loop-vectorize -S -scalable-vectorization=on < %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; CHECK: LV: Disabling scalable vectorization, because target does not support scalable vectors.
; CHECK: remark: <unknown>:0:0: Disabling scalable vectorization, because target does not support scalable vectors.
; CHECK: LV: The Widest register safe to use is: 32 bits.
define void @test1(i32* %a, i32* %b) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %2 = add nuw nsw i64 %iv, 4
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

!0 = !{!0, !1, !2}
!1 = !{!"llvm.loop.vectorize.width", i32 4}
!2 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
