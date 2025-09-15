; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -pass-remarks-analysis=loop-vectorize -debug-only=loop-vectorize --disable-output -S -scalable-vectorization=on < %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; CHECK: LV: User VF=vscale x 4 is ignored because scalable vectors are not available.
; CHECK: remark: <unknown>:0:0: User-specified vectorization factor vscale x 4 is ignored because the target does not support scalable vectors. The compiler will pick a more suitable value.
; CHECK: LV: The Widest register safe to use is: 32 bits.
define void @test1(ptr %a, ptr %b) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %2 = add nuw nsw i64 %iv, 4
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %2
  store i32 %add, ptr %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

!0 = !{!0, !1, !2}
!1 = !{!"llvm.loop.vectorize.width", i32 4}
!2 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
