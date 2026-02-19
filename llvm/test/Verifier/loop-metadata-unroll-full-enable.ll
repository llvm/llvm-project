; RUN: opt < %s -passes=verify -S -o /dev/null

; Test that the verifier accepts loop metadata containing both
; llvm.loop.unroll.full and llvm.loop.unroll.enable simultaneously.

define void @loop_full_and_enable() {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i32 %i, 1
  %cmp = icmp ult i32 %inc, 4
  br i1 %cmp, label %loop, label %exit, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.unroll.full"}
!2 = !{!"llvm.loop.unroll.enable"}
