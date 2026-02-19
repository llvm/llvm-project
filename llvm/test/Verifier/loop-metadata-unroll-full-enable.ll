; RUN: opt < %s -passes=verify -S -o /dev/null

; Test that the verifier accepts loop metadata containing both
; llvm.loop.unroll.full and llvm.loop.unroll.enable simultaneously.

define void @loop_full_and_enable(ptr nocapture %a) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %ptr = getelementptr inbounds i32, ptr %a, i64 %iv
  %val = load i32, ptr %ptr, align 4
  %inc = add nsw i32 %val, 1
  store i32 %inc, ptr %ptr, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp = icmp eq i64 %iv.next, 32
  br i1 %cmp, label %exit, label %for.body, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.unroll.full"}
!2 = !{!"llvm.loop.unroll.enable"}
