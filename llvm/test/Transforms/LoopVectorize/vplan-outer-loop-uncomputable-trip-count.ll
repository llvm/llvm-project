; RUN: opt -passes=loop-vectorize -enable-vplan-native-path -pass-remarks=loop-vectorize -pass-remarks-analysis=loop-vectorize -disable-output %s 2>&1 | FileCheck --implicit-check-not='vectorized outer loop' %s

declare i1 @cond()

; Make sure we do not vectorize (or crash) on outer loops with uncomputable
; trip-counts.

define void @test() {
; CHECK: remark: <unknown>:0:0: loop not vectorized: loop induction variable could not be identified
;
entry:
  br label %outer.header

outer.header:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %outer.latch ]
  br label %inner

inner:
  %iv = phi i64 [ 0, %outer.header ], [ %iv.next, %inner ]
  %iv.next = add nuw nsw i64 %iv, 1
  %inner.ec = icmp eq i64 %iv.next, 0
  br i1 %inner.ec, label %outer.latch, label %inner

outer.latch:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %c = call i1 @cond()
  br i1 %c, label %exit, label %outer.header, !llvm.loop !0

exit:
  ret void
}

; The outer loop has no induction phi, but a computable backedge-taken count
; (0). It must be rejected like an inner loop without an integer induction,
; as the widest induction type is used for the canonical IV and trip count.
define void @outer_loop_without_induction(ptr %p) {
; CHECK: remark: <unknown>:0:0: loop not vectorized: Unsupported outer loop
;
entry:
  br label %outer.header

outer.header:
  br label %inner

inner:
  %iv = phi i64 [ 0, %outer.header ], [ %iv.next, %inner ]
  %gep = getelementptr inbounds i32, ptr %p, i64 %iv
  store i32 0, ptr %gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 8
  br i1 %ec, label %outer.latch, label %inner

outer.latch:
  %outer.ec = icmp eq i64 %iv.next, 8
  br i1 %outer.ec, label %exit, label %outer.header, !llvm.loop !1

exit:
  ret void
}

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.vectorize.enable"}
!3 = !{!"llvm.loop.vectorize.width", i32 4}
