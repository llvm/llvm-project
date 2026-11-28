; RUN: opt -passes='loop(loop-unroll-full)' -S < %s | FileCheck %s
;
; Fully unrolling the outer loop yields 1599 sibling copies of the inner loop.
; Computing one copy's backedge-taken count used to recurse into the previous
; copy: proving a copy's entry guard means reading the previous copy's compare,
; and building a SCEV for that compare's operand asked for its range, which
; needs that copy's own count. One stack frame per copy, ~1500 before overflow.

; CHECK: define void @f(i64 %s) {
; The bug needs the copies to exist: assert the unroll actually happened, so this
; test fails loudly instead of passing without exercising anything if the unroll
; heuristics ever change.
; CHECK-COUNT-1600: call void @sink()

target triple = "x86_64-unknown-linux-gnu"

define void @f(i64 %s) {
entry:
  %entered = icmp slt i64 %s, 0
  br i1 %entered, label %outer.preheader, label %exit

outer.preheader:
  br label %outer

outer:
  %k = phi i32 [ %inc, %outer.latch ], [ 0, %outer.preheader ]
  br label %inner

exit.loopexit:
  br label %exit

exit:
  ret void

outer.latch:
  %inc = add nuw nsw i32 %k, 1
  %again = icmp samesign ult i32 %k, 1599
  br i1 %again, label %outer, label %exit.loopexit, !llvm.loop !1

inner:
  %j = phi i64 [ %s, %outer ], [ %add, %inner ]
  call void @sink()
  %add = add nsw i64 %j, 3
  %more = icmp slt i64 %j, -3
  br i1 %more, label %inner, label %outer.latch, !llvm.loop !0
}

declare void @sink()

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.mustprogress"}
!3 = !{!"llvm.loop.unroll.enable"}
