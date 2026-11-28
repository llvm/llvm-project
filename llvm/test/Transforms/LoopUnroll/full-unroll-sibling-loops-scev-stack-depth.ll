; RUN: opt -passes='loop(loop-unroll-full)' -S %s | FileCheck %s

; Each unrolled copy of the inner loop is guarded by the exit of the previous
; one. Make sure computing their BTCs does not overflow the stack.

; CHECK-LABEL: define void @sibling_loops(
; CHECK:       inner.1499:
define void @sibling_loops(i64 %start) {
entry:
  %guard = icmp slt i64 %start, 0
  br i1 %guard, label %outer.header, label %exit

outer.header:
  %outer.iv = phi i32 [ 0, %entry ], [ %outer.iv.next, %outer.latch ]
  br label %inner

inner:
  %iv = phi i64 [ %start, %outer.header ], [ %iv.next, %inner ]
  %iv.next = add nsw i64 %iv, 1
  %ec = icmp slt i64 %iv.next, 0
  br i1 %ec, label %inner, label %outer.latch

outer.latch:
  %outer.iv.next = add i32 %outer.iv, 1
  %outer.ec = icmp ult i32 %outer.iv, 1499
  br i1 %outer.ec, label %outer.header, label %exit, !llvm.loop !0

exit:
  ret void
}

; CHECK-LABEL: define void @sibling_loops_logical_or_exit(
; CHECK:       inner.1499:
define void @sibling_loops_logical_or_exit(i64 %start, i1 %c) {
entry:
  %guard = icmp slt i64 %start, 0
  br i1 %guard, label %outer.header, label %exit

outer.header:
  %outer.iv = phi i32 [ 0, %entry ], [ %outer.iv.next, %outer.latch ]
  br label %inner

inner:
  %iv = phi i64 [ %start, %outer.header ], [ %iv.next, %inner ]
  %iv.next = add nsw i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, 0
  %ec = select i1 %cmp, i1 true, i1 %c
  br i1 %ec, label %inner, label %outer.latch

outer.latch:
  %outer.iv.next = add i32 %outer.iv, 1
  %outer.ec = icmp ult i32 %outer.iv, 1499
  br i1 %outer.ec, label %outer.header, label %exit, !llvm.loop !1

exit:
  ret void
}

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.unroll.enable"}
