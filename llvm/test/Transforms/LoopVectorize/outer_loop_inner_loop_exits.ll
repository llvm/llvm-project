; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -enable-vplan-native-path -debug-only=loop-vectorize -disable-output -S %s 2>&1 | FileCheck %s

; Inner loops of an outer loop to be vectorized must exit via their latch only.
; An exit with an outer-loop invariant (uniform) condition passes the branch
; check for supported conditional branches, so it needs to be rejected
; separately.

; The inner loop exits early to the outer loop's latch.
define void @inner_loop_uniform_exit_to_outer_latch(ptr %a, ptr %b, i64 %N, i64 %M, i1 %c) {
; CHECK-LABEL: LV: Checking a loop in 'inner_loop_uniform_exit_to_outer_latch'
; CHECK: LV: Not vectorizing: Nested loop does not exit via its latch.
; CHECK: LV: Not vectorizing: Unsupported outer loop.
entry:
  br label %outer.body

outer.body:
  %iv.outer = phi i64 [ 0, %entry ], [ %iv.outer.next, %outer.inc ]
  %mul = mul nsw i64 %iv.outer, %M
  br label %inner.body

inner.body:
  %iv.inner = phi i64 [ 0, %outer.body ], [ %iv.inner.next, %inner.latch ]
  %idx = add nsw i64 %iv.inner, %mul
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %idx
  %l = load i32, ptr %gep.b, align 4
  %add = add nsw i32 %l, 1
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %idx
  store i32 %add, ptr %gep.a, align 4
  br i1 %c, label %outer.inc, label %inner.latch

inner.latch:
  %iv.inner.next = add nuw nsw i64 %iv.inner, 1
  %ec.inner = icmp eq i64 %iv.inner.next, %M
  br i1 %ec.inner, label %outer.inc, label %inner.body

outer.inc:
  %iv.outer.next = add nuw nsw i64 %iv.outer, 1
  %ec.outer = icmp eq i64 %iv.outer.next, %N
  br i1 %ec.outer, label %exit, label %outer.body, !llvm.loop !0

exit:
  ret void
}

; Same as above, but with the extra exit in the innermost loop of a 3-deep
; nest, so all nested loops need to be checked, not just the immediate
; children of the loop to be vectorized.
define void @innermost_loop_uniform_exit(ptr %a, ptr %b, i64 %N, i64 %M, i64 %K, i1 %c) {
; CHECK-LABEL: LV: Checking a loop in 'innermost_loop_uniform_exit'
; CHECK: LV: Not vectorizing: Nested loop does not exit via its latch.
; CHECK: LV: Not vectorizing: Unsupported outer loop.
entry:
  br label %outer.body

outer.body:
  %iv.outer = phi i64 [ 0, %entry ], [ %iv.outer.next, %outer.inc ]
  %mul = mul nsw i64 %iv.outer, %M
  br label %middle.body

middle.body:
  %iv.middle = phi i64 [ 0, %outer.body ], [ %iv.middle.next, %middle.latch ]
  br label %inner.body

inner.body:
  %iv.inner = phi i64 [ 0, %middle.body ], [ %iv.inner.next, %inner.latch ]
  %idx = add nsw i64 %iv.inner, %mul
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %idx
  %l = load i32, ptr %gep.b, align 4
  %add = add nsw i32 %l, 1
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %idx
  store i32 %add, ptr %gep.a, align 4
  br i1 %c, label %middle.latch, label %inner.latch

inner.latch:
  %iv.inner.next = add nuw nsw i64 %iv.inner, 1
  %ec.inner = icmp eq i64 %iv.inner.next, %K
  br i1 %ec.inner, label %middle.latch, label %inner.body

middle.latch:
  %iv.middle.next = add nuw nsw i64 %iv.middle, 1
  %ec.middle = icmp eq i64 %iv.middle.next, %M
  br i1 %ec.middle, label %outer.inc, label %middle.body

outer.inc:
  %iv.outer.next = add nuw nsw i64 %iv.outer, 1
  %ec.outer = icmp eq i64 %iv.outer.next, %N
  br i1 %ec.outer, label %exit, label %outer.body, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.vectorize.width", i32 4}
!2 = !{!"llvm.loop.vectorize.enable"}
