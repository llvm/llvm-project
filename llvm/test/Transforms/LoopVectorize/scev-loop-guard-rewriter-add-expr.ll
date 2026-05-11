; RUN: opt -passes=loop-vectorize -force-vector-width=8 -S < %s | FileCheck %s
;
; Test that applyLoopGuards correctly rewrites SCEVAddExpr trip counts.
;
; The trip count is `a + b` (a SCEVAddExpr). The loop guard `tc >= 8` means
; inside the loop we know tc >= 8. The SCEVLoopGuardRewriter must look up the
; full AddExpr in the rewrite map to propagate this constraint to LV.
;
; Without the fix, visitAddExpr only rewrites operands and misses the map
; entry for the whole expression, causing LV to emit a redundant runtime check.

; CHECK-LABEL: @add_expr_loop_guard
; CHECK-NOT:   min.iters.check
; CHECK:       vector.body:

define void @add_expr_loop_guard(i32 %a, i32 %b, ptr %p) {
entry:
  %tc = add nuw i32 %a, %b
  %guard = icmp uge i32 %tc, 8
  br i1 %guard, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %iv = phi i32 [ 0, %loop.ph ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds i8, ptr %p, i32 %iv
  store i8 0, ptr %gep, align 1
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %tc
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}
