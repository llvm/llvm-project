; RUN: opt -passes=loop-vectorize -force-vector-width=16 -S < %s | FileCheck %s
;
; Test that applyLoopGuards correctly rewrites SCEVMulExpr trip counts.
;
; The trip count here is `n & -16` which SCEV models as `(16 * (n /u 16))`.
; The loop guard `tc != 0` combined with divisibility info means tc >= 16.
; The SCEVLoopGuardRewriter must look up the full MulExpr in the rewrite map
; (not just rewrite its operands) to propagate this constraint.
;
; Without the fix, visitMulExpr only rewrites operands and misses the map
; entry for the whole expression, causing LV to emit a redundant runtime check.

; CHECK-LABEL: @mul_expr_loop_guard
; CHECK-NOT:   min.iters.check
; CHECK:       vector.body:

define void @mul_expr_loop_guard(i32 %n, ptr %p) {
entry:
  %tc = and i32 %n, -16
  %cmp = icmp ne i32 %tc, 0
  br i1 %cmp, label %loop.ph, label %exit

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
