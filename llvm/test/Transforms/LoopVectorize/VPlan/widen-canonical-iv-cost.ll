; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:   -force-target-supports-masked-memory-ops -tail-folding-policy=must-fold-tail \
; RUN:   -debug-only=loop-vectorize -disable-output %s 2>&1 | FileCheck %s

; Two reductions force the canonical IV to be widened so its lane values
; can feed the active-lane-mask compare under tail folding. Check the
; cost reported for VPWidenCanonicalIVRecipe.

; CHECK-LABEL: LV: Checking a loop in 'two_reductions'
; CHECK:       Cost of 0 for VF 4: EMIT vp<{{.*}}> = WIDEN-CANONICAL-INDUCTION

define i32 @two_reductions(i64 %N, ptr %a, ptr %b) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %sum.a = phi i32 [ 0, %entry ], [ %sum.a.next, %loop ]
  %sum.b = phi i32 [ 0, %entry ], [ %sum.b.next, %loop ]
  %ga = getelementptr inbounds i32, ptr %a, i64 %iv
  %gb = getelementptr inbounds i32, ptr %b, i64 %iv
  %la = load i32, ptr %ga, align 4
  %lb = load i32, ptr %gb, align 4
  %sum.a.next = add i32 %sum.a, %la
  %sum.b.next = add i32 %sum.b, %lb
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %N
  br i1 %ec, label %exit, label %loop

exit:
  %r.a = phi i32 [ %sum.a.next, %loop ]
  %r.b = phi i32 [ %sum.b.next, %loop ]
  %res = add i32 %r.a, %r.b
  ret i32 %res
}
