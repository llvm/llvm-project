; RUN: opt -passes=loop-vectorize -mtriple=x86_64-apple-macosx -debug-only=loop-vectorize -disable-output %s 2>&1 | FileCheck %s

; REQUIRES: asserts

; Check the cost of unconditional branches in the scalar loop. Use minsize, so
; branches are not free.

; The unconditional branch from %then to %latch cannot be folded, as %latch has
; multiple predecessors.
define void @uncond_br_to_block_with_multiple_preds(ptr noalias %dst, ptr noalias %src) minsize {
; CHECK-LABEL: LV: Checking a loop in 'uncond_br_to_block_with_multiple_preds'
; CHECK:       LV: Computing best VF using cost kind: Code Size
; CHECK-NEXT:  Cost of 0 for VF 1: EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]
; CHECK-NEXT:  Cost of 0 for VF 1: EMIT ir<%g.src> = getelementptr inbounds ir<%src>, ir<%iv>
; CHECK-NEXT:  Cost of 1 for VF 1: EMIT-SCALAR ir<%l> = load ir<%g.src>
; CHECK-NEXT:  Cost of 1 for VF 1: EMIT ir<%c> = icmp eq ir<%l>, ir<0>
; CHECK-NEXT:  Cost of 1 for VF 1: EMIT branch-on-cond ir<%c> (!vplan.prof.estimated estimated {805306368, 1342177280})
; CHECK-NEXT:  Cost of 0 for VF 1: EMIT ir<%g.dst> = getelementptr inbounds ir<%dst>, ir<%iv> (!vplan.execution.frequency 6917529027641081855 (37.5%, estimated))
; CHECK-NEXT:  Cost of 2 for VF 1: EMIT store ir<1>, ir<%g.dst> (!vplan.execution.frequency 6917529027641081855 (37.5%, estimated))
; CHECK-NEXT:  Cost of 1 for VF 1: EMIT ir<%iv.next> = add nuw nsw ir<%iv>, ir<1>
; CHECK-NEXT:  Cost of 1 for VF 1: EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1000>
; CHECK-NEXT:  Cost of 1 for VF 1: EMIT branch-on-cond ir<%ec>
; CHECK-NEXT:  LV: Scalar loop costs: 9.
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %g.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %l = load i32, ptr %g.src
  %c = icmp eq i32 %l, 0
  br i1 %c, label %then, label %latch

then:
  %g.dst = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i32 1, ptr %g.dst
  br label %latch

latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1000
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; The unconditional branch from %loop to %latch can be folded, as %latch has a
; single predecessor, so it is free.
define void @uncond_br_to_block_with_single_pred(ptr noalias %dst, ptr noalias %src) minsize {
; CHECK-LABEL: LV: Checking a loop in 'uncond_br_to_block_with_single_pred'
; CHECK:       LV: Computing best VF using cost kind: Code Size
; CHECK-NEXT:  Cost of 0 for VF 1: EMIT-SCALAR ir<%iv> = phi [ ir<0>, vector.ph ], [ ir<%iv.next>, latch ]
; CHECK-NEXT:  Cost of 0 for VF 1: EMIT ir<%g.src> = getelementptr inbounds ir<%src>, ir<%iv>
; CHECK-NEXT:  Cost of 1 for VF 1: EMIT-SCALAR ir<%l> = load ir<%g.src>
; CHECK-NEXT:  Cost of 0 for VF 1: EMIT ir<%g.dst> = getelementptr inbounds ir<%dst>, ir<%iv>
; CHECK-NEXT:  Cost of 2 for VF 1: EMIT store ir<%l>, ir<%g.dst>
; CHECK-NEXT:  Cost of 1 for VF 1: EMIT ir<%iv.next> = add nuw nsw ir<%iv>, ir<1>
; CHECK-NEXT:  Cost of 1 for VF 1: EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<1000>
; CHECK-NEXT:  Cost of 1 for VF 1: EMIT branch-on-cond ir<%ec>
; CHECK-NEXT:  LV: Scalar loop costs: 6.
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %g.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %l = load i32, ptr %g.src
  br label %latch

latch:
  %g.dst = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i32 %l, ptr %g.dst
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1000
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
