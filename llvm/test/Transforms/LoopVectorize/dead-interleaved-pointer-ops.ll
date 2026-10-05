; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -enable-interleaved-mem-accesses=true -debug-only=loop-vectorize -disable-output %s 2>&1 | FileCheck %s

; %offset.1 is first found to be dead only in the vector loop, as %gep.1
; is initially only ignored as pointer of an interleave group member. Once
; %l.1 and %gep.1 are ignored, %offset.1 becomes dead in the scalar loop too.
; This must also be propagated to its already visited operand %index.1.

; CHECK-LABEL: LV: Checking a loop in 'interleave_pointer_becomes_scalar_dead'
; CHECK:      Cost of 1 for VF 1: EMIT-SCALAR ir<%iv> = phi
; CHECK-NEXT: Cost of 1 for VF 1: EMIT ir<%iv.next> = add nuw ir<%iv>, ir<1>
; CHECK-NEXT: Cost of 1 for VF 1: EMIT ir<%ec> = icmp eq ir<%iv.next>, ir<%n>
; CHECK-NEXT: Cost of 1 for VF 1: EMIT branch-on-cond ir<%ec>
; CHECK-NEXT: LV: Scalar loop costs: 4.

define void @interleave_pointer_becomes_scalar_dead(ptr noalias %src, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %index.0 = mul i64 %iv, 3
  %gep.0 = getelementptr i32, ptr %src, i64 %index.0
  %l.0 = load i32, ptr %gep.0
  %index.1 = mul i64 %iv, 3
  %offset.1 = add i64 %index.1, 1
  %gep.1 = getelementptr i32, ptr %src, i64 %offset.1
  %l.1 = load i32, ptr %gep.1
  %offset.2 = add i64 %index.0, 2
  %gep.2 = getelementptr i32, ptr %src, i64 %offset.2
  %l.2 = load i32, ptr %gep.2
  %dead.user = add i64 %offset.1, 7
  %iv.next = add nuw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
