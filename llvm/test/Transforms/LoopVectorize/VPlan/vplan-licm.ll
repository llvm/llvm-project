; Test the licm VPlan transform that performs loop-invariant code motion.

; RUN: opt -passes='vplan-test<create-loop-regions;widen-from-metadata;licm>' -disable-output %s | FileCheck %s

; Check that loop-invariant add is hoisted to vector.ph.
; CHECK-LABEL: VPlan ' for UF>=1' {
; CHECK:      vector.ph:
; CHECK-NEXT:   WIDEN ir<%invariant> = add ir<%N>, ir<100>
; CHECK-NEXT: Successor(s): vector loop
; CHECK:      <x1> vector loop: {
; CHECK:        vector.body:
; CHECK-NOT:      WIDEN ir<%invariant> = add
; CHECK:        No successors
; CHECK:      }

define void @test_licm(ptr %A, i64 %N) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  ; Loop-invariant computation that should be hoisted
  %invariant = add i64 %N, 100, !vplan.widen !{}
  %gep = getelementptr i32, ptr %A, i64 %iv
  %val = load i32, ptr %gep
  %add = add i32 %val, 1, !vplan.widen !{}
  store i32 %add, ptr %gep
  %iv.next = add i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %invariant
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
