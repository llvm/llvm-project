; Test the remove-dead-recipes VPlan transform that removes unused recipes.

; RUN: opt -passes=loop-vectorize -vplan-test-transform='create-loop-regions,widen-from-metadata,remove-dead-recipes,print' -disable-output %s 2>&1 | FileCheck %s

; Check that dead recipes (%cmp and %4) are removed.
; CHECK-LABEL: VPlan ' for UF>=1' {
; CHECK: <x1> vector loop: {
; CHECK:   vector.body:
; CHECK:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK:     EMIT ir<%gep> = getelementptr
; CHECK:     EMIT ir<%val> = load ir<%gep>
; CHECK:     WIDEN ir<%add> = add ir<%val>, ir<1>
; CHECK:     EMIT store ir<%add>, ir<%gep>
; CHECK-NOT: EMIT ir<%cmp> = icmp
; CHECK-NOT: EMIT vp<{{%.+}}> = not ir<%cmp>
; CHECK:     EMIT branch-on-count
; CHECK:   No successors
; CHECK: }

define void @test_dce(ptr %A, i64 %N) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i32, ptr %A, i64 %iv
  %val = load i32, ptr %gep
  %add = add i32 %val, 1, !vplan.widen !0
  store i32 %add, ptr %gep
  %iv.next = add i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

!0 = !{!"widen"}
