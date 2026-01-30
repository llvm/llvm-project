; Test the simplify-recipes VPlan transform that simplifies recipe patterns.

; RUN: opt -passes='vplan-test<create-loop-regions;widen-from-metadata;simplify-recipes>' -disable-output %s | FileCheck %s

; Check that 'or %val, 0' is simplified, so store uses %val directly.
; CHECK-LABEL: VPlan ' for UF>=1' {
; CHECK: <x1> vector loop: {
; CHECK:   vector.body:
; CHECK:     EMIT ir<%val> = load ir<%gep>
; CHECK:     EMIT store ir<%val>, ir<%gep>
; CHECK:   No successors
; CHECK: }

define void @test_simplify(ptr %A, i64 %N) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i32, ptr %A, i64 %iv
  %val = load i32, ptr %gep
  ; Identity: or X, 0 -> X (should be simplified away)
  %result = or i32 %val, 0, !vplan.widen !{}
  store i32 %result, ptr %gep
  %iv.next = add i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
