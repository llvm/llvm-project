; Test the widen-from-metadata VPlan transform that converts VPInstructions to
; recipes based on !vplan.widen and !vplan.replicate metadata.

; RUN: opt -passes='vplan-test<create-loop-regions;widen-from-metadata>' -disable-output %s | FileCheck %s --check-prefix=WIDEN
; RUN: opt -passes='vplan-test<create-loop-regions;widen-from-metadata>' -disable-output %s | FileCheck %s --check-prefix=REPLICATE

; Test widen conversion: VPInstruction with !vplan.widen
; should be converted to VPWidenRecipe (WIDEN instead of EMIT).
; WIDEN-LABEL: VPlan ' for UF>=1' {
; WIDEN: <x1> vector loop: {
; WIDEN:   vector.body:
; WIDEN:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; WIDEN:     EMIT ir<%gep> = getelementptr
; WIDEN:     EMIT ir<%val> = load ir<%gep>
; WIDEN:     WIDEN ir<%add> = add ir<%val>, ir<1>
; WIDEN:     EMIT store ir<%add>, ir<%gep>
; WIDEN:     EMIT branch-on-count
; WIDEN:   No successors
; WIDEN: }

; Test replicate conversion: VPInstruction with !vplan.replicate
; should be converted to VPReplicateRecipe (REPLICATE instead of EMIT).
; REPLICATE-LABEL: VPlan ' for UF>=1' {
; REPLICATE: <x1> vector loop: {
; REPLICATE:   vector.body:
; REPLICATE:     EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; REPLICATE:     EMIT ir<%gep> = getelementptr
; REPLICATE:     EMIT ir<%val> = load ir<%gep>
; REPLICATE:     REPLICATE ir<%add> = add ir<%val>, ir<1>
; REPLICATE:     EMIT store ir<%add>, ir<%gep>
; REPLICATE:     EMIT branch-on-count
; REPLICATE:   No successors
; REPLICATE: }

define void @test_widen(ptr %A, i64 %N) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i32, ptr %A, i64 %iv
  %val = load i32, ptr %gep
  %add = add i32 %val, 1, !vplan.widen !{}
  store i32 %add, ptr %gep
  %iv.next = add i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @test_replicate(ptr %A, i64 %N) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i32, ptr %A, i64 %iv
  %val = load i32, ptr %gep
  %add = add i32 %val, 1, !vplan.replicate !{}
  store i32 %add, ptr %gep
  %iv.next = add i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
