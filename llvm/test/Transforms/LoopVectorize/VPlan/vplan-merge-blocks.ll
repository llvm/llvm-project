; Test the merge-blocks VPlan transform that merges blocks into predecessors.
; This test uses a loop with early exit to create multiple blocks in VPlan.

; RUN: opt -passes='vplan-test<create-loop-regions;widen-from-metadata;merge-blocks>' -disable-output %s | FileCheck %s

; Check that continue block is merged into vector.body.
; CHECK-LABEL: VPlan ' for UF>=1' {
; CHECK: <x1> vector loop: {
; CHECK:   vector.body:
; CHECK:     EMIT ir<%cond> = icmp sgt ir<%val>, ir<100>
; CHECK-NOT: Successor(s): continue
; CHECK:     WIDEN ir<%add> = add ir<%val>, ir<1>
; CHECK:     EMIT branch-on-count
; CHECK:   No successors
; CHECK: }

define void @test_merge_blocks(ptr %A, i64 %N) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %continue ]
  %gep = getelementptr i32, ptr %A, i64 %iv
  %val = load i32, ptr %gep
  ; Early exit condition
  %cond = icmp sgt i32 %val, 100
  br i1 %cond, label %exit, label %continue

continue:
  %add = add i32 %val, 1, !vplan.widen !{}
  store i32 %add, ptr %gep
  %iv.next = add i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
