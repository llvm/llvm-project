; This test checks that nested loops are revisited in various scenarios when
; unrolling. Note that if we ever start doing outer loop peeling a test case
; for that should be added here that will look essentially like a hybrid of the
; current two cases.
;
; RUN: opt < %s -disable-output -debug-pass-manager 2>&1 \
; RUN: -passes='require<opt-remark-emit>,loop(loop-unroll-full)' \
; RUN:     | FileCheck %s
;
; Also run in a special mode that visits children.
; RUN: opt < %s -disable-output -debug-pass-manager -unroll-revisit-child-loops 2>&1 \
; RUN: -passes='require<opt-remark-emit>,loop(loop-unroll-full)' \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-CHILDREN

; Basic test is fully unrolled and we revisit the post-unroll new sibling
; loops, including the ones that used to be child loops.
define void @full_unroll(i1* %ptr) {
; CHECK-LABEL: OptimizationRemarkEmitterAnalysis on full_unroll
; CHECK-NOT: LoopFullUnrollPass

entry:
  br label %l0

l0:
  %cond.0 = load volatile i1, i1* %ptr
  br i1 %cond.0, label %l0.0.ph, label %exit

l0.0.ph:
  br label %l0.0

l0.0:
  %iv = phi i32 [ %iv.next, %l0.0.latch ], [ 0, %l0.0.ph ]
  %iv.next = add i32 %iv, 1
  br label %l0.0.0.ph

l0.0.0.ph:
  br label %l0.0.0

l0.0.0:
  %cond.0.0.0 = load volatile i1, i1* %ptr
  br i1 %cond.0.0.0, label %l0.0.0, label %l0.0.1.ph
; CHECK: LoopFullUnrollPass on l0.0.0
; CHECK-NOT: LoopFullUnrollPass

l0.0.1.ph:
  br label %l0.0.1

l0.0.1:
  %cond.0.0.1 = load volatile i1, i1* %ptr
  br i1 %cond.0.0.1, label %l0.0.1, label %l0.0.latch
; CHECK: LoopFullUnrollPass on l0.0.1
; CHECK-NOT: LoopFullUnrollPass

l0.0.latch:
  %cmp = icmp slt i32 %iv.next, 2
  br i1 %cmp, label %l0.0, label %l0.latch
; CHECK: LoopFullUnrollPass on l0.0
; CHECK-NOT: LoopFullUnrollPass
;
; Unrolling occurs, so we visit what were the inner loops twice over. First we
; visit their clones, and then we visit the original loops re-parented.
; CHECK: LoopFullUnrollPass on l0.0.1.1
; CHECK-NOT: LoopFullUnrollPass
; CHECK: LoopFullUnrollPass on l0.0.0.1
; CHECK-NOT: LoopFullUnrollPass
; CHECK: LoopFullUnrollPass on l0.0.1
; CHECK-NOT: LoopFullUnrollPass
; CHECK: LoopFullUnrollPass on l0.0.0
; CHECK-NOT: LoopFullUnrollPass

l0.latch:
  br label %l0
; CHECK: LoopFullUnrollPass on l0
; CHECK-NOT: LoopFullUnrollPass

exit:
  ret void
}

; Now we test forced runtime partial unrolling with metadata. Here we end up
; duplicating child loops without changing their structure and so they aren't by
; default visited, but will be visited with a special parameter.
define void @partial_unroll(i32 %count, i1* %ptr) {
; CHECK-LABEL: OptimizationRemarkEmitterAnalysis on partial_unroll
; CHECK-NOT: LoopFullUnrollPass

entry:
  br label %l0

l0:
  %cond.0 = load volatile i1, i1* %ptr
  br i1 %cond.0, label %l0.0.ph, label %exit

l0.0.ph:
  br label %l0.0

l0.0:
  %iv = phi i32 [ %iv.next, %l0.0.latch ], [ 0, %l0.0.ph ]
  %iv.next = add i32 %iv, 1
  br label %l0.0.0.ph

l0.0.0.ph:
  br label %l0.0.0

l0.0.0:
  %cond.0.0.0 = load volatile i1, i1* %ptr
  br i1 %cond.0.0.0, label %l0.0.0, label %l0.0.1.ph
; CHECK: LoopFullUnrollPass on l0.0.0
; CHECK-NOT: LoopFullUnrollPass

l0.0.1.ph:
  br label %l0.0.1

l0.0.1:
  %cond.0.0.1 = load volatile i1, i1* %ptr
  br i1 %cond.0.0.1, label %l0.0.1, label %l0.0.latch
; CHECK: LoopFullUnrollPass on l0.0.1
; CHECK-NOT: LoopFullUnrollPass

l0.0.latch:
  %cmp = icmp slt i32 %iv.next, %count
  br i1 %cmp, label %l0.0, label %l0.latch, !llvm.loop !1
; CHECK: LoopFullUnrollPass on l0.0
; CHECK-NOT: LoopFullUnrollPass
;
; Partial unrolling occurs which introduces both new child loops and new sibling
; loops. We only visit the child loops in a special mode, not by default.
; CHECK-CHILDREN: LoopFullUnrollPass on l0.0.0
; CHECK-CHILDREN-NOT: LoopFullUnrollPass
; CHECK-CHILDREN: LoopFullUnrollPass on l0.0.1
; CHECK-CHILDREN-NOT: LoopFullUnrollPass
; CHECK-CHILDREN: LoopFullUnrollPass on l0.0.0.1
; CHECK-CHILDREN-NOT: LoopFullUnrollPass
; CHECK-CHILDREN: LoopFullUnrollPass on l0.0.1.1
; CHECK-CHILDREN-NOT: LoopFullUnrollPass
;
; When we revisit children, we also revisit the current loop.
; CHECK-CHILDREN: LoopFullUnrollPass on l0.0
; CHECK-CHILDREN-NOT: LoopFullUnrollPass
;
; Revisit the children of the outer loop that are part of the epilogue.
;
; CHECK: LoopFullUnrollPass on l0.0.1.epil
; CHECK-NOT: LoopFullUnrollPass
; CHECK: LoopFullUnrollPass on l0.0.0.epil
; CHECK-NOT: LoopFullUnrollPass
l0.latch:
  br label %l0
; CHECK: LoopFullUnrollPass on l0
; CHECK-NOT: LoopFullUnrollPass

exit:
  ret void
}
!1 = !{!1, !2}
!2 = !{!"llvm.loop.unroll.count", i32 2}
