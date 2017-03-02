; This test checks that nested loops are revisited in various scenarios when
; unrolling. Note that if we ever start doing outer loop peeling a test case
; for that should be added here that will look essentially like a hybrid of the
; current two cases.
;
; RUN: opt < %s -disable-output -debug-pass-manager 2>&1 \
; RUN:     -passes='require<opt-remark-emit>,loop(unroll)' \
; RUN:     | FileCheck %s
;
; Also run in a special mode that visits children.
; RUN: opt < %s -disable-output -debug-pass-manager -unroll-revisit-child-loops 2>&1 \
; RUN:     -passes='require<opt-remark-emit>,loop(unroll)' \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-CHILDREN

; Basic test is fully unrolled and we revisit the post-unroll new sibling
; loops, including the ones that used to be child loops.
define void @full_unroll(i1* %ptr) {
; CHECK-LABEL: FunctionToLoopPassAdaptor{{.*}} on full_unroll
; CHECK-NOT: LoopUnrollPass

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
; CHECK: LoopUnrollPass on Loop at depth 3 containing: %l0.0.0<header>
; CHECK-NOT: LoopUnrollPass

l0.0.1.ph:
  br label %l0.0.1

l0.0.1:
  %cond.0.0.1 = load volatile i1, i1* %ptr
  br i1 %cond.0.0.1, label %l0.0.1, label %l0.0.latch
; CHECK: LoopUnrollPass on Loop at depth 3 containing: %l0.0.1<header>
; CHECK-NOT: LoopUnrollPass

l0.0.latch:
  %cmp = icmp slt i32 %iv.next, 2
  br i1 %cmp, label %l0.0, label %l0.latch
; CHECK: LoopUnrollPass on Loop at depth 2 containing: %l0.0
; CHECK-NOT: LoopUnrollPass
;
; Unrolling occurs, so we visit what were the inner loops twice over. First we
; visit their clones, and then we visit the original loops re-parented.
; CHECK: LoopUnrollPass on Loop at depth 2 containing: %l0.0.1.1<header>
; CHECK-NOT: LoopUnrollPass
; CHECK: LoopUnrollPass on Loop at depth 2 containing: %l0.0.0.1<header>
; CHECK-NOT: LoopUnrollPass
; CHECK: LoopUnrollPass on Loop at depth 2 containing: %l0.0.1<header>
; CHECK-NOT: LoopUnrollPass
; CHECK: LoopUnrollPass on Loop at depth 2 containing: %l0.0.0<header>
; CHECK-NOT: LoopUnrollPass

l0.latch:
  br label %l0
; CHECK: LoopUnrollPass on Loop at depth 1 containing: %l0<header>
; CHECK-NOT: LoopUnrollPass

exit:
  ret void
}

; Now we test forced runtime partial unrolling with metadata. Here we end up
; duplicating child loops without changing their structure and so they aren't by
; default visited, but will be visited with a special parameter.
define void @partial_unroll(i32 %count, i1* %ptr) {
; CHECK-LABEL: FunctionToLoopPassAdaptor{{.*}} on partial_unroll
; CHECK-NOT: LoopUnrollPass

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
; CHECK: LoopUnrollPass on Loop at depth 3 containing: %l0.0.0<header>
; CHECK-NOT: LoopUnrollPass

l0.0.1.ph:
  br label %l0.0.1

l0.0.1:
  %cond.0.0.1 = load volatile i1, i1* %ptr
  br i1 %cond.0.0.1, label %l0.0.1, label %l0.0.latch
; CHECK: LoopUnrollPass on Loop at depth 3 containing: %l0.0.1<header>
; CHECK-NOT: LoopUnrollPass

l0.0.latch:
  %cmp = icmp slt i32 %iv.next, %count
  br i1 %cmp, label %l0.0, label %l0.latch, !llvm.loop !1
; CHECK: LoopUnrollPass on Loop at depth 2 containing: %l0.0
; CHECK-NOT: LoopUnrollPass
;
; Partial unrolling occurs which introduces both new child loops and new sibling
; loops. We only visit the child loops in a special mode, not by default.
; CHECK-CHILDREN: LoopUnrollPass on Loop at depth 3 containing: %l0.0.0<header>
; CHECK-CHILDREN-NOT: LoopUnrollPass
; CHECK-CHILDREN: LoopUnrollPass on Loop at depth 3 containing: %l0.0.1<header>
; CHECK-CHILDREN-NOT: LoopUnrollPass
; CHECK-CHILDREN: LoopUnrollPass on Loop at depth 3 containing: %l0.0.0.1<header>
; CHECK-CHILDREN-NOT: LoopUnrollPass
; CHECK-CHILDREN: LoopUnrollPass on Loop at depth 3 containing: %l0.0.1.1<header>
; CHECK-CHILDREN-NOT: LoopUnrollPass
;
; When we revisit children, we also revisit the current loop.
; CHECK-CHILDREN: LoopUnrollPass on Loop at depth 2 containing: %l0.0<header>
; CHECK-CHILDREN-NOT: LoopUnrollPass
;
; Revisit the children of the outer loop that are part of the epilogue.
; 
; CHECK: LoopUnrollPass on Loop at depth 2 containing: %l0.0.0.epil<header>
; CHECK-NOT: LoopUnrollPass
; CHECK: LoopUnrollPass on Loop at depth 2 containing: %l0.0.1.epil<header>
; CHECK-NOT: LoopUnrollPass
l0.latch:
  br label %l0
; CHECK: LoopUnrollPass on Loop at depth 1 containing: %l0<header>
; CHECK-NOT: LoopUnrollPass

exit:
  ret void
}
!1 = !{!1, !2}
!2 = !{!"llvm.loop.unroll.count", i32 2}
