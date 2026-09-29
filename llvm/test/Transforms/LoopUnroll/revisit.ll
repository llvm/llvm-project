; This test checks that nested loops are revisited in various scenarios when
; unrolling. Note that if we ever start doing outer loop peeling a test case
; for that should be added here.
;
; RUN: opt < %s -disable-output -debug-pass-manager 2>&1 \
; RUN: -passes='require<opt-remark-emit>,loop(loop-unroll-full)' \
; RUN:     | FileCheck %s
; 
; Basic test is fully unrolled and we revisit the post-unroll new sibling
; loops, including the ones that used to be child loops.
define void @full_unroll(ptr %ptr) {
; CHECK-LABEL: OptimizationRemarkEmitterAnalysis on full_unroll
; CHECK-NOT: LoopFullUnrollPass

entry:
  br label %l0

l0:
  %cond.0 = load volatile i1, ptr %ptr
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
  %cond.0.0.0 = load volatile i1, ptr %ptr
  br i1 %cond.0.0.0, label %l0.0.0, label %l0.0.1.ph
; CHECK: LoopFullUnrollPass on loop %l0.0.0
; CHECK-NOT: LoopFullUnrollPass

l0.0.1.ph:
  br label %l0.0.1

l0.0.1:
  %cond.0.0.1 = load volatile i1, ptr %ptr
  br i1 %cond.0.0.1, label %l0.0.1, label %l0.0.latch
; CHECK: LoopFullUnrollPass on loop %l0.0.1 in function full_unroll
; CHECK-NOT: LoopFullUnrollPass

l0.0.latch:
  %cmp = icmp slt i32 %iv.next, 2
  br i1 %cmp, label %l0.0, label %l0.latch
; CHECK: LoopFullUnrollPass on loop %l0.0 in function full_unroll
; CHECK-NOT: LoopFullUnrollPass
;
; Unrolling occurs, so we visit what were the inner loops twice over. First we
; visit their clones, and then we visit the original loops re-parented.
; CHECK: LoopFullUnrollPass on loop %l0.0.1.1 in function full_unroll 
; CHECK-NOT: LoopFullUnrollPass
; CHECK: LoopFullUnrollPass on loop %l0.0.0.1 in function full_unroll
; CHECK-NOT: LoopFullUnrollPass
; CHECK: LoopFullUnrollPass on loop %l0.0.1 in function full_unroll
; CHECK-NOT: LoopFullUnrollPass
; CHECK: LoopFullUnrollPass on loop %l0.0.0 in function full_unroll
; CHECK-NOT: LoopFullUnrollPass

l0.latch:
  br label %l0
; CHECK: LoopFullUnrollPass on loop %l0 in function full_unroll
; CHECK-NOT: LoopFullUnrollPass

exit:
  ret void
}
