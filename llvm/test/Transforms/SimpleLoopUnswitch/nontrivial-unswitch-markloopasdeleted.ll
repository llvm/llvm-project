; RUN: opt < %s -enable-loop-distribute -passes='loop-distribute,loop-mssa(simple-loop-unswitch<nontrivial>),loop-distribute' -o /dev/null -S -debug-pass-manager=verbose 2>&1 | FileCheck %s


; Running loop-distribute will result in LoopAccessAnalysis being required and
; cached in the LoopAnalysisManagerFunctionProxy.
;
; CHECK: Running analysis: LoopAccessAnalysis on Loop at depth 2 containing: %loop_a_inner<header><latch><exiting>


; Then simple-loop-unswitch is removing/replacing some loops (resulting in
; Loop objects used as key in the analyses cache is destroyed). So here we
; want to see that any analysis results cached on the destroyed loop is
; cleared. A special case here is that loop_a_inner is destroyed when
; unswitching the parent loop.
;
; The bug solved and verified by this test case was related to the
; SimpleLoopUnswitch not marking the Loop as removed, so we missed clearing
; the analysis caches.
;
; CHECK: Running pass: SimpleLoopUnswitchPass on Loop at depth 1 containing: %loop_begin<header>,%loop_b,%loop_b_inner,%loop_b_inner_exit,%loop_a,%loop_a_inner,%loop_a_inner_exit,%latch<latch><exiting>
; CHECK-NEXT: Clearing all analysis results for: loop_a_inner


; When running loop-distribute the second time we can see that loop_a_inner
; isn't analysed because the loop no longer exists (instead we find a new loop,
; loop_a_inner.us). This kind of verifies that it was correct to remove the
; loop_a_inner related analysis above.
;
; CHECK: Running analysis: LoopAccessAnalysis on Loop at depth 2 containing: %loop_a_inner.us<header><latch><exiting>


define i32 @test6(i1* %ptr, i1 %cond1, i32* %a.ptr, i32* %b.ptr) {
entry:
  br label %loop_begin

loop_begin:
  %v = load i1, i1* %ptr
  br i1 %cond1, label %loop_a, label %loop_b

loop_a:
  br label %loop_a_inner

loop_a_inner:
  %va = load i1, i1* %ptr
  %a = load i32, i32* %a.ptr
  br i1 %va, label %loop_a_inner, label %loop_a_inner_exit

loop_a_inner_exit:
  %a.lcssa = phi i32 [ %a, %loop_a_inner ]
  br label %latch

loop_b:
  br label %loop_b_inner

loop_b_inner:
  %vb = load i1, i1* %ptr
  %b = load i32, i32* %b.ptr
  br i1 %vb, label %loop_b_inner, label %loop_b_inner_exit

loop_b_inner_exit:
  %b.lcssa = phi i32 [ %b, %loop_b_inner ]
  br label %latch

latch:
  %ab.phi = phi i32 [ %a.lcssa, %loop_a_inner_exit ], [ %b.lcssa, %loop_b_inner_exit ]
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  %ab.lcssa = phi i32 [ %ab.phi, %latch ]
  ret i32 %ab.lcssa
}
