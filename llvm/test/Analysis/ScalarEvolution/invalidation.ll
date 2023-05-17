; Test that SCEV gets invalidated when one of its dependencies is invalidated.
;
; Each of the RUNs checks that the pass manager runs SCEV, then invalidates it
; due to a dependency being invalidated, and then re-urns it. This will
; directly fail and indicates a failure that would occur later if we ddidn't
; invalidate SCEV in this way.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; RUN: opt < %s -passes='require<scalar-evolution>,invalidate<domtree>,print<scalar-evolution>' \
; RUN:     -debug-pass-manager -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefixes=CHECK,CHECK-DT-INVALIDATE
;
; CHECK-DT-INVALIDATE: Running pass: RequireAnalysisPass
; CHECK-DT-INVALIDATE: Running analysis: ScalarEvolutionAnalysis
; CHECK-DT-INVALIDATE: Running analysis: DominatorTreeAnalysis
; CHECK-DT-INVALIDATE: Running pass: InvalidateAnalysisPass
; CHECK-DT-INVALIDATE: Invalidating analysis: DominatorTreeAnalysis
; CHECK-DT-INVALIDATE: Running pass: ScalarEvolutionPrinterPass
; CHECK-DT-INVALIDATE: Running analysis: ScalarEvolutionAnalysis
; CHECK-DT-INVALIDATE: Running analysis: DominatorTreeAnalysis

; RUN: opt < %s -passes='require<scalar-evolution>,invalidate<loops>,print<scalar-evolution>' \
; RUN:     -debug-pass-manager -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefixes=CHECK,CHECK-LI-INVALIDATE
;
; CHECK-LI-INVALIDATE: Running pass: RequireAnalysisPass
; CHECK-LI-INVALIDATE: Running analysis: ScalarEvolutionAnalysis
; CHECK-LI-INVALIDATE: Running analysis: LoopAnalysis
; CHECK-LI-INVALIDATE: Running pass: InvalidateAnalysisPass
; CHECK-LI-INVALIDATE: Invalidating analysis: LoopAnalysis
; CHECK-LI-INVALIDATE: Running pass: ScalarEvolutionPrinterPass
; CHECK-LI-INVALIDATE: Running analysis: ScalarEvolutionAnalysis
; CHECK-LI-INVALIDATE: Running analysis: LoopAnalysis

; This test isn't particularly interesting, its just enough to make sure we
; actually do some work inside of SCEV so that if we regress here despite the
; debug pass printing continuing to match, ASan and other tools can catch it.
define void @test(i32 %n) {
; CHECK-LABEL: Classifying expressions for: @test
; CHECK: Loop %loop: backedge-taken count is 14
; CHECK: Loop %loop: constant max backedge-taken count is 14
; CHECK: Loop %loop: Predicated backedge-taken count is 14

entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add nsw i32 %iv, 3
  %becond = icmp ne i32 %iv.inc, 46
  br i1 %becond, label %loop, label %leave

leave:
  ret void
}
