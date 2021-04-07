; Test that the loop PM infrastructure is invalidated appropriately.
;
; Check that we always nuke the LPM stuff when the loops themselves are
; invalidated.
; RUN: opt -disable-output -disable-verify -verify-cfg-preserved=0 -debug-pass-manager -aa-pipeline= %s 2>&1 \
; RUN:     -passes='loop(no-op-loop),invalidate<loops>,loop(no-op-loop)' \
; RUN:     | FileCheck %s --check-prefix=CHECK-LOOP-INV
;
; If we ended up building the standard analyses, their invalidation should nuke
; stuff as well.
; RUN: opt -disable-output -disable-verify -verify-cfg-preserved=0 -debug-pass-manager %s -aa-pipeline= 2>&1 \
; RUN:     -passes='loop(no-op-loop),invalidate<scalar-evolution>,loop(no-op-loop)' \
; RUN:     | FileCheck %s --check-prefix=CHECK-SCEV-INV
;
; Also provide a test that can delete loops after populating analyses for them.
; RUN: opt -disable-output -disable-verify -verify-cfg-preserved=0  -debug-pass-manager %s -aa-pipeline= 2>&1 \
; RUN:     -passes='loop(no-op-loop,loop-deletion),invalidate<scalar-evolution>,loop(no-op-loop)' \
; RUN:     | FileCheck %s --check-prefix=CHECK-SCEV-INV-AFTER-DELETE

define void @no_loops() {
; CHECK-LOOP-INV: Starting {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-LOOP-INV-NEXT: Running analysis: LoopAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: DominatorTreeAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: AssumptionAnalysis
; CHECK-LOOP-INV-NEXT: Running pass: LCSSAPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}LoopAnalysis
; CHECK-LOOP-INV-NEXT: Invalidating analysis: LoopAnalysis
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-LOOP-INV-NEXT: Running analysis: LoopAnalysis
; CHECK-LOOP-INV-NEXT: Running pass: LCSSAPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Function pass manager run.
;
; CHECK-SCEV-INV: Starting {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-SCEV-INV-NEXT: Running analysis: LoopAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: DominatorTreeAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: AssumptionAnalysis
; CHECK-SCEV-INV-NEXT: Running pass: LCSSAPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}ScalarEvolutionAnalysis
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-SCEV-INV-NEXT: Running pass: LCSSAPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Function pass manager run.

entry:
  ret void
}

define void @one_loop(i1* %ptr) {
; CHECK-LOOP-INV: Starting {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-LOOP-INV-NEXT: Running analysis: LoopAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: DominatorTreeAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: AssumptionAnalysis
; CHECK-LOOP-INV-NEXT: Running pass: LCSSAPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running analysis: AAManager
; CHECK-LOOP-INV-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: TargetIRAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}LoopAnalysis
; CHECK-LOOP-INV-NEXT: Clearing all analysis results for: <possibly invalidated loop>
; CHECK-LOOP-INV-NEXT: Invalidating analysis: LoopAnalysis
; CHECK-LOOP-INV-NEXT: Invalidating analysis: ScalarEvolutionAnalysis
; CHECK-LOOP-INV-NEXT: Invalidating analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-LOOP-INV-NEXT: Running analysis: LoopAnalysis
; CHECK-LOOP-INV-NEXT: Running pass: LCSSAPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Function pass manager run.
;
; CHECK-SCEV-INV: Starting {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-SCEV-INV-NEXT: Running analysis: LoopAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: DominatorTreeAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: AssumptionAnalysis
; CHECK-SCEV-INV-NEXT: Running pass: LCSSAPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running analysis: AAManager
; CHECK-SCEV-INV-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: TargetIRAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}ScalarEvolutionAnalysis
; CHECK-SCEV-INV-NEXT: Clearing all analysis results for: <possibly invalidated loop>
; CHECK-SCEV-INV-NEXT: Invalidating analysis: ScalarEvolutionAnalysis
; CHECK-SCEV-INV-NEXT: Invalidating analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-SCEV-INV-NEXT: Running pass: LCSSAPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Function pass manager run.

entry:
  br label %l0.header

l0.header:
  %flag0 = load volatile i1, i1* %ptr
  br i1 %flag0, label %l0.header, label %exit

exit:
  ret void
}

define void @nested_loops(i1* %ptr) {
; CHECK-LOOP-INV: Starting {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-LOOP-INV-NEXT: Running analysis: LoopAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: DominatorTreeAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: AssumptionAnalysis
; CHECK-LOOP-INV-NEXT: Running pass: LCSSAPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running analysis: AAManager
; CHECK-LOOP-INV-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: TargetIRAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-LOOP-INV: Finished {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}LoopAnalysis
; CHECK-LOOP-INV-NEXT: Clearing all analysis results for: <possibly invalidated loop>
; CHECK-LOOP-INV-NEXT: Clearing all analysis results for: <possibly invalidated loop>
; CHECK-LOOP-INV-NEXT: Invalidating analysis: LoopAnalysis
; CHECK-LOOP-INV-NEXT: Invalidating analysis: ScalarEvolutionAnalysis
; CHECK-LOOP-INV-NEXT: Invalidating analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-LOOP-INV-NEXT: Running analysis: LoopAnalysis
; CHECK-LOOP-INV-NEXT: Running pass: LCSSAPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-LOOP-INV: Finished {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Function pass manager run.
;
; CHECK-SCEV-INV: Starting {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-SCEV-INV-NEXT: Running analysis: LoopAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: DominatorTreeAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: AssumptionAnalysis
; CHECK-SCEV-INV-NEXT: Running pass: LCSSAPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running analysis: AAManager
; CHECK-SCEV-INV-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: TargetIRAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-SCEV-INV: Finished {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}ScalarEvolutionAnalysis
; CHECK-SCEV-INV-NEXT: Clearing all analysis results for: <possibly invalidated loop>
; CHECK-SCEV-INV-NEXT: Clearing all analysis results for: <possibly invalidated loop>
; CHECK-SCEV-INV-NEXT: Invalidating analysis: ScalarEvolutionAnalysis
; CHECK-SCEV-INV-NEXT: Invalidating analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-SCEV-INV-NEXT: Running pass: LCSSAPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-SCEV-INV: Finished {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Function pass manager run.

entry:
  br label %l.0.header

l.0.header:
  br label %l.0.0.header

l.0.0.header:
  %flag.0.0 = load volatile i1, i1* %ptr
  br i1 %flag.0.0, label %l.0.0.header, label %l.0.latch

l.0.latch:
  %flag.0 = load volatile i1, i1* %ptr
  br i1 %flag.0, label %l.0.header, label %exit

exit:
  ret void
}

define void @dead_loop() {
; CHECK-LOOP-INV: Starting {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-LOOP-INV-NEXT: Running analysis: LoopAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: DominatorTreeAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: AssumptionAnalysis
; CHECK-LOOP-INV-NEXT: Running pass: LCSSAPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running analysis: AAManager
; CHECK-LOOP-INV-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: TargetIRAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}LoopAnalysis
; CHECK-LOOP-INV-NEXT: Clearing all analysis results for: <possibly invalidated loop>
; CHECK-LOOP-INV-NEXT: Invalidating analysis: LoopAnalysis
; CHECK-LOOP-INV-NEXT: Invalidating analysis: ScalarEvolutionAnalysis
; CHECK-LOOP-INV-NEXT: Invalidating analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-LOOP-INV-NEXT: Running analysis: LoopAnalysis
; CHECK-LOOP-INV-NEXT: Running pass: LCSSAPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-LOOP-INV-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-LOOP-INV-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-LOOP-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Loop pass manager run.
; CHECK-LOOP-INV-NEXT: Finished {{.*}}Function pass manager run.
;
; CHECK-SCEV-INV: Starting {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-SCEV-INV-NEXT: Running analysis: LoopAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: DominatorTreeAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: AssumptionAnalysis
; CHECK-SCEV-INV-NEXT: Running pass: LCSSAPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running analysis: AAManager
; CHECK-SCEV-INV-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: TargetIRAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}ScalarEvolutionAnalysis
; CHECK-SCEV-INV-NEXT: Clearing all analysis results for: <possibly invalidated loop>
; CHECK-SCEV-INV-NEXT: Invalidating analysis: ScalarEvolutionAnalysis
; CHECK-SCEV-INV-NEXT: Invalidating analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running pass: LoopSimplifyPass
; CHECK-SCEV-INV-NEXT: Running pass: LCSSAPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Function pass manager run
; CHECK-SCEV-INV-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-SCEV-INV-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-SCEV-INV-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Running pass: NoOpLoopPass
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-NEXT: Finished {{.*}}Function pass manager run.
;
; CHECK-SCEV-INV-AFTER-DELETE-LABEL: Running pass: LoopSimplifyPass on dead_loop
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running analysis: LoopAnalysis
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running analysis: DominatorTreeAnalysis
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running analysis: AssumptionAnalysis
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running pass: LCSSAPass
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Finished {{.*}}Function pass manager run
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running analysis: AAManager
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running analysis: TargetLibraryAnalysis
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running analysis: ScalarEvolutionAnalysis
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running analysis: TargetIRAnalysis
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Starting {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running pass: NoOpLoopPass
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running pass: LoopDeletionPass
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Clearing all analysis results for:
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Finished {{.*}}Loop pass manager run.
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running pass: InvalidateAnalysisPass<{{.*}}ScalarEvolutionAnalysis
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Invalidating analysis: ScalarEvolutionAnalysis
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Invalidating analysis: InnerAnalysisManagerProxy<{{.*}}Loop
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Starting {{.*}}Function pass manager run
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running pass: LoopSimplifyPass
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Running pass: LCSSAPass
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Finished {{.*}}Function pass manager run
; CHECK-SCEV-INV-AFTER-DELETE-NEXT: Finished {{.*}}Function pass manager run.

entry:
  br label %l0.header

l0.header:
  br i1 false, label %l0.header, label %exit

exit:
  ret void
}
