;RUN: opt %s -aa-pipeline= -passes='adce,loop(loop-rotate),adce' -S -verify-analysis-invalidation=0 -debug-pass-manager -debug-only=loop-rotate 2>&1 | FileCheck %s
;RUN: opt %s -aa-pipeline= -passes='adce,loop-mssa(loop-rotate),adce' -S -verify-analysis-invalidation=0 -debug-pass-manager -debug-only=loop-rotate -verify-memoryssa 2>&1 | FileCheck %s --check-prefix=MSSA
;REQUIRES: asserts

; This test is to make sure we invalidate the post dominator pass after loop rotate simplifies the loop latch.
; The adce passes are here to make sure post dominator analysis is required.

; CHECK: Running pass: ADCEPass on f
; CHECK-NEXT: Running analysis: PostDominatorTreeAnalysis on f
; CHECK-NEXT: Running pass: LoopSimplifyPass on f
; CHECK-NEXT: Running analysis: LoopAnalysis on f
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on f
; CHECK-NEXT: Running analysis: AssumptionAnalysis on f
; CHECK-NEXT: Running analysis: TargetIRAnalysis on f
; CHECK-NEXT: Running pass: LCSSAPass on f
; CHECK-NEXT: Running analysis: AAManager on f
; CHECK-NEXT: Running analysis: TargetLibraryAnalysis on f
; CHECK-NEXT: Running analysis: ScalarEvolutionAnalysis on f
; CHECK-NEXT: Running analysis: InnerAnalysisManagerProxy{{.*}} on f
; CHECK-NEXT: Running pass: LoopRotatePass on loop %bb in function f
; CHECK-NEXT: Folding loop latch bb4 into bb
; CHECK-NEXT: Invalidating analysis: PostDominatorTreeAnalysis on f
; CHECK-NEXT: Running pass: ADCEPass on f
; CHECK-NEXT: Running analysis: PostDominatorTreeAnalysis on f

; MSSA: Running pass: ADCEPass on f
; MSSA-NEXT: Running analysis: PostDominatorTreeAnalysis on f
; MSSA-NEXT: Running pass: LoopSimplifyPass on f
; MSSA-NEXT: Running analysis: LoopAnalysis on f
; MSSA-NEXT: Running analysis: DominatorTreeAnalysis on f
; MSSA-NEXT: Running analysis: AssumptionAnalysis on f
; MSSA-NEXT: Running analysis: TargetIRAnalysis on f
; MSSA-NEXT: Running pass: LCSSAPass on f
; MSSA-NEXT: Running analysis: MemorySSAAnalysis on f
; MSSA-NEXT: Running analysis: AAManager on f
; MSSA-NEXT: Running analysis: TargetLibraryAnalysis on f
; MSSA-NEXT: Running analysis: ScalarEvolutionAnalysis on f
; MSSA-NEXT: Running analysis: InnerAnalysisManagerProxy{{.*}} on f
; MSSA-NEXT: Running pass: LoopRotatePass on loop %bb in function f
; MSSA-NEXT: Folding loop latch bb4 into bb
; MSSA-NEXT: Invalidating analysis: PostDominatorTreeAnalysis on f
; MSSA-NEXT: Running pass: ADCEPass on f
; MSSA-NEXT: Running analysis: PostDominatorTreeAnalysis on f

; CHECK-LABEL: define i8 @f() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %bb
; CHECK:       bb:                                               ; preds = %bb, %entry
; CHECK-NEXT:    %mode.0 = phi i8 [ 0, %entry ], [ %indvar.next, %bb ]
; CHECK-NEXT:    %tmp5 = icmp eq i8 %mode.0, 1
; CHECK-NEXT:    %indvar.next = add i8 %mode.0, 1
; CHECK-NEXT:    br i1 %tmp5, label %bb5, label %bb
; CHECK:       bb5:                                              ; preds = %bb
; CHECK-NEXT:    tail call void @raise_exception() #0
; CHECK-NEXT:    unreachable
; CHECK-NEXT:  }
; CHECK:       ; Function Attrs: noreturn
; CHECK:       declare void @raise_exception() #0
; CHECK:       attributes #0 = { noreturn }

; MSSA-LABEL: define i8 @f() {
; MSSA-NEXT:  entry:
; MSSA-NEXT:    br label %bb
; MSSA:       bb:                                               ; preds = %bb, %entry
; MSSA-NEXT:    %mode.0 = phi i8 [ 0, %entry ], [ %indvar.next, %bb ]
; MSSA-NEXT:    %tmp5 = icmp eq i8 %mode.0, 1
; MSSA-NEXT:    %indvar.next = add i8 %mode.0, 1
; MSSA-NEXT:    br i1 %tmp5, label %bb5, label %bb
; MSSA:       bb5:                                              ; preds = %bb
; MSSA-NEXT:    tail call void @raise_exception() #0
; MSSA-NEXT:    unreachable
; MSSA-NEXT:  }
; MSSA:       ; Function Attrs: noreturn
; MSSA:       declare void @raise_exception() #0
; MSSA:       attributes #0 = { noreturn }

define i8 @f() {
entry:
  br label %bb

bb:                                               ; preds = %bb4, %entry
  %mode.0 = phi i8 [ 0, %entry ], [ %indvar.next, %bb4 ]
  %tmp5 = icmp eq i8 %mode.0, 1
  br i1 %tmp5, label %bb5, label %bb4

bb4:                                              ; preds = %bb2
  %indvar.next = add i8 %mode.0, 1
  br label %bb

bb5:                                              ; preds = %bb2
  tail call void @raise_exception() #0
  unreachable
}

; Function Attrs: noreturn
declare void @raise_exception() #0

attributes #0 = { noreturn }
