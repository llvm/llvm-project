;RUN: opt %s -passes='adce,loop(rotate),adce' -S -debug-pass-manager -debug-only=loop-rotate 2>&1 | FileCheck %s
;REQUIRES: asserts

; This test is to make sure we invalidate the post dominator pass after loop rotate simplifies the loop latch.
; The adce passes are here to make sure post dominator analysis is required.

; CHECK: Starting llvm::Function pass manager run.
; CHECK-NEXT: Running pass: ADCEPass on f
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on f
; CHECK-NEXT: Running analysis: PostDominatorTreeAnalysis on f
; CHECK-NEXT: Running pass: FunctionToLoopPassAdaptor{{.*}} on f
; CHECK-NEXT: Running analysis: LoopAnalysis on f
; CHECK-NEXT: Running analysis: AssumptionAnalysis on f
; CHECK-NEXT: Running analysis: AAManager on f
; CHECK-NEXT: Running analysis: TargetLibraryAnalysis on f
; CHECK-NEXT: Running analysis: ScalarEvolutionAnalysis on f
; CHECK-NEXT: Running analysis: TargetIRAnalysis on f
; CHECK-NEXT: Running analysis: InnerAnalysisManagerProxy{{.*}} on f
; CHECK-NEXT: Starting Loop pass manager run.
; CHECK-NEXT: Running pass: LoopRotatePass on Loop at depth 1 containing: %bb<header><exiting>,%bb4<latch>
; CHECK-NEXT: Folding loop latch bb4 into bb
; CHECK-NEXT: Invalidating all non-preserved analyses for: bb
; CHECK-NEXT: Finished Loop pass manager run.
; CHECK-NEXT: Invalidating all non-preserved analyses for: f
; CHECK-NEXT: Invalidating analysis: PostDominatorTreeAnalysis on f
; CHECK-NEXT: Running pass: ADCEPass on f
; CHECK-NEXT: Running analysis: PostDominatorTreeAnalysis on f
; CHECK-NEXT: Finished llvm::Function pass manager run.

; CHECK-LABEL: define i8 @f() {
; CHECK-NEXT : entry:
; CHECK-NEXT :   br label %bb
; CHECK-NEXT :
; CHECK-NEXT : bb:                                               ; preds = %bb, %entry
; CHECK-NEXT :   %mode.0 = phi i8 [ 0, %entry ], [ %indvar.next, %bb ]
; CHECK-NEXT :   %tmp5 = icmp eq i8 %mode.0, 1
; CHECK-NEXT :   %indvar.next = add i8 %mode.0, 1
; CHECK-NEXT :   br i1 %tmp5, label %bb5, label %bb
; CHECK-NEXT :
; CHECK-NEXT : bb5:                                              ; preds = %bb
; CHECK-NEXT :   tail call void @raise_exception() #0
; CHECK-NEXT :   unreachable
; CHECK-NEXT : }
; CHECK-NEXT :
; CHECK-NEXT : ; Function Attrs: noreturn
; CHECK-NEXT : declare void @raise_exception() #0
; CHECK-NEXT :
; CHECK-NEXT : attributes #0 = { noreturn }

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
