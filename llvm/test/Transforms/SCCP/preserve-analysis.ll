; RUN: opt < %s -debug-pass=Structure -globals-aa -loop-vectorize -sccp -loop-vectorize -globals-aa -enable-new-pm=0 2>&1 -S | FileCheck %s
; RUN: opt < %s -debug-pass-manager -passes='loop-vectorize,sccp,loop-vectorize' 2>&1 -S | FileCheck --check-prefix=NEW-PM %s

; Check CFG-only analysis are preserved by SCCP by running it between 2
; loop-vectorize runs.

; CHECK: Globals Alias Analysis
; CHECK: Dominator Tree Construction
; CHECK: Natural Loop Information
; CHECK: Sparse Conditional Constant Propagation
; CHECK: Post-Dominator Tree Construction
; CHECK-NOT: Dominator Tree Construction
; CHECK-NOT: Natural Loop Information
; CHECK-NOT: Globals Alias Analysis
; CHECK: Loop Vectorization

; NEW-PM-DAG: Running analysis: LoopAnalysis on test
; NEW-PM-DAG: Running analysis: DominatorTreeAnalysis on test
; NEW-PM-DAG: Running analysis: AssumptionAnalysis on test
; NEW-PM-DAG: Running analysis: TargetLibraryAnalysis on test
; NEW-PM-DAG: Running analysis: TargetIRAnalysis on test
; NEW-PM: Running pass: SCCPPass on test
; NEW-PM-NOT: Running analysis: LoopAnalysis on test
; NEW-PM-NOT: Running analysis: DominatorTreeAnalysis on test
; NEW-PM-NOT: Running analysis: AssumptionAnalysis on test
; NEW-PM-NOT: Running analysis: TargetLibraryAnalysis on test
; NEW-PM-NOT: Running analysis: TargetIRAnalysis on test


define i32 @test() {
entry:
  %res = add i32 1, 10
  ret i32 %res
}
