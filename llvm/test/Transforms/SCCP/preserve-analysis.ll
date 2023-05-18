; RUN: opt < %s -debug-pass-manager -passes='loop-vectorize,sccp,loop-vectorize' 2>&1 -S | FileCheck --check-prefix=NEW-PM %s

; Check that DT is preserved by SCCP by running it between 2
; loop-vectorize runs.

; NEW-PM-DAG: Running analysis: LoopAnalysis on test
; NEW-PM-DAG: Running analysis: DominatorTreeAnalysis on test
; NEW-PM: Running pass: SCCPPass on test
; NEW-PM: Running analysis: TargetLibraryAnalysis on test
; NEW-PM: Running analysis: LoopAnalysis on test
; NEW-PM-NOT: Running analysis: DominatorTreeAnalysis on test
; NEW-PM-NOT: Running analysis: AssumptionAnalysis on test
; NEW-PM-NOT: Running analysis: TargetLibraryAnalysis on test
; NEW-PM-NOT: Running analysis: TargetIRAnalysis on test


define i32 @test() {
entry:
  %res = add i32 1, 10
  ret i32 %res
}
