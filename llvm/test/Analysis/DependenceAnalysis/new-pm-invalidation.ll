; RUN: opt < %s -passes='require<da>,invalidate<scalar-evolution>,print<da>'   \
; RUN:   -disable-output -debug-pass-manager 2>&1 | FileCheck %s

; CHECK: Running analysis: DependenceAnalysis on test_no_noalias
; CHECK: Running analysis: ScalarEvolutionAnalysis on test_no_noalias
; CHECK: Invalidating analysis: ScalarEvolutionAnalysis on test_no_noalias
; CHECK: Invalidating analysis: DependenceAnalysis on test_no_noalias
; CHECK: Running analysis: DependenceAnalysis on test_no_noalias
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
define void @test_no_noalias(ptr %A, ptr %B) {
  store i32 1, ptr %A
  store i32 2, ptr %B
  ret void
}
