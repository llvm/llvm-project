; RUN: opt -passes='function(require<domtree>),attributor-light,function(require<domtree>,no-op-function)' -debug-pass-manager -disable-output %s 2>&1 | FileCheck %s

; attributor-light does not add or remove functions, so it must not clear the
; function analyses.

; CHECK: Running analysis: DominatorTreeAnalysis on f
; CHECK: Running pass: AttributorLightPass on [module]
; CHECK-NEXT: Running pass: RequireAnalysisPass<llvm::DominatorTreeAnalysis, llvm::Function, llvm::AnalysisManager<Function>> on f
; CHECK-NEXT: Running pass: NoOpFunctionPass on f

define void @f() {
  ret void
}
