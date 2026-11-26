; RUN: opt -disable-output -debug-pass-manager -verify-analysis-invalidation=false -passes='function(require<domtree>),globaldce,function(require<domtree>)' %s 2>&1 | FileCheck %s

; Deleting @dead should only clear its own analyses, and not invalidate the
; analyses of @live.

; CHECK: Running analysis: DominatorTreeAnalysis on live{{$}}
; CHECK: Running analysis: DominatorTreeAnalysis on dead{{$}}
; CHECK-NEXT: Running pass: GlobalDCEPass on [module]{{$}}
; CHECK-NEXT: Clearing all analysis results for: dead{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on live (1 instruction){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

define void @live() {
  ret void
}

define internal void @dead() {
  ret void
}
