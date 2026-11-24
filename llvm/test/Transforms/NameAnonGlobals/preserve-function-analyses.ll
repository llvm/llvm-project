; RUN: opt -disable-output -debug-pass-manager -verify-analysis-invalidation=false -passes='function(require<domtree>,require<lazy-value-info>),name-anon-globals,function(require<domtree>,require<lazy-value-info>)' %s 2>&1 | FileCheck %s

; Naming @0 should not invalidate the analyses of @f.

; CHECK: Running pass: NameAnonGlobalPass on [module]{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

@0 = global i32 0

define ptr @f() {
  ret ptr @0
}
