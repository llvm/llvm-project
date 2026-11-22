; RUN: opt -disable-output -debug-pass-manager -passes='function(require<domtree>,require<lazy-value-info>),canonicalize-aliases,function(require<domtree>,require<lazy-value-info>)' %s 2>&1 | FileCheck %s

; Canonicalizing @a2 should not invalidate the analyses of @f.

; CHECK: Running pass: CanonicalizeAliasesPass on [module]{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

@g = global i32 0
@a1 = alias i32, ptr @g
@a2 = alias i32, ptr @a1

define ptr @f() {
  ret ptr @a2
}
