; RUN: opt -disable-output -debug-pass-manager -passes='inferattrs,function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>),alloc-token<mode=increment>,function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s

; Rewriting the allocation in @f should only invalidate the non-CFG analyses of
; @f, and not the analyses of @other.

; CHECK: Running pass: AllocTokenPass on [module]{{$}}
; CHECK-NEXT: Running analysis: OptimizationRemarkEmitterAnalysis on f{{$}}
; CHECK-NEXT: Invalidating analysis: LazyValueAnalysis on f{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on f (2 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on f (2 instructions){{$}}
; CHECK-NEXT: Running analysis: LazyValueAnalysis on f{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on f (2 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on f (2 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

define ptr @f() sanitize_alloc_token {
  %p = call ptr @malloc(i64 64)
  ret ptr %p
}

define void @other() {
  ret void
}

declare ptr @malloc(i64)
