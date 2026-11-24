; RUN: opt -disable-output -debug-pass-manager -verify-analysis-invalidation=0 -passes='inferattrs,function(require<domtree>,require<target-lib-info>,require<target-ir>),strip-dead-prototypes,function(require<domtree>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s

; Deleting @dead should clear its analyses, and not invalidate the analyses of
; @f.

; CHECK: Running pass: StripDeadPrototypesPass on [module]{{$}}
; CHECK-NEXT: Clearing all analysis results for: dead{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

declare void @dead()

define void @f() {
  ret void
}
