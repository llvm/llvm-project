; RUN: opt -disable-output -debug-pass-manager -passes='function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>),norecurse-lto-inference,function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s

; Inferring norecurse on @callee and @caller should only invalidate the non-CFG
; analyses of those functions, other than TargetLibraryInfo and
; TargetTransformInfo, and not the analyses of @other.

; CHECK: Running pass: NoRecurseLTOInferencePass on [module]{{$}}
; CHECK-NEXT: Running analysis: LazyCallGraphAnalysis on [module]{{$}}
; CHECK-NEXT: Invalidating analysis: LazyValueAnalysis on callee{{$}}
; CHECK-NEXT: Invalidating analysis: LazyValueAnalysis on caller{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on callee (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on callee (1 instruction){{$}}
; CHECK-NEXT: Running analysis: LazyValueAnalysis on callee{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on callee (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on callee (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on caller (2 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on caller (2 instructions){{$}}
; CHECK-NEXT: Running analysis: LazyValueAnalysis on caller{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on caller (2 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on caller (2 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

define internal void @callee() {
  ret void
}

define void @caller() {
  call void @callee()
  ret void
}

define void @other() norecurse {
  ret void
}
