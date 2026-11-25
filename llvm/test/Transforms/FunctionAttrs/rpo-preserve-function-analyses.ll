; RUN: opt -disable-output -debug-pass-manager -verify-analysis-invalidation=0 -passes='function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>),rpo-function-attrs,function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s

; Inferring norecurse on @callee and nofpclass on @fp_callee should only
; invalidate the non-CFG analyses of the callees and their callers, other than
; TargetLibraryInfo and TargetTransformInfo, and not the analyses of @other.

; CHECK: Running pass: ReversePostOrderFunctionAttrsPass on [module]{{$}}
; CHECK-NEXT: Running analysis: LazyCallGraphAnalysis on [module]{{$}}
; CHECK-NEXT: Invalidating analysis: LazyValueAnalysis on fp_callee{{$}}
; CHECK-NEXT: Invalidating analysis: LazyValueAnalysis on fp_caller{{$}}
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
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on fp_callee (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on fp_callee (1 instruction){{$}}
; CHECK-NEXT: Running analysis: LazyValueAnalysis on fp_callee{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on fp_callee (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on fp_callee (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on fp_caller (2 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on fp_caller (2 instructions){{$}}
; CHECK-NEXT: Running analysis: LazyValueAnalysis on fp_caller{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on fp_caller (2 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on fp_caller (2 instructions){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

define internal void @callee() {
  ret void
}

define void @caller() norecurse {
  call void @callee()
  ret void
}

define void @other() {
  ret void
}

define internal void @fp_callee(float %x) {
  ret void
}

define void @fp_caller() {
  call void @fp_callee(float 1.0)
  ret void
}
