; RUN: opt -disable-output -debug-pass-manager -passes='require<lcg>,function(require<domtree>,require<target-lib-info>,require<target-ir>),inferattrs,function(require<domtree>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s

; Inferring attributes on @strlen should not invalidate the call graph, or the
; CFG analyses, TargetLibraryInfo or TargetTransformInfo of @f.

; CHECK: Running analysis: LazyCallGraphAnalysis on [module]{{$}}
; CHECK: Running analysis: TargetLibraryAnalysis on f{{$}}
; CHECK: Running analysis: DominatorTreeAnalysis on f{{$}}
; CHECK: Running analysis: TargetIRAnalysis on f{{$}}
; CHECK-NEXT: Running pass: InferFunctionAttrsPass on [module]{{$}}
; CHECK-NEXT: Running analysis: TargetLibraryAnalysis on strlen{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on f (2 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on f (2 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on f (2 instructions){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

declare i64 @strlen(ptr)

define i64 @f(ptr %p) {
  %len = call i64 @strlen(ptr %p)
  ret i64 %len
}
