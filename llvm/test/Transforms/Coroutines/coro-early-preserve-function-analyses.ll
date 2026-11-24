; RUN: split-file %s %t
; RUN: opt -disable-output -debug-pass-manager \
; RUN:   -passes='function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>),coro-early,function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>)' \
; RUN:   %t/precise.ll 2>&1 | FileCheck --check-prefix=PRECISE %s
; RUN: opt -disable-output -debug-pass-manager \
; RUN:   -passes='function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>),coro-early,function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>)' \
; RUN:   %t/noalias.ll 2>&1 | FileCheck --check-prefix=NOALIAS %s

; Lowering should only invalidate the non-CFG analyses of the lowered
; functions. Removing noalias may affect the analyses of callers, so the non-CFG
; analyses of all functions are invalidated.

; PRECISE: Running pass: CoroEarlyPass on [module]{{$}}
; PRECISE-NEXT: Invalidating analysis: LazyValueAnalysis on lowered{{$}}
; PRECISE-NEXT: Invalidating analysis: LazyValueAnalysis on suspends_no_noalias{{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on lowered (3 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on lowered (3 instructions){{$}}
; PRECISE-NEXT: Running analysis: LazyValueAnalysis on lowered{{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on lowered (3 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on lowered (3 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on caller (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on caller (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on caller (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on caller (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on suspends_no_noalias (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on suspends_no_noalias (2 instructions){{$}}
; PRECISE-NEXT: Running analysis: LazyValueAnalysis on suspends_no_noalias{{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on suspends_no_noalias (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on suspends_no_noalias (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on suspends_no_noalias_caller (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on suspends_no_noalias_caller (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on suspends_no_noalias_caller (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on suspends_no_noalias_caller (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on unchanged (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on unchanged (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on unchanged (2 instructions){{$}}
; PRECISE-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on unchanged (2 instructions){{$}}
; PRECISE-NEXT: Running pass: VerifierPass on [module]{{$}}

; NOALIAS: Running pass: CoroEarlyPass on [module]{{$}}
; NOALIAS-NEXT: Invalidating analysis: LazyValueAnalysis on suspends{{$}}
; NOALIAS-NEXT: Invalidating analysis: LazyValueAnalysis on suspends_caller{{$}}
; NOALIAS-NEXT: Invalidating analysis: LazyValueAnalysis on unchanged{{$}}
; NOALIAS-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on suspends (2 instructions){{$}}
; NOALIAS-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on suspends (2 instructions){{$}}
; NOALIAS-NEXT: Running analysis: LazyValueAnalysis on suspends{{$}}
; NOALIAS-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on suspends (2 instructions){{$}}
; NOALIAS-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on suspends (2 instructions){{$}}
; NOALIAS-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on suspends_caller (2 instructions){{$}}
; NOALIAS-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on suspends_caller (2 instructions){{$}}
; NOALIAS-NEXT: Running analysis: LazyValueAnalysis on suspends_caller{{$}}
; NOALIAS-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on suspends_caller (2 instructions){{$}}
; NOALIAS-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on suspends_caller (2 instructions){{$}}
; NOALIAS-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on unchanged (2 instructions){{$}}
; NOALIAS-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on unchanged (2 instructions){{$}}
; NOALIAS-NEXT: Running analysis: LazyValueAnalysis on unchanged{{$}}
; NOALIAS-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on unchanged (2 instructions){{$}}
; NOALIAS-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on unchanged (2 instructions){{$}}
; NOALIAS-NEXT: Running pass: VerifierPass on [module]{{$}}

;--- precise.ll
define void @lowered(ptr %hdl) {
  call void @llvm.coro.resume(ptr %hdl)
  ret void
}

define void @caller(ptr %hdl) {
  call void @lowered(ptr %hdl)
  ret void
}

define void @suspends_no_noalias(ptr %p) {
  %s = call i8 @llvm.coro.suspend(token none, i1 true)
  ret void
}

define void @suspends_no_noalias_caller(ptr %p) {
  call void @suspends_no_noalias(ptr %p)
  ret void
}

define void @unchanged(ptr %p) {
  %s = call i8 @llvm.coro.suspend(token none, i1 false)
  ret void
}

declare void @llvm.coro.resume(ptr)
declare i8 @llvm.coro.suspend(token, i1)

;--- noalias.ll
define void @suspends(ptr noalias %p) {
  %s = call i8 @llvm.coro.suspend(token none, i1 false)
  ret void
}

define void @suspends_caller(ptr %p) {
  call void @suspends(ptr %p)
  ret void
}

define void @unchanged(ptr %p) {
  %s = call i8 @llvm.coro.suspend(token none, i1 false)
  ret void
}

declare void @llvm.coro.resume(ptr)
declare i8 @llvm.coro.suspend(token, i1)
