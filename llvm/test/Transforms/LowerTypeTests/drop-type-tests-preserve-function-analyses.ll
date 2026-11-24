; RUN: opt -disable-output -debug-pass-manager -passes='function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>),drop-type-tests,function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck -check-prefix=ASSUME %s
; RUN: opt -disable-output -debug-pass-manager -passes='function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>),drop-type-tests<all>,function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck -check-prefix=ALL %s

; Dropping type tests should only invalidate the non-CFG analyses of the
; modified functions, and not the analyses of @other.

; ASSUME: Running pass: DropTypeTestsPass on [module]{{$}}
; ASSUME-NEXT: Invalidating analysis: LazyValueAnalysis on f{{$}}
; ASSUME-NEXT: Invalidating analysis: LazyValueAnalysis on g{{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on f (1 instruction){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on f (1 instruction){{$}}
; ASSUME-NEXT: Running analysis: LazyValueAnalysis on f{{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on f (1 instruction){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on f (1 instruction){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on g (1 instruction){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on g (1 instruction){{$}}
; ASSUME-NEXT: Running analysis: LazyValueAnalysis on g{{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on g (1 instruction){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on g (1 instruction){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on h (3 instructions){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on h (3 instructions){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on h (3 instructions){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on h (3 instructions){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on i (3 instructions){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on i (3 instructions){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on i (3 instructions){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on i (3 instructions){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on other (1 instruction){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on other (1 instruction){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on other (1 instruction){{$}}
; ASSUME-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on other (1 instruction){{$}}
; ASSUME-NEXT: Running pass: VerifierPass on [module]{{$}}

; ALL: Running pass: DropTypeTestsPass on [module]{{$}}
; ALL-NEXT: Invalidating analysis: LazyValueAnalysis on f{{$}}
; ALL-NEXT: Invalidating analysis: LazyValueAnalysis on h{{$}}
; ALL-NEXT: Invalidating analysis: LazyValueAnalysis on i{{$}}
; ALL-NEXT: Invalidating analysis: LazyValueAnalysis on g{{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on f (1 instruction){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on f (1 instruction){{$}}
; ALL-NEXT: Running analysis: LazyValueAnalysis on f{{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on f (1 instruction){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on f (1 instruction){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on g (1 instruction){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on g (1 instruction){{$}}
; ALL-NEXT: Running analysis: LazyValueAnalysis on g{{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on g (1 instruction){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on g (1 instruction){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on h (5 instructions){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on h (5 instructions){{$}}
; ALL-NEXT: Running analysis: LazyValueAnalysis on h{{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on h (5 instructions){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on h (5 instructions){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on i (4 instructions){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on i (4 instructions){{$}}
; ALL-NEXT: Running analysis: LazyValueAnalysis on i{{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on i (4 instructions){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on i (4 instructions){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on other (1 instruction){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on other (1 instruction){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on other (1 instruction){{$}}
; ALL-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on other (1 instruction){{$}}
; ALL-NEXT: Running pass: VerifierPass on [module]{{$}}

define void @f(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid")
  call void @llvm.assume(i1 %x)
  ret void
}

define void @g(ptr %p) {
  %x = call i1 @llvm.public.type.test(ptr %p, metadata !"typeid")
  call void @llvm.assume(i1 %x)
  ret void
}

define ptr @h(ptr %p) {
  %r = call { ptr, i1 } @llvm.type.checked.load(ptr %p, i32 0, metadata !"typeid")
  %v = extractvalue { ptr, i1 } %r, 0
  ret ptr %v
}

define ptr @i(ptr %p) {
  %r = call { ptr, i1 } @llvm.type.checked.load.relative(ptr %p, i32 0, metadata !"typeid")
  %v = extractvalue { ptr, i1 } %r, 0
  ret ptr %v
}

define void @other() {
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare i1 @llvm.public.type.test(ptr, metadata)
declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)
declare { ptr, i1 } @llvm.type.checked.load.relative(ptr, i32, metadata)
declare void @llvm.assume(i1)
