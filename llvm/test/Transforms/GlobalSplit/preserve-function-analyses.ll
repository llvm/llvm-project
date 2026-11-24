; RUN: opt -disable-output -debug-pass-manager -verify-analysis-invalidation=false -passes='function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>),globalsplit,function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s

; Splitting @global should not invalidate the CFG analyses, TargetLibraryInfo
; or TargetTransformInfo of the functions.

; CHECK: Running pass: GlobalSplitPass on [module]{{$}}
; CHECK-NEXT: Invalidating analysis: LazyValueAnalysis on f{{$}}
; CHECK-NEXT: Invalidating analysis: LazyValueAnalysis on test{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running analysis: LazyValueAnalysis on f{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on test (2 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on test (2 instructions){{$}}
; CHECK-NEXT: Running analysis: LazyValueAnalysis on test{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on test (2 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on test (2 instructions){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

@global = internal constant { [1 x ptr], [1 x ptr] } {
  [1 x ptr] [ptr @f],
  [1 x ptr] [ptr @test]
}, !type !0

define ptr @f() {
  ret ptr getelementptr inrange(0, 8) ({ [1 x ptr], [1 x ptr] }, ptr @global, i32 0, i32 1, i32 0)
}

define i1 @test(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid")
  ret i1 %x
}

declare i1 @llvm.type.test(ptr, metadata)

!0 = !{i32 8, !"typeid"}
