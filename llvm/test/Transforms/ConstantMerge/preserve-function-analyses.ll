; RUN: opt -disable-output -debug-pass-manager -verify-analysis-invalidation=false -passes='function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>),constmerge,function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s

; Merging @a into @b should not invalidate the CFG analyses, TargetLibraryInfo
; or TargetTransformInfo of @f.

; CHECK: Running pass: ConstantMergePass on [module]{{$}}
; CHECK-NEXT: Invalidating analysis: LazyValueAnalysis on f{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on f (4 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on f (4 instructions){{$}}
; CHECK-NEXT: Running analysis: LazyValueAnalysis on f{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on f (4 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on f (4 instructions){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

@a = internal unnamed_addr constant i32 1
@b = internal unnamed_addr constant i32 1

define i32 @f() {
  %x = load i32, ptr @a
  %y = load i32, ptr @b
  %sum = add i32 %x, %y
  ret i32 %sum
}
