; RUN: opt -disable-output -debug-pass-manager -passes='function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>),coro-cleanup,function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s

; Replacing the size in @target_fp may affect any function, so the non-CFG
; analyses of @reader should also be invalidated, other than TargetLibraryInfo
; and TargetTransformInfo.

; CHECK: Running pass: CoroCleanupPass on [module]{{$}}
; CHECK-NEXT: Invalidating analysis: DominatorTreeAnalysis on replace{{$}}
; CHECK-NEXT: Invalidating analysis: LazyValueAnalysis on replace{{$}}
; CHECK-NEXT: Running pass: SimplifyCFGPass on replace (1 instruction){{$}}
; CHECK-NEXT: Invalidating analysis: LazyValueAnalysis on reader{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on replace (1 instruction){{$}}
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on replace{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on replace (1 instruction){{$}}
; CHECK-NEXT: Running analysis: LazyValueAnalysis on replace{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on replace (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on replace (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on reader (3 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on reader (3 instructions){{$}}
; CHECK-NEXT: Running analysis: LazyValueAnalysis on reader{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on reader (3 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on reader (3 instructions){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

@target_fp = constant <{ i32, i32 }> <{ i32 0, i32 8 }>
@source_fp = constant <{ i32, i32 }> <{ i32 0, i32 16 }>

define void @replace() {
  call void @llvm.coro.async.size.replace(ptr @target_fp, ptr @source_fp)
  ret void
}

define i32 @reader() {
  %size.addr = getelementptr inbounds <{ i32, i32 }>, ptr @target_fp, i32 0, i32 1
  %size = load i32, ptr %size.addr
  ret i32 %size
}

declare void @llvm.coro.async.size.replace(ptr, ptr)
