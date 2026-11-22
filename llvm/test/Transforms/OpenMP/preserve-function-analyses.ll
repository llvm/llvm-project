; RUN: opt -disable-output -debug-pass-manager -passes='function(require<domtree>,require<target-lib-info>,require<target-ir>),openmp-opt,function(require<domtree>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s
; REQUIRES: amdgpu-registered-target

; Moving the shared allocation in @foo to the stack should not invalidate
; TargetLibraryInfo or TargetTransformInfo. The analyses of the deleted @dead
; should be cleared.

; CHECK: Running pass: OpenMPOptPass on [module]{{$}}
; CHECK-NEXT: Running analysis: LoopAnalysis on foo{{$}}
; CHECK-NEXT: Running analysis: OptimizationRemarkEmitterAnalysis on foo{{$}}
; CHECK-NEXT: Clearing all analysis results for: dead{{$}}
; CHECK-NEXT: Invalidating analysis: DominatorTreeAnalysis on foo{{$}}
; CHECK-NEXT: Invalidating analysis: LoopAnalysis on foo{{$}}
; CHECK-NEXT: Invalidating analysis: DominatorTreeAnalysis on other{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on foo (4 instructions){{$}}
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on foo{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on foo (4 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on foo (4 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on other{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

target triple = "amdgpu9-amd-amdhsa"

define internal void @dead() {
  ret void
}

define void @foo() {
  %x = call ptr @__kmpc_alloc_shared(i64 4)
  call void @use(ptr %x)
  call void @__kmpc_free_shared(ptr %x, i64 4)
  ret void
}

define void @other() {
  ret void
}

declare void @use(ptr nocapture)
declare ptr @__kmpc_alloc_shared(i64)
declare void @__kmpc_free_shared(ptr, i64)

!llvm.module.flags = !{!0, !1}

!0 = !{i32 7, !"openmp", i32 50}
!1 = !{i32 7, !"openmp-device", i32 50}
