; RUN: opt -disable-output -debug-pass-manager -passes='function(require<domtree>,require<target-lib-info>,require<target-ir>),openmp-opt,function(require<domtree>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s

; Changing @foo should not invalidate its TargetLibraryInfo or
; TargetTransformInfo.

; CHECK: Running pass: OpenMPOptPass on [module]{{$}}
; CHECK-NEXT: Running analysis: LoopAnalysis on foo{{$}}
; CHECK-NEXT: Running analysis: OptimizationRemarkEmitterAnalysis on foo{{$}}
; CHECK-NEXT: Invalidating analysis: DominatorTreeAnalysis on foo{{$}}
; CHECK-NEXT: Invalidating analysis: LoopAnalysis on foo{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on foo (6 instructions){{$}}
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on foo{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on foo (6 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on foo (6 instructions){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

%struct.ConfigurationEnvironmentTy = type { i8, i8, i8, i32, i32, i32, i32, i32, i32 }
%struct.KernelEnvironmentTy = type { %struct.ConfigurationEnvironmentTy, ptr, ptr }

@foo_kernel_environment = local_unnamed_addr constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 0, i8 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 }, ptr null, ptr null }

define void @foo() "kernel" {
  %i = call i32 @__kmpc_target_init(ptr @foo_kernel_environment)
  %x = call ptr @__kmpc_alloc_shared(i64 4)
  call void @use(ptr %x)
  call void @__kmpc_free_shared(ptr %x)
  call void @__kmpc_target_deinit()
  ret void
}

declare void @use(ptr)
declare ptr @__kmpc_alloc_shared(i64)
declare void @__kmpc_free_shared(ptr)
declare i32 @__kmpc_target_init(ptr)
declare void @__kmpc_target_deinit()

!llvm.module.flags = !{!0, !1}

!0 = !{i32 7, !"openmp", i32 50}
!1 = !{i32 7, !"openmp-device", i32 50}
