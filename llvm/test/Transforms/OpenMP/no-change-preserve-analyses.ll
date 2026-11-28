; RUN: opt -disable-output -debug-pass-manager -passes='function(require<domtree>,require<lazy-value-info>),openmp-opt,function(require<domtree>,require<lazy-value-info>)' %s 2>&1 | FileCheck %s
; REQUIRES: amdgpu-registered-target

; OpenMPOpt does not change this device module, so no analyses should be
; invalidated.

; CHECK: Running pass: OpenMPOptPass on [module]{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on f (1 instruction){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

target triple = "amdgpu9-amd-amdhsa"

define void @f() {
  ret void
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 7, !"openmp", i32 50}
!1 = !{i32 7, !"openmp-device", i32 50}
