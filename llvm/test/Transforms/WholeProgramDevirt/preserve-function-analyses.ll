; RUN: opt -disable-output -debug-pass-manager -whole-program-visibility -passes='function(require<domtree>,require<target-lib-info>,require<target-ir>),wholeprogramdevirt,function(require<domtree>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s

; Devirtualizing the call in @call should not invalidate TargetLibraryInfo or
; TargetTransformInfo.

; CHECK: Running pass: WholeProgramDevirtPass on [module]{{$}}
; CHECK-NEXT: Invalidating analysis: DominatorTreeAnalysis on vf{{$}}
; CHECK-NEXT: Invalidating analysis: DominatorTreeAnalysis on call{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on vf (1 instruction){{$}}
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on vf{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on vf (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on vf (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on call (6 instructions){{$}}
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on call{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on call (6 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on call (6 instructions){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

@vt = constant [1 x ptr] [ptr @vf], !type !0

define void @vf(ptr %this) {
  ret void
}

define void @call(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  call void %fptr(ptr %obj)
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}
