; RUN: opt -mtriple=wasm32-unknown-unknown -expand-variadics-override=optimize -disable-output -debug-pass-manager -passes='inferattrs,function(require<domtree>,require<target-lib-info>,require<target-ir>),expand-variadics,function(require<domtree>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s
; REQUIRES: webassembly-registered-target

; The analyses of the erased functions should be cleared, and the other
; functions should keep TargetLibraryInfo and TargetTransformInfo.

; CHECK: Running pass: ExpandVariadicsPass on [module]{{$}}
; CHECK-NEXT: Clearing all analysis results for: {{$}}
; CHECK-NEXT: Clearing all analysis results for: llvm.va_end.p0{{$}}
; CHECK-NEXT: Invalidating analysis: DominatorTreeAnalysis on caller{{$}}
; CHECK-NEXT: Invalidating analysis: DominatorTreeAnalysis on other{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on vararg (7 instructions){{$}}
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on vararg{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on vararg (7 instructions){{$}}
; CHECK-NEXT: Running analysis: TargetLibraryAnalysis on vararg{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on vararg (7 instructions){{$}}
; CHECK-NEXT: Running analysis: TargetIRAnalysis on vararg{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on vararg.valist (3 instructions){{$}}
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on vararg.valist{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on vararg.valist (3 instructions){{$}}
; CHECK-NEXT: Running analysis: TargetLibraryAnalysis on vararg.valist{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on vararg.valist (3 instructions){{$}}
; CHECK-NEXT: Running analysis: TargetIRAnalysis on vararg.valist{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on caller (7 instructions){{$}}
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on caller{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on caller (7 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on caller (7 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on other{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

define void @vararg(...) {
  %va = alloca ptr
  call void @llvm.va_start.p0(ptr %va)
  call void @llvm.va_end.p0(ptr %va)
  ret void
}

define void @caller() {
  call void (...) @vararg(i32 1)
  ret void
}

define void @other() {
  ret void
}

declare void @llvm.va_start.p0(ptr)
declare void @llvm.va_end.p0(ptr)
