; RUN: opt -disable-output -debug-pass-manager -passes='function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>),simplify-type-tests,function(require<domtree>,require<lazy-value-info>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s

; Simplifying the type test in @cmp should only invalidate its non-CFG
; analyses. Refining the operand in @sub does not invalidate anything.

; CHECK: Running pass: SimplifyTypeTestsPass on [module]{{$}}
; CHECK-NEXT: Invalidating analysis: LazyValueAnalysis on cmp{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on _Z2fpv.cfi (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on _Z2fpv.cfi (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on _Z2fpv.cfi (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on _Z2fpv.cfi (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on cmp (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on cmp (1 instruction){{$}}
; CHECK-NEXT: Running analysis: LazyValueAnalysis on cmp{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on cmp (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on cmp (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on sub (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on sub (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on sub (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on sub (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LazyValueAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

@__typeid__ZTSFvvE_global_addr = external hidden global [0 x i8], code_model "small"

define void @_Z2fpv.cfi() !type !0 {
  ret void
}

define i1 @cmp() {
  %c = icmp eq ptr @_Z2fpv, @__typeid__ZTSFvvE_global_addr
  ret i1 %c
}

define i64 @sub() {
  ret i64 sub (i64 ptrtoint (ptr @__typeid__ZTSFvvE_global_addr to i64), i64 ptrtoint (ptr @_Z2fpv to i64))
}

define void @other() {
  ret void
}

declare hidden void @_Z2fpv()

!0 = !{i64 0, !"_ZTSFvvE"}
