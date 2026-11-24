; RUN: opt -disable-output -debug-pass-manager -pgo-kind=pgo-instr-use-pipeline -pgo-cold-func-opt=minsize -passes='function(require<domtree>,require<target-lib-info>,require<target-ir>),pgo-force-function-attrs,function(require<domtree>,require<target-lib-info>,require<target-ir>)' %s 2>&1 | FileCheck %s

; Adding minsize to @cold should clear all of its analyses, since
; TargetTransformInfo may depend on minsize, and not affect @other.

; CHECK: Running pass: PGOForceFunctionAttrsPass on [module]{{$}}
; CHECK-NEXT: Running analysis: ProfileSummaryAnalysis on [module]{{$}}
; CHECK-NEXT: Clearing all analysis results for: cold{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on cold (1 instruction){{$}}
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on cold{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on cold (1 instruction){{$}}
; CHECK-NEXT: Running analysis: TargetLibraryAnalysis on cold{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on cold (1 instruction){{$}}
; CHECK-NEXT: Running analysis: TargetIRAnalysis on cold{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetLibraryAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}TargetIRAnalysis{{.*}}> on other (1 instruction){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

define void @cold() cold {
  ret void
}

define void @other() {
  ret void
}
