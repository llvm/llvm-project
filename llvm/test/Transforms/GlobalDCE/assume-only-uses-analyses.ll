; RUN: opt -disable-output -debug-pass-manager -passes='function(require<domtree>,require<demanded-bits>),globaldce<drop-assumes>,function(require<domtree>,require<demanded-bits>)' %s 2>&1 | FileCheck %s

; Dropping an assumption invalidates the analyses of its function, except those
; of the CFG, and leaves the other functions alone.

; CHECK:      Running pass: GlobalDCEPass on [module]{{$}}
; CHECK-NEXT: Invalidating analysis: DemandedBitsAnalysis on changed{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on changed (1 instruction){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DemandedBitsAnalysis{{.*}}> on changed (1 instruction){{$}}
; CHECK-NEXT: Running analysis: DemandedBitsAnalysis on changed{{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis{{.*}}> on unchanged (3 instructions){{$}}
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DemandedBitsAnalysis{{.*}}> on unchanged (3 instructions){{$}}
; CHECK-NEXT: Running pass: VerifierPass on [module]{{$}}

@str = private constant [4 x i8] c"STR\00"

define void @changed(ptr %p) {
  %cmp = icmp ne ptr %p, @str
  call void @llvm.assume(i1 %cmp)
  ret void
}

define void @unchanged(ptr %p) {
  %cmp = icmp ne ptr %p, null
  call void @llvm.assume(i1 %cmp)
  ret void
}

declare void @llvm.assume(i1)
