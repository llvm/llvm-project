; Test that access-info gets invalidated when the analyses it depends on are
; invalidated.

; This is a reproducer for https://github.com/llvm/llvm-project/issues/61324.
; We want to see that LoopAccessAnalysis+AAManger is being updated at the end,
; instead of crashing when using a stale AAManager.
;
; RUN: opt < %s -disable-output -debug-pass-manager -passes='function(require<access-info>,invalidate<aa>),print<access-info>' 2>&1 | FileCheck %s --check-prefix=CHECK-AA
;
; CHECK-AA: Running pass: RequireAnalysisPass
; CHECK-AA-NEXT: Running analysis: LoopAccessAnalysis on foo
; CHECK-AA: Running pass: InvalidateAnalysisPass
; CHECK-AA-NEXT: Invalidating analysis: AAManager on foo
; CHECK-AA-NEXT: Invalidating analysis: LoopAccessAnalysis on foo
; CHECK-AA-NEXT: Running pass: LoopAccessInfoPrinterPass on foo
; CHECK-AA-NEXT: Running analysis: LoopAccessAnalysis on foo
; CHECK-AA-NEXT: Running analysis: AAManager on foo


; Verify that an explicit invalidate request for access-info result in an
; invalidation.
;
; RUN: opt < %s -disable-output -debug-pass-manager -passes='function(require<access-info>,invalidate<access-info>)' 2>&1 | FileCheck %s --check-prefix=CHECK-INV-AA
;
; CHECK-INV-AA: Running pass: RequireAnalysisPass
; CHECK-INV-AA-NEXT: Running analysis: LoopAccessAnalysis on foo
; CHECK-INV-AA: Running pass: InvalidateAnalysisPass
; CHECK-INV-AA-NEXT: Invalidating analysis: LoopAccessAnalysis on foo


; Invalidation of scalar-evolution should transitively invalidate access-info.
;
; RUN: opt < %s -disable-output -debug-pass-manager -passes='function(require<access-info>,invalidate<scalar-evolution>)' 2>&1 | FileCheck %s --check-prefix=CHECK-SCEV
;
; CHECK-SCEV: Running pass: RequireAnalysisPass
; CHECK-SCEV-NEXT: Running analysis: LoopAccessAnalysis on foo
; CHECK-SCEV: Running pass: InvalidateAnalysisPass
; CHECK-SCEV-NEXT: Invalidating analysis: ScalarEvolutionAnalysis on foo
; CHECK-SCEV-NEXT: Invalidating analysis: LoopAccessAnalysis on foo


; Invalidation of domtree should transitively invalidate access-info.
;
; RUN: opt < %s -disable-output -debug-pass-manager -passes='function(require<access-info>,invalidate<domtree>)' 2>&1 | FileCheck %s --check-prefix=CHECK-DT
;
; CHECK-DT: Running pass: RequireAnalysisPass
; CHECK-DT-NEXT: Running analysis: LoopAccessAnalysis on foo
; CHECK-DT: Running pass: InvalidateAnalysisPass
; CHECK-DT-NEXT: Invalidating analysis: DominatorTreeAnalysis on foo
; CHECK-DT: Invalidating analysis: LoopAccessAnalysis on foo


define void @foo(ptr nocapture writeonly %a, ptr nocapture writeonly %b) memory(argmem: write) {
entry:
  br label %for.cond1

for.cond1:
  store i16 0, ptr %b, align 1
  store i16 0, ptr %a, align 1
  br i1 true, label %for.end6, label %for.cond1

for.end6:
  ret void
}
