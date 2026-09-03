; RUN: opt -passes='require<memoryssa>,loop-versioning,require<memoryssa>,require<domtree>,require<loops>,require<scalar-evolution>' \
; RUN:     -verify-analysis-invalidation=false -debug-pass-manager -disable-output %s 2>&1 | FileCheck %s

; Verify perserved analyses.
; RUN: opt -passes='require<memoryssa>,loop-versioning,verify<domtree>,verify<loops>,verify<memoryssa>' \
; RUN:     -disable-output %s

; RUN: opt -passes='loop-versioning,verify<domtree>,verify<loops>' -disable-output %s

; CHECK:      Running pass: LoopVersioningPass on f
; CHECK:      Invalidating analysis: ScalarEvolutionAnalysis on f
; CHECK-NEXT: Invalidating analysis: LoopAccessAnalysis on f
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}MemorySSAAnalysis
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}LoopAnalysis
; CHECK-NEXT: Running pass: RequireAnalysisPass<{{.*}}ScalarEvolutionAnalysis
; CHECK-NEXT: Running analysis: ScalarEvolutionAnalysis on f

define void @f(ptr %a, ptr %b, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %l = load i32, ptr %gep.a, align 4
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %l, ptr %gep.b, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
