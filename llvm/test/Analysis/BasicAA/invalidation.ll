; Test that the BasicAA analysis gets invalidated when its dependencies go
; away.
;
; Check DomTree specifically.
; RUN: opt -disable-output -disable-verify -debug-pass-manager %s 2>&1 \
; RUN:     -passes='require<aa>,invalidate<domtree>,aa-eval' -aa-pipeline='basic-aa' \
; RUN:     | FileCheck %s --check-prefix=CHECK-DT-INVALIDATE
; CHECK-DT-INVALIDATE: Running pass: RequireAnalysisPass
; CHECK-DT-INVALIDATE: Running analysis: BasicAA
; CHECK-DT-INVALIDATE: Running pass: InvalidateAnalysisPass
; CHECK-DT-INVALIDATE: Invalidating analysis: DominatorTreeAnalysis
; CHECK-DT-INVALIDATE: Invalidating analysis: BasicAA
; CHECK-DT-INVALIDATE: Running pass: AAEvaluator
; CHECK-DT-INVALIDATE: Running analysis: BasicAA
;
; Check LoopInfo specifically.
; RUN: opt -disable-output -disable-verify -debug-pass-manager %s 2>&1 \
; RUN:     -passes='require<loops>,require<aa>,invalidate<loops>,aa-eval' -aa-pipeline='basic-aa' \
; RUN:     | FileCheck %s --check-prefix=CHECK-LI-INVALIDATE
; CHECK-LI-INVALIDATE: Running pass: RequireAnalysisPass
; CHECK-LI-INVALIDATE: Running analysis: BasicAA
; CHECK-LI-INVALIDATE: Running pass: InvalidateAnalysisPass
; CHECK-LI-INVALIDATE: Invalidating analysis: LoopAnalysis
; CHECK-LI-INVALIDATE: Invalidating analysis: BasicAA
; CHECK-LI-INVALIDATE: Running pass: AAEvaluator
; CHECK-LI-INVALIDATE: Running analysis: BasicAA

; Some code that will result in actual AA queries, including inside of a loop.
; FIXME: Sadly, none of these queries managed to use either the domtree or
; loopinfo that basic-aa cache. But nor does any other test in LLVM. It would
; be good to enhance this to actually use these other analyses to make this
; a more thorough test.
define void @foo(i1 %x, i8* %p1, i8* %p2) {
entry:
  %p3 = alloca i8
  store i8 42, i8* %p1
  %gep2 = getelementptr i8, i8* %p2, i32 0
  br i1 %x, label %loop, label %exit

loop:
  store i8 13, i8* %p3
  %tmp1 = load i8, i8* %gep2
  br label %loop

exit:
  ret void
}
