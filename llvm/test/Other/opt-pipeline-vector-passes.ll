; RUN: opt -disable-verify -debug-pass-manager -passes='default<O1>' -S %s 2>&1 | FileCheck %s --check-prefixes=O1
; RUN: opt -disable-verify -debug-pass-manager -passes='default<O2>' -S %s 2>&1 | FileCheck %s --check-prefixes=O2
; RUN: opt -disable-verify -debug-pass-manager -passes='default<O2>' -extra-vectorizer-passes -S %s 2>&1 | FileCheck %s --check-prefixes=O2_EXTRA

; REQUIRES: asserts

; The loop vectorizer still runs at both -O1/-O2 even with the
; debug flag, but it only works on loops explicitly annotated
; with pragmas.

; SLP does not run at -O1. Loop vectorization runs, but it only
; works on loops explicitly annotated with pragmas.
; O1-LABEL:  Running pass: LoopVectorizePass
; O1-NOT:    Running pass: SLPVectorizerPass
; O1:        Running pass: VectorCombinePass

; Everything runs at -O2.
; O2-LABEL:  Running pass: LoopVectorizePass
; O2:        Running pass: SLPVectorizerPass
; O2:        Running pass: VectorCombinePass

; Optionally run cleanup passes.
; O2_EXTRA-LABEL: Running pass: LoopVectorizePass
; O2_EXTRA: Running pass: EarlyCSEPass
; O2_EXTRA: Running pass: CorrelatedValuePropagationPass
; O2_EXTRA: Running pass: InstCombinePass
; O2_EXTRA: Running pass: LICMPass
; O2_EXTRA: Running pass: SimpleLoopUnswitchPass
; O2_EXTRA: Running pass: SimplifyCFGPass
; O2_EXTRA: Running pass: InstCombinePass
; O2_EXTRA: Running pass: SLPVectorizerPass
; O2_EXTRA: Running pass: EarlyCSEPass
; O2_EXTRA: Running pass: VectorCombinePass

define i64 @f(i1 %cond) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i64 %i, 1
  br i1 %cond, label %loop, label %exit

exit:
  ret i64 %i
}
