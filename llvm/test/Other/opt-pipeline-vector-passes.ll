; RUN: opt -O1                          -debug-pass=Structure  < %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=OLDPM_O1
; RUN: opt -O2                          -debug-pass=Structure  < %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=OLDPM_O2
; RUN: opt -O2 -extra-vectorizer-passes -debug-pass=Structure  < %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=OLDPM_O2_EXTRA
; RUN: opt -O1 -vectorize-loops=0       -debug-pass=Structure  < %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=OLDPM_O1_FORCE_OFF
; RUN: opt -O2 -vectorize-loops=0       -debug-pass=Structure  < %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=OLDPM_O2_FORCE_OFF
; RUN: opt -disable-verify -debug-pass-manager -passes='default<O1>' -S %s 2>&1 | FileCheck %s --check-prefixes=NEWPM_O1
; RUN: opt -disable-verify -debug-pass-manager -passes='default<O2>' -S %s 2>&1 | FileCheck %s --check-prefixes=NEWPM_O2

; REQUIRES: asserts

; SLP does not run at -O1. Loop vectorization runs, but it only
; works on loops explicitly annotated with pragmas.

; OLDPM_O1-LABEL:  Pass Arguments:
; OLDPM_O1:        Loop Vectorization
; OLDPM_O1-NOT:    SLP Vectorizer
; OLDPM_O1:        Optimize scalar/vector ops

; Everything runs at -O2.

; OLDPM_O2-LABEL:  Pass Arguments:
; OLDPM_O2:        Loop Vectorization
; OLDPM_O2:        SLP Vectorizer
; OLDPM_O2:        Optimize scalar/vector ops

; Optionally run cleanup passes.

; OLDPM_O2_EXTRA-LABEL:  Pass Arguments:
; OLDPM_O2_EXTRA:        Loop Vectorization
; OLDPM_O2_EXTRA:        Early CSE
; OLDPM_O2_EXTRA:        Value Propagation
; OLDPM_O2_EXTRA:        Combine redundant instructions
; OLDPM_O2_EXTRA:        Loop Invariant Code Motion
; OLDPM_O2_EXTRA:        Unswitch loops
; OLDPM_O2_EXTRA:        Simplify the CFG
; OLDPM_O2_EXTRA:        Combine redundant instructions
; OLDPM_O2_EXTRA:        SLP Vectorizer
; OLDPM_O2_EXTRA:        Early CSE
; OLDPM_O2_EXTRA:        Optimize scalar/vector ops


; The loop vectorizer still runs at both -O1/-O2 even with the
; debug flag, but it only works on loops explicitly annotated
; with pragmas.

; OLDPM_O1_FORCE_OFF-LABEL:  Pass Arguments:
; OLDPM_O1_FORCE_OFF:        Loop Vectorization
; OLDPM_O1_FORCE_OFF-NOT:    SLP Vectorizer
; OLDPM_O1_FORCE_OFF:        Optimize scalar/vector ops

; OLDPM_O2_FORCE_OFF-LABEL:  Pass Arguments:
; OLDPM_O2_FORCE_OFF:        Loop Vectorization
; OLDPM_O2_FORCE_OFF:        SLP Vectorizer
; OLDPM_O2_FORCE_OFF:        Optimize scalar/vector ops

; There should be no difference with the new pass manager.
; This is tested more thoroughly in other test files.

; NEWPM_O1-LABEL:  Running pass: LoopVectorizePass
; NEWPM_O1-NOT:    Running pass: SLPVectorizerPass
; NEWPM_O1:        Running pass: VectorCombinePass

; NEWPM_O2-LABEL:  Running pass: LoopVectorizePass
; NEWPM_O2:        Running pass: SLPVectorizerPass
; NEWPM_O2:        Running pass: VectorCombinePass

define void @f() {
  ret void
}
