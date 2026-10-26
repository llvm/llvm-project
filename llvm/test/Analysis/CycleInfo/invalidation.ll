; RUN: opt < %s -disable-output -debug-pass-manager \
; RUN:   -passes='require<cycles>,dce,require<cycles>' 2>&1 \
; RUN:   | FileCheck %s --check-prefix=PRESERVE
; RUN: opt < %s -disable-output -debug-pass-manager \
; RUN:   -passes='require<cycles>,simplifycfg,require<cycles>' 2>&1 \
; RUN:   | FileCheck %s --check-prefix=INVALIDATE

; DCE preserves the CFG, so CycleAnalysis should not be recomputed.
; PRESERVE: Running analysis: CycleAnalysis on f
; PRESERVE: Running pass: DCEPass on f
; PRESERVE-NOT: Invalidating analysis: CycleAnalysis on f
; PRESERVE-NOT: Running analysis: CycleAnalysis on f

; SimplifyCFG does not preserve the CFG, so CycleAnalysis must be recomputed.
; INVALIDATE: Running analysis: CycleAnalysis on f
; INVALIDATE: Running pass: SimplifyCFGPass on f
; INVALIDATE: Invalidating analysis: CycleAnalysis on f
; INVALIDATE: Running analysis: CycleAnalysis on f

define void @f(i32 %x) {
entry:
  %dead = add i32 %x, 1
  br label %loop

loop:
  br i1 true, label %loop, label %exit

exit:
  ret void
}
