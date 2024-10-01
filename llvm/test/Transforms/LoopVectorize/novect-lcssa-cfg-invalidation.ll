; RUN: opt -S -passes="loop-vectorize,jump-threading" -debug-pass-manager < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Checks what analyses are invalidated after Loop Vectorization when no actual
; vectorization happens, and the only change LV makes is LCSSA formation.

define i32 @novect(ptr %p) {

; CHECK:           Running pass: LoopVectorizePass on novect
; CHECK-NOT:       Invalidating analysis: ScalarEvolutionAnalysis on novect
; CHECK-NOT:       Invalidating analysis: BranchProbabilityAnalysis on novect
; CHECK-NOT:       Invalidating analysis: BlockFrequencyAnalysis on novect
; CHECK:           Invalidating analysis: DemandedBitsAnalysis on novect
; CHECK-NOT:       Invalidating analysis: LoopAccessAnalysis on novect
; CHECK:           Running pass: JumpThreadingPass on novect

; CHECK:           entry:
; CHECK:             br label %middle
; CHECK:           middle:
; CHECK:             %iv = phi i32 [ 0, %entry ], [ %iv.next, %middle ]
; CHECK:             %x = load volatile i32, ptr %p
; CHECK:             %iv.next = add i32 %iv, 1
; CHECK:             %cond = icmp slt i32 %iv, 1000
; CHECK:             br i1 %cond, label %exit, label %middle
; CHECK:           exit:
; CHECK:             %x.lcssa = phi i32 [ %x, %middle ]
; CHECK:             ret i32 %x.lcssa

entry:
  br label %middle

middle:
  %iv = phi i32 [0, %entry], [%iv.next, %middle]
  %x = load volatile i32, ptr %p
  %iv.next = add i32 %iv, 1
  %cond = icmp slt i32 %iv, 1000
  br i1 %cond, label %exit, label %middle

exit:
  ret i32 %x
}
