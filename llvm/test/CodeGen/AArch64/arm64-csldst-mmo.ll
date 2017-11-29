; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a57 -enable-misched=0 -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s
; REQUIRES: asserts

@G = external global [0 x i32], align 4

; Check that MMOs are added to epilogue calle-save restore loads so
; that the store to G is not considered dependant on the callee-save
; loads.
;
; CHECK: Before post-MI-sched:
; CHECK-LABEL: # Machine code for function test1:
; CHECK: SU(2):   STRWui %wzr
; CHECK: SU(3):   %x21<def>, %x20<def> = LDPXi %sp
; CHECK:  Predecessors:
; CHECK-NEXT:   SU(0): Out
; CHECK-NEXT:   SU(0): Out
; CHECK-NEXT:   SU(0): Ord
; CHECK-NEXT:  Successors:
define void @test1() {
entry:
  tail call void asm sideeffect "nop", "~{x20},~{x21},~{x22},~{x23}"() nounwind
  store i32 0, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @G, i64 0, i64 0), align 4
  ret void
}
