; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-m3 | FileCheck %s

@foo = common global ptr null, align 4

; Make sure in the presence of a tail call, r9 doesn't get used to hold
; the destination address. It's callee-saved in AAPCS.
define arm_aapcscc void @test(i32 %a) nounwind {
; CHECK-LABEL: test:
; CHECK-NOT: bx r9
  %tmp = load ptr, ptr @foo, align 4
  tail call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r12}"() nounwind
  tail call arm_aapcscc void %tmp() nounwind
  ret void
}
