; RUN: llc < %s -mtriple=i686-- -no-integrated-as | FileCheck %s

; If the target does not have 64-bit integer registers, emit 32-bit register
; names.

; CHECK: movq (%e{{[abcd]}}x, %ebx, 4)

define void @q_modifier(i32* %p) {
entry:
  tail call void asm sideeffect "movq (${0:q}, %ebx, 4), %mm0", "r,~{dirflag},~{fpsr},~{flags}"(i32* %p)
  ret void
}
