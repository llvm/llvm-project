; RUN: llc < %s -mtriple=arm64-apple-ios | FileCheck %s
; rdar://12829704

define void @t8() nounwind ssp {
; CHECK-LABEL: t8:
; CHECK-NOT: stp	fp, lr, [sp, #-16]!
; CHECK-NOT: mov	fp, sp
; CHECK: nop
; CHECK-NOT: mov	sp, fp
; CHECK-NOT: ldp	fp, lr, [sp], #16
  tail call void asm sideeffect "nop", "~{v8}"() nounwind
  ret void
}
