; RUN: llc -mtriple=aarch64 -verify-machineinstrs < %s | FileCheck %s

; Test that both numeric register names (x29, x30) and their architectural
; aliases (fp, lr) work correctly as clobbers in inline assembly.

define void @clobber_x29() nounwind {
; CHECK-LABEL: clobber_x29:
; CHECK:       str x29, [sp
; CHECK-NEXT:  //APP
; CHECK-NEXT:  //NO_APP
; CHECK-NEXT:  ldr x29, [sp
  tail call void asm sideeffect "", "~{x29}"()
  ret void
}

define void @clobber_fp() nounwind {
; CHECK-LABEL: clobber_fp:
; CHECK:       str x29, [sp
; CHECK-NEXT:  //APP
; CHECK-NEXT:  //NO_APP
; CHECK-NEXT:  ldr x29, [sp
  tail call void asm sideeffect "", "~{fp}"()
  ret void
}

define void @clobber_x30() nounwind {
; CHECK-LABEL: clobber_x30:
; CHECK:       str x30, [sp
; CHECK-NEXT:  //APP
; CHECK-NEXT:  //NO_APP
; CHECK-NEXT:  ldr x30, [sp
  tail call void asm sideeffect "", "~{x30}"()
  ret void
}

define void @clobber_lr() nounwind {
; CHECK-LABEL: clobber_lr:
; CHECK:       str x30, [sp
; CHECK-NEXT:  //APP
; CHECK-NEXT:  //NO_APP
; CHECK-NEXT:  ldr x30, [sp
  tail call void asm sideeffect "", "~{lr}"()
  ret void
}
