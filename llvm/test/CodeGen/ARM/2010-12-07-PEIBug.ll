; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a9 | FileCheck %s
; rdar://8728956

define hidden void @foo() nounwind ssp {
entry:
; CHECK: foo:
; CHECK: mov r7, sp
; CHECK-NEXT: vpush {d8}
; CHECK-NEXT: vpush {d10, d11}
  tail call void asm sideeffect "","~{d8},~{d10},~{d11}"() nounwind
; CHECK: vpop {d10, d11}
; CHECK-NEXT: vpop {d8}
  ret void
}

declare hidden float @bar() nounwind readnone ssp
