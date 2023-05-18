; RUN: llc < %s -verify-machineinstrs -mtriple=powerpc-unknown-unkown \
; RUN:   -mcpu=pwr7 2>&1 | FileCheck %s
; RUN: llc < %s -verify-machineinstrs -mtriple=powerpc64-unknown-unkown \
; RUN:   -mcpu=pwr7 2>&1 | FileCheck %s

define void @test_r1_clobber() {
entry:
  call void asm sideeffect "nop", "~{r1}"()
  ret void
}

; CHECK: warning: inline asm clobber list contains reserved registers: R1
; CHECK-NEXT: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.

define void @test_x1_clobber() {
entry:
  call void asm sideeffect "nop", "~{x1}"()
  ret void
}

; CHECK: warning: inline asm clobber list contains reserved registers: X1
; CHECK-NEXT: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
