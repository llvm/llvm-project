; RUN: llc < %s -mtriple=powerpc-unknown-aix-xcoff -verify-machineinstrs \
; RUN:     -mcpu=pwr7 -mattr=+altivec -O0 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-aix-xcoff -verify-machineinstrs \
; RUN:     -mcpu=pwr7 -mattr=+altivec -O0 2>&1 | FileCheck %s

; CHECK: warning: inline asm clobber list contains reserved registers: R2
; CHECK-NEXT: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.

@a = external global i32, align 4

define void @bar() {
  store i32 0, ptr @a, align 4
  call void asm sideeffect "li 2, 1", "~{r2}"()
  ret void
}
