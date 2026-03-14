; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown-unknown -mattr=+avx < %s | FileCheck %s

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Checks that interrupt handler code does not call "vzeroupper" instruction
;; before iret.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; CHECK: vzeroupper
; CHECK-NEXT: call
; CHECK-NOT: vzeroupper
; CHECK: iret

define x86_intrcc void @foo(ptr byval(i8) %frame) {
  call void @bar()
  ret void
}

declare void @bar()

