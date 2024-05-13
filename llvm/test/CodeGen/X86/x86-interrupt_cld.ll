; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Checks that interrupt handler code calls cld before calling an external
;; function.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; CHECK: cld
; CHECK: call

define x86_intrcc void @foo(ptr byval(i8) %frame) {
  call void @bar()
  ret void
}

declare void @bar()

