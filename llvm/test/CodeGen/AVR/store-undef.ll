; RUN: llc < %s -march=avr | FileCheck %s

; This test checks that we can successfully lower a store
; to an undefined pointer.

; CHECK-LABEL: foo
define void @foo() {

  ; CHECK:      ldi [[SRC:r[0-9]+]], 0
  ; CHECK-NEXT: st [[PTRREG:X|Y|Z]], [[SRC]]
  store i8 0, i8* undef, align 4
  ret void
}
