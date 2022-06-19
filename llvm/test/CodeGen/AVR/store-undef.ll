; RUN: llc < %s -march=avr | FileCheck %s

; This test checks that we can successfully lower a store
; to an undefined pointer.

; CHECK-LABEL: foo
define void @foo() {

  ; CHECK: st [[PTRREG:X|Y|Z]], r1
  store i8 0, i8* undef, align 4
  ret void
}
