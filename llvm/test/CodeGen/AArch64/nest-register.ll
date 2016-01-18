; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

; Tests that the 'nest' parameter attribute causes the relevant parameter to be
; passed in the right register.

define i8* @nest_receiver(i8* nest %arg) nounwind {
; CHECK-LABEL: nest_receiver:
; CHECK-NEXT: // BB#0:
; CHECK-NEXT: mov x0, x18
; CHECK-NEXT: ret

  ret i8* %arg
}

define i8* @nest_caller(i8* %arg) nounwind {
; CHECK-LABEL: nest_caller:
; CHECK: mov x18, x0
; CHECK-NEXT: bl nest_receiver
; CHECK: ret

  %result = call i8* @nest_receiver(i8* nest %arg)
  ret i8* %result
}
