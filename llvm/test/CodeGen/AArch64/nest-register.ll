; RUN: llc -disable-post-ra -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

; Tests that the 'nest' parameter attribute causes the relevant parameter to be
; passed in the right register.

define ptr @nest_receiver(ptr nest %arg) nounwind {
; CHECK-LABEL: nest_receiver:
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: mov x0, x18
; CHECK-NEXT: ret

  ret ptr %arg
}

define ptr @nest_caller(ptr %arg) nounwind {
; CHECK-LABEL: nest_caller:
; CHECK: mov x18, x0
; CHECK-NEXT: bl nest_receiver
; CHECK: ret

  %result = call ptr @nest_receiver(ptr nest %arg)
  ret ptr %result
}
