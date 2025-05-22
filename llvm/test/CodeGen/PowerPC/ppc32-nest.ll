; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc-unknown-linux-gnu"

; Tests that the 'nest' parameter attribute causes the relevant parameter to be
; passed in the right register (r11 for PPC).

define ptr @nest_receiver(ptr nest %arg) nounwind {
; CHECK-LABEL: nest_receiver:
; CHECK: # %bb.0:
; CHECK-NEXT: mr 3, 11
; CHECK-NEXT: blr

  ret ptr %arg
}

define ptr @nest_caller(ptr %arg) nounwind {
; CHECK-LABEL: nest_caller:
; CHECK: mr 11, 3
; CHECK: stw 0, 20(1)
; CHECK-NEXT: bl nest_receiver
; CHECK: blr

  %result = call ptr @nest_receiver(ptr nest %arg)
  ret ptr %result
}

