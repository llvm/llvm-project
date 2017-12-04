; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc-unknown-linux-gnu"

; Tests that the 'nest' parameter attribute causes the relevant parameter to be
; passed in the right register (r11 for PPC).

define i8* @nest_receiver(i8* nest %arg) nounwind {
; CHECK-LABEL: nest_receiver:
; CHECK: # %bb.0:
; CHECK-NEXT: mr 3, 11
; CHECK-NEXT: blr

  ret i8* %arg
}

define i8* @nest_caller(i8* %arg) nounwind {
; CHECK-LABEL: nest_caller:
; CHECK: mr 11, 3
; CHECK-NEXT: bl nest_receiver
; CHECK: blr

  %result = call i8* @nest_receiver(i8* nest %arg)
  ret i8* %result
}

