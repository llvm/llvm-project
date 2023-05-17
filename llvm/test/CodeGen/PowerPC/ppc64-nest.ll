; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

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
; CHECK: std 0, 128(1)
; CHECK-NEXT: bl nest_receiver
; CHECK: blr

  %result = call ptr @nest_receiver(ptr nest %arg)
  ret ptr %result
}

define void @test_indirect(ptr nocapture %f, ptr %p) {
entry:

; CHECK-LABEL: test_indirect
; CHECK-DAG: ld [[DEST:[0-9]+]], 0(3)
; CHECK-DAG: ld 2, 8(3)
; CHECK-DAG: mr 11, 4
; CHECK: mtctr [[DEST]]
; CHECK: bctrl
; CHECK: blr

  %call = tail call signext i32 %f(ptr nest %p)
  ret void
}

