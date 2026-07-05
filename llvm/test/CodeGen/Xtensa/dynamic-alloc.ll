; RUN: llc -mtriple=xtensa -disable-block-placement -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

define ptr @test_simple_alloca(i32 %numelts) {
; CHECK-LABEL: test_simple_alloca
; CHECK:       addi  a8, a1, -16
; CHECK:       or  a1, a8, a8
; CHECK:       s32i  a15, a1, 0
; CHECK:       or a15, a1, a1
; CHECK:       addi  a8, a2, 3
; CHECK-NEXT:  movi  a9, -4
; CHECK-NEXT:  and  a8, a8, a9
; CHECK-NEXT:  addi  a8, a8, 31
; CHECK-NEXT:  movi  a9, -32
; CHECK-NEXT:  and  a8, a8, a9
; CHECK-NEXT:  sub  a1, a1, a8
; CHECK-NEXT:  or  a2, a1, a1
; CHECK-NEXT:  or  a1, a15, a15
; CHECK-NEXT:  l32i  a15, a1, 0
; CHECK-NEXT:  addi  a8, a1, 16
; CHECK-NEXT:  or  a1, a8, a8
; CHECK-NEXT:  ret

  %addr = alloca i8, i32 %numelts
  ret ptr %addr
}
