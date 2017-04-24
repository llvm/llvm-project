; RUN: llc -mtriple=thumb-eabi %s -verify-machineinstrs -o - | FileCheck %s
; RUN: llc -mtriple=thumbv6-eabi %s -verify-machineinstrs -o - | FileCheck %s

define i1 @test(i64 %arg) {
entry:
  %ispos = icmp sgt i64 %arg, -1
  %neg = sub i64 0, %arg
  %sel = select i1 %ispos, i64 %arg, i64 %neg
  %cmp2 = icmp eq i64 %sel, %arg
  ret i1 %cmp2
}

; The scheduler used to ignore OptionalDefs, and could unwittingly insert
; a flag-setting instruction in between an ADDS and the corresponding ADC.

; CHECK: adds
; CHECK-NOT: eors
; CHECK: adcs
