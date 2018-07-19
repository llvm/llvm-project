; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -mcpu=cortex-m0 -verify-machineinstrs | FileCheck --check-prefix CHECK-T1 %s
; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -mcpu=cortex-m3 -verify-machineinstrs | FileCheck --check-prefix CHECK-T2 %s

; CHECK-T1-LABEL: @mov_and_add
; CHECK-T2-LABEL: @mov_and_add
; CHECK-T1: movs r0, #255
; CHECK-T1: adds r0, #12
; CHECK-T2: movw r0, #267
define i32 @mov_and_add() {
  ret i32 267
}

; CHECK-T1-LABEL: @mov_and_add2
; CHECK-T2-LABEL: @mov_and_add2
; CHECK-T1: ldr r0,
; CHECK-T2: movw r0, #511
define i32 @mov_and_add2() {
  ret i32 511
}
