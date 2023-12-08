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

; CHECK-T1-LABEL: @test64
; CHECK-T2-LABEL: @test64
; CHECK-T1: movs r4, #0
; CHECK-T1: mvns r5, r4
; CHECK-T1: mov r0, r5
; CHECK-T1: subs r0, #15
; CHECK-T2: subs.w r0, r{{[0-9]+}}, #15
; CHECK-T2-NEXT: sbc r1, r{{[0-9]+}}, #0
define i32 @test64() {
entry:
  tail call void @fn1(i64 -1)
  tail call void @fn1(i64 -1)
  tail call void @fn1(i64 -16)
  ret i32 0
}
declare void @fn1(i64) ;
