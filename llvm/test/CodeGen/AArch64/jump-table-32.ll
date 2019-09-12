; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64_32-apple-ios7.0 -aarch64-enable-atomic-cfg-tidy=0 | FileCheck %s

define i32 @test_jumptable(i32 %in) {
; CHECK: test_jumptable

  switch i32 %in, label %def [
    i32 0, label %lbl1
    i32 1, label %lbl2
    i32 2, label %lbl3
    i32 4, label %lbl4
  ]
; CHECK: adrp    [[JTPAGE:x[0-9]+]], LJTI0_0@PAGE
; CHECK: mov     w[[INDEX:[0-9]+]], w0
; CHECK: add     x[[JT:[0-9]+]], [[JTPAGE]], LJTI0_0@PAGEOFF
; CHECK: adr     [[BASE_BLOCK:x[0-9]+]], LBB0_2
; CHECK: ldrb    w[[OFFSET:[0-9]+]], [x[[JT]], x[[INDEX]]]
; CHECK: add     [[DEST:x[0-9]+]], [[BASE_BLOCK]], x[[OFFSET]], lsl #2
; CHECK: br      [[DEST]]

def:
  ret i32 0

lbl1:
  ret i32 1

lbl2:
  ret i32 2

lbl3:
  ret i32 4

lbl4:
  ret i32 8

}

; CHECK: LJTI0_0:
; CHECK-NEXT: .byte
; CHECK-NEXT: .byte
; CHECK-NEXT: .byte
; CHECK-NEXT: .byte
; CHECK-NEXT: .byte
