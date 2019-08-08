; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64-apple-ios -aarch64-enable-atomic-cfg-tidy=0 | FileCheck %s

; CHECK-LABEL: test_jumptable:
; CHECK: mov   w[[INDEX:[0-9]+]], w0
; CHECK: cmp   x[[INDEX]], #5
; CHECK: csel  [[INDEX2:x[0-9]+]], x[[INDEX]], xzr, ls
; CHECK: adrp  [[JTPAGE:x[0-9]+]], LJTI0_0@PAGE
; CHECK: add   x[[JT:[0-9]+]], [[JTPAGE]], LJTI0_0@PAGEOFF
; CHECK: ldrsw [[OFFSET:x[0-9]+]], [x[[JT]], [[INDEX2]], lsl #2]
; CHECK: add   [[DEST:x[0-9]+]], x[[JT]], [[OFFSET]]
; CHECK: br    [[DEST]]

define i32 @test_jumptable(i32 %in) "jump-table-hardening" {

  switch i32 %in, label %def [
    i32 0, label %lbl1
    i32 1, label %lbl2
    i32 2, label %lbl3
    i32 4, label %lbl4
    i32 5, label %lbl5
  ]

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

lbl5:
  ret i32 10

}

; CHECK: LJTI0_0:
; CHECK-NEXT: .long LBB{{[0-9_]+}}-LJTI0_0
; CHECK-NEXT: .long LBB{{[0-9_]+}}-LJTI0_0
; CHECK-NEXT: .long LBB{{[0-9_]+}}-LJTI0_0
; CHECK-NEXT: .long LBB{{[0-9_]+}}-LJTI0_0
; CHECK-NEXT: .long LBB{{[0-9_]+}}-LJTI0_0
; CHECK-NEXT: .long LBB{{[0-9_]+}}-LJTI0_0
