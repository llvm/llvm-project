; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_jumptable(i32 %in) {
  switch i32 %in, label %def [
    i32 0, label %lbl1
    i32 1, label %lbl2
    i32 2, label %lbl3
    i32 4, label %lbl4
  ]

; CHECK: li $t4
; CHECK: Li_NM
; CHECK: lwxs $t4, $a0($t4)
; CHECK: LWXS_NM
; CHECK: jrc $t4
; CHECK: JRC_NM

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
