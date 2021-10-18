; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @ext0(i32 %a) {
; CHECK-NOT: srl $a0, $a0, 12
; CHECK-NOT: andi $a0, $a0, 255
; CHECK: ext $a0, $a0, 12, 8
  %rshift = lshr i32 %a, 12
  %and = and i32 %rshift, 255 ; 0xff
  ret i32 %and
}

define i32 @ext1(i32 %a) {
; CHECK-NOT: li $a1, 1048575
; CHECK-NOT: and $a0, $a0, $a1
; CHECK: ext $a0, $a0, 0, 20
  %and = and i32 %a, 1048575 ; 0xfffff
  ret i32 %and
}

define i32 @notext(i32 %a) {
; CHECK-NOT: ext $a0, $a0, 0, 12
; CHECK: andi $a0, $a0, 4095
  %and = and i32 %a, 4095 ; 0xfff
  ret i32 %and
}
