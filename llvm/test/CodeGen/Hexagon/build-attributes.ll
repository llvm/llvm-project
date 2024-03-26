;; Generate build attributes from llc.

; RUN: llc -mtriple=hexagon-unknown-elf \
; RUN:  -mattr=+hvxv73,+cabac,+v71,+hvx-ieee-fp,+hvx-length128b %s -o - | FileCheck  %s

;      CHECK: .attribute      4, 71  // Tag_arch
; CHECK-NEXT: .attribute      5, 73  // Tag_hvx_arch
; CHECK-NEXT: .attribute      6, 1   // Tag_hvx_ieeefp
; CHECK-NEXT: .attribute      7, 1   // Tag_hvx_qfloat
; CHECK-NEXT: .attribute      8, 1   // Tag_zreg
; CHECK-NEXT: .attribute      10, 1   // Tag_cabac

define i32 @addi(i32 %a) {
  %1 = add i32 %a, 1
  ret i32 %1
}