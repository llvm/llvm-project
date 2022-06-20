; RUN: llc --mtriple=loongarch32 < %s | FileCheck %s

define i32 @lshr40_and255(i32 %a) {
; CHECK-LABEL: lshr40_and255:
; CHECK:       # %bb.0:
; CHECK-NEXT:    bstrpick.w $a0, $a0, 17, 10
; CHECK-NEXT:    jirl $zero, $ra, 0
  %shr = lshr i32 %a, 10
  %and = and i32 %shr, 255
  ret i32 %and
}

define i32 @ashr50_and511(i32 %a) {
; CHECK-LABEL: ashr50_and511:
; CHECK:       # %bb.0:
; CHECK-NEXT:    bstrpick.w $a0, $a0, 28, 20
; CHECK-NEXT:    jirl $zero, $ra, 0
  %shr = ashr i32 %a, 20
  %and = and i32 %shr, 511
  ret i32 %and
}

define i32 @zext_i16_to_i32(i16 %a) {
; CHECK-LABEL: zext_i16_to_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    bstrpick.w $a0, $a0, 15, 0
; CHECK-NEXT:    jirl $zero, $ra, 0
  %res = zext i16 %a to i32
  ret i32 %res
}

define i32 @and8191(i32 %a) {
; CHECK-LABEL: and8191:
; CHECK:       # %bb.0:
; CHECK-NEXT:    bstrpick.w $a0, $a0, 12, 0
; CHECK-NEXT:    jirl $zero, $ra, 0
  %and = and i32 %a, 8191
  ret i32 %and
}

;; Check that andi but not bstrpick.d is generated.
define i32 @and4095(i32 %a) {
; CHECK-LABEL: and4095:
; CHECK:       # %bb.0:
; CHECK-NEXT:    andi $a0, $a0, 4095
; CHECK-NEXT:    jirl $zero, $ra, 0
  %and = and i32 %a, 4095
  ret i32 %and
}
