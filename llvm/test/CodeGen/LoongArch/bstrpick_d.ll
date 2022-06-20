; RUN: llc --mtriple=loongarch64 < %s | FileCheck %s

define i64 @lshr40_and255(i64 %a) {
; CHECK-LABEL: lshr40_and255:
; CHECK:       # %bb.0:
; CHECK-NEXT:    bstrpick.d $a0, $a0, 47, 40
; CHECK-NEXT:    jirl $zero, $ra, 0
  %shr = lshr i64 %a, 40
  %and = and i64 %shr, 255
  ret i64 %and
}

define i64 @ashr50_and511(i64 %a) {
; CHECK-LABEL: ashr50_and511:
; CHECK:       # %bb.0:
; CHECK-NEXT:    bstrpick.d $a0, $a0, 58, 50
; CHECK-NEXT:    jirl $zero, $ra, 0
  %shr = ashr i64 %a, 50
  %and = and i64 %shr, 511
  ret i64 %and
}

define i64 @zext_i32_to_i64(i32 %a) {
; CHECK-LABEL: zext_i32_to_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    bstrpick.d $a0, $a0, 31, 0
; CHECK-NEXT:    jirl $zero, $ra, 0
  %res = zext i32 %a to i64
  ret i64 %res
}

define i64 @and8191(i64 %a) {
; CHECK-LABEL: and8191:
; CHECK:       # %bb.0:
; CHECK-NEXT:    bstrpick.d $a0, $a0, 12, 0
; CHECK-NEXT:    jirl $zero, $ra, 0
  %and = and i64 %a, 8191
  ret i64 %and
}

;; Check that andi but not bstrpick.d is generated.
define i64 @and4095(i64 %a) {
; CHECK-LABEL: and4095:
; CHECK:       # %bb.0:
; CHECK-NEXT:    andi $a0, $a0, 4095
; CHECK-NEXT:    jirl $zero, $ra, 0
  %and = and i64 %a, 4095
  ret i64 %and
}
