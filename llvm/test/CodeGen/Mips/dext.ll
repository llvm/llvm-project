; RUN: llc -march=mips64 -mcpu=mips64r2 -target-abi=n64 < %s -o - | FileCheck %s

define i64 @dext_add_zext(i32 signext %n) {
entry:
  %add = add i32 %n, 1
  %res = zext i32 %add to i64
  ret i64 %res

; CHECK-LABEL: dext_add_zext:
; CHECK:       dext $[[R0:[0-9]+]], $[[R0:[0-9]+]], 0, 32

}

define i32 @ext_and24(i32 signext %a) {
entry:
  %and = and i32 %a, 16777215
  ret i32 %and

; CHECK-LABEL: ext_and24:
; CHECK:       ext $[[R0:[0-9]+]], $[[R1:[0-9]+]], 0, 24

}

define i64 @dext_and32(i64 zeroext %a) {
entry:
  %and = and i64 %a, 4294967295
  ret i64 %and

; CHECK-LABEL: dext_and32:
; CHECK:       dext $[[R0:[0-9]+]], $[[R1:[0-9]+]], 0, 32

}

define i64 @dext_and35(i64 zeroext %a) {
entry:
  %and = and i64 %a, 34359738367
  ret i64 %and

; CHECK-LABEL: dext_and35:
; CHECK:       dextm $[[R0:[0-9]+]], $[[R1:[0-9]+]], 0, 35

}

define i64 @dext_and20(i64 zeroext %a) {
entry:
  %and = and i64 %a, 1048575
  ret i64 %and

; CHECK-LABEL: dext_and20:
; CHECK:       dext $[[R0:[0-9]+]], $[[R1:[0-9]+]], 0, 20

}

define i64 @dext_and16(i64 zeroext %a) {
entry:
  %and = and i64 %a, 65535
  ret i64 %and

; CHECK-LABEL: dext_and16:
; CHECK:       andi $[[R0:[0-9]+]], $[[R1:[0-9]+]], 65535

}

define i64 @dext_lsr_and20(i64 zeroext %a) {
entry:
  %shr = lshr i64 %a, 5
  %and = and i64 %shr, 1048575
  ret i64 %and

; CHECK-LABEL: dext_lsr_and20:
; CHECK:       dext $[[R0:[0-9]+]], $[[R1:[0-9]+]], 5, 20

}

define i64 @dext_lsr_and8(i64 zeroext %a) {
entry:
  %shr = lshr i64 %a, 40
  %and = and i64 %shr, 255
  ret i64 %and

; CHECK-LABEL: dext_lsr_and8:
; CHECK:       dextu $[[R0:[0-9]+]], $[[R1:[0-9]+]], 40, 8

}

define i64 @dext_zext(i32 signext %a) {
entry:
  %conv = zext i32 %a to i64
  ret i64 %conv

; CHECK-LABEL: dext_zext:
; CHECK:       dext $[[R0:[0-9]+]], $[[R1:[0-9]+]], 0, 32

}

define i64 @dext_and_lsr(i64 zeroext %n) {
entry:
  %and = lshr i64 %n, 8
  %shr = and i64 %and, 4095
  ret i64 %shr

; CHECK-LABEL: dext_and_lsr:
; CHECK:       dext $[[R0:[0-9]+]], $[[R1:[0-9]+]], 8, 12

}
