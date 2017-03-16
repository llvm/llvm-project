; RUN: llc -march=mips64 -mcpu=octeon -target-abi=n64 < %s -o - | FileCheck %s

define i64 @cins_zext(i32 signext %n) {
entry:
  %shl = shl i32 %n, 5
  %conv = zext i32 %shl to i64
  ret i64 %conv

; CHECK-LABEL: cins_zext:
; CHECK:       cins $[[R0:[0-9]+]], $[[R1:[0-9]+]], 5, 26

}

define i64 @cins_and_shl(i64 zeroext %n) {
entry:
  %and = shl i64 %n, 8
  %shl = and i64 %and, 16776960
  ret i64 %shl

; CHECK-LABEL: cins_and_shl:
; CHECK:       cins $[[R0:[0-9]+]], $[[R1:[0-9]+]], 8, 15

}

define i64 @cins_and_shl32(i64 zeroext %n) {
entry:
  %and = shl i64 %n, 38
  %shl = and i64 %and, 18014123631575040
  ret i64 %shl

; CHECK-LABEL: cins_and_shl32:
; CHECK:       cins32 $[[R0:[0-9]+]], $[[R1:[0-9]+]], 6, 15

}

define zeroext i16 @cins_and_shl_16(i16 zeroext %n) {
entry:
  %0 = shl i16 %n, 2
  %1 = and i16 %0, 60
  ret i16 %1

; CHECK-LABEL: cins_and_shl_16:
; CHECK:       cins $[[R0:[0-9]+]], $[[R1:[0-9]+]], 2, 3

}

define zeroext i8 @cins_and_shl_8(i8 zeroext %n) {
entry:
  %0 = shl i8 %n, 2
  %1 = and i8 %0, 12
  ret i8 %1

; CHECK-LABEL: cins_and_shl_8:
; CHECK:       cins $[[R0:[0-9]+]], $[[R1:[0-9]+]], 2, 1

}

define i32 @cins_i32(i32 signext %a) {
entry:
  %and = shl i32 %a, 17
  %shl = and i32 %and, 536739840
  ret i32 %shl

; CHECK-LABEL: cins_i32:
; CHECK:       cins $[[R0:[0-9]+]], $[[R1:[0-9]+]], 17, 11

}

define i64 @cins_shl_and(i32 signext %n) {
entry:
  %and = and i32 %n, 65535
  %conv = zext i32 %and to i64
  %shl = shl nuw nsw i64 %conv, 31
  ret i64 %shl

; CHECK-LABEL: cins_shl_and:
; CHECK:       cins $[[R0:[0-9]+]], $[[R1:[0-9]+]], 31, 15

}


define i64 @cins_shl_and32(i32 signext %n) {
entry:
  %and = and i32 %n, 65535
  %conv = zext i32 %and to i64
  %shl = shl nuw nsw i64 %conv, 47
  ret i64 %shl

; CHECK-LABEL: cins_shl_and32:
; CHECK:       cins32 $[[R0:[0-9]+]], $[[R1:[0-9]+]], 15, 15

}
