; RUN: llc < %s -march=avr | FileCheck %s

; zext R25:R24, R24
; eor R25, R25
define i16 @zext1(i8 %x) {
; CHECK-LABEL: zext1:
; CHECK: eor r25, r25
  %1 = zext i8 %x to i16
  ret i16 %1
}

; zext R25:R24, R20
; mov R24, R20
; eor R25, R25
define i16 @zext2(i8 %x, i8 %y) {
; CHECK-LABEL: zext2:
; CHECK: mov r24, r22
; CHECK: eor r25, r25
  %1 = zext i8 %y to i16
  ret i16 %1
}

; zext R25:R24, R24
; eor R25, R25
define i16 @zext_i1(i1 %x) {
; CHECK-LABEL: zext_i1:
; CHECK: eor r25, r25
  %1 = zext i1 %x to i16
  ret i16 %1
}

