; RUN: llc < %s -march=avr | FileCheck %s

; sext R17:R16, R13
; mov r16, r13
; mov r17, r13
; lsl r17
; sbc r17, r17
define i16 @sext1(i8 %x, i8 %y) {
; CHECK-LABEL: sext1:
; CHECK: mov r24, r22
; CHECK: mov r25, r22
; CHECK: lsl r25
; CHECK: sbc r25, r25
  %1 = sext i8 %y to i16
  ret i16 %1
}

; sext R17:R16, R16
; mov r17, r16
; lsl r17
; sbc r17, r17
define i16 @sext2(i8 %x) {
; CHECK-LABEL: sext2:
; CHECK: mov r25, r24
; CHECK: lsl r25
; CHECK: sbc r25, r25
  %1 = sext i8 %x to i16
  ret i16 %1
}
