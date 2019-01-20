; RUN: llc -mattr=avr6,-mul < %s -march=avr | FileCheck %s
; RUN: llc -mcpu=attiny85 < %s -march=avr | FileCheck %s
; RUN: llc -mcpu=ata5272 < %s -march=avr | FileCheck %s
; RUN: llc -mcpu=attiny861a < %s -march=avr | FileCheck %s
; RUN: llc -mcpu=at90usb82 < %s -march=avr | FileCheck %s

; Tests lowering of multiplication to compiler support routines.

; CHECK-LABEL: mul8:
define i8 @mul8(i8 %a, i8 %b) {
; CHECK: mov  r25, r24
; CHECK: mov  r24, r22
; CHECK: mov  r22, r25
; CHECK: call __mulqi3
  %mul = mul i8 %b, %a
  ret i8 %mul
}

; CHECK-LABEL: mul16:
define i16 @mul16(i16 %a, i16 %b) {
; CHECK: movw  r18, r24
; CHECK: movw  r24, r22
; CHECK: movw  r22, r18
; CHECK: call  __mulhi3
  %mul = mul nsw i16 %b, %a
  ret i16 %mul
}

