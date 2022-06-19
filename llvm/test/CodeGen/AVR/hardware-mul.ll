; RUN: llc -mattr=mul,movw < %s -march=avr | FileCheck %s

; Tests lowering of multiplication to hardware instructions.

define i8 @mult8(i8 %a, i8 %b) {
; CHECK-LABEL: mult8:
; CHECK: muls r22, r24
; CHECK: clr r1
; CHECK: mov  r24, r0
  %mul = mul i8 %b, %a
  ret i8 %mul
}

define i16 @mult16(i16 %a, i16 %b) {
; CHECK-LABEL: mult16:
; CHECK: muls r22, r25
; CHECK: mov  r20, r0
; CHECK: mul  r22, r24
; CHECK: mov  r21, r0
; CHECK: mov  r18, r1
; CHECK: clr r1
; CHECK: add  r18, r20
; CHECK: muls r23, r24
; CHECK: clr r1
; CHECK: add  r18, r0
; :TODO: finish after reworking shift instructions
  %mul = mul nsw i16 %b, %a
  ret i16 %mul
}
