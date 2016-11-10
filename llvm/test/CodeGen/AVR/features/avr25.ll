; RUN: llc -mattr=avr25 -O0 < %s -march=avr | FileCheck %s

; On most cores, the 16-bit 'MOVW' instruction can be used
define i16 @reg_copy16(i16 %a) {
; CHECK-LABEL: reg_copy16
; CHECK: movw r18, r24
  ret i16 %a
}
