; RUN: llc -mattr=avrtiny -O0 < %s -march=avr | FileCheck %s

define i16 @reg_copy16(i16 %a) {
; CHECK-LABEL: reg_copy16
; CHECK: mov r18, r24
; CHECK: mov r19, r25

  ret i16 %a
}
