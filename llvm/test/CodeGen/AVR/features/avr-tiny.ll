; RUN: llc -mattr=avrtiny -O0 < %s -march=avr | FileCheck %s

define i16 @reg_copy16(i16, i16 %a) {
; CHECK-LABEL: reg_copy16
; CHECK: mov r24, r22
; CHECK: mov r25, r23

  ret i16 %a
}
