; RUN: llc < %s -march=avr | FileCheck %s

define i8 @trunc8_loreg(i16 %x, i16 %y) {
; CHECK-LABEL: trunc8_loreg:
; CHECK: mov r24, r22
; CHECK-NEXT: ret
  %conv = trunc i16 %y to i8
  ret i8 %conv
}

define i8 @trunc8_hireg(i16 %x, i16 %y) {
; CHECK-LABEL: trunc8_hireg:
; CHECK: mov r24, r23
; CHECK-NEXT: ret
  %shr1 = lshr i16 %y, 8
  %conv = trunc i16 %shr1 to i8
  ret i8 %conv
}
