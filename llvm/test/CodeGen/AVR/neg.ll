; RUN: llc < %s -march=avr | FileCheck %s

define i8 @neg8(i8 %x) {
; CHECK-LABEL: neg8:
; CHECK: neg r24
  %sub = sub i8 0, %x
  ret i8 %sub
}
