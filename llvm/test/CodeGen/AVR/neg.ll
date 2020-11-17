; RUN: llc < %s -march=avr | FileCheck %s

define i8 @neg8(i8 %x) {
; CHECK-LABEL: neg8:
; CHECK: neg r24
  %sub = sub i8 0, %x
  ret i8 %sub
}

define i16 @neg16(i16 %x) {
; CHECK-LABEL: neg16:
; CHECK:       neg r25
; CHECK-next:  neg r24
; CHECK-next:  sbci r25, 0
; CHECK-next:  ret
  %sub = sub i16 0, %x
  ret i16 %sub
}
