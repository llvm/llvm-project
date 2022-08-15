; RUN: llc -mattr=avr6,sram < %s -march=avr | FileCheck %s

; This file tests whether the compiler correctly works with the r1 register,
; clearing it when needed.

; Test regular use of r1 as a zero register.
; CHECK-LABEL: store8zero:
; CHECK:      st {{[XYZ]}}, r1
; CHECK-NEXT: mov r24, r1
; CHECK-NEXT: ret
define i8 @store8zero(i8* %x) {
  store i8 0, i8* %x
  ret i8 0
}

; Test that mulitplication instructions (mul, muls, etc) clobber r1 and require
; a "clr r1" instruction.
; CHECK-LABEL: mul:
; CHECK:      muls
; CHECK-NEXT: clr r1
; CHECK-NEXT: st {{[XYZ]}}, r0
; CHECK-NEXT: ret
define void @mul(i8* %ptr, i8 %n) {
  %result = mul i8 %n, 3
  store i8 %result, i8* %ptr
  ret void
}
