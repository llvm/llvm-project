; RUN: llc -mattr=mul,movw < %s -mtriple=avr | FileCheck %s

; Unsigned 8-bit remision
define i8 @urem8(i8 %a, i8 %b) {
; CHECK-LABEL: rem8:
; CHECK: call __udivmodqi4
; CHECK-NEXT: mov r24, r25
; CHECK-NEXT: ret
  %rem = urem i8 %a, %b
  ret i8 %rem
}

; Signed 8-bit remision
define i8 @srem8(i8 %a, i8 %b) {
; CHECK-LABEL: srem8:
; CHECK: call __divmodqi4
; CHECK-NEXT: mov r24, r25
; CHECK-NEXT: ret
  %rem = srem i8 %a, %b
  ret i8 %rem
}

; Unsigned 16-bit remision
define i16 @urem16(i16 %a, i16 %b) {
; CHECK-LABEL: urem16:
; CHECK: call __udivmodhi4
; CHECK-NEXT: ret
  %rem = urem i16 %a, %b
  ret i16 %rem
}

; Signed 16-bit remision
define i16 @srem16(i16 %a, i16 %b) {
; CHECK-LABEL: srem16:
; CHECK: call __divmodhi4
; CHECK-NEXT: ret
  %rem = srem i16 %a, %b
  ret i16 %rem
}

; Unsigned 32-bit remision
define i32 @urem32(i32 %a, i32 %b) {
; CHECK-LABEL: urem32:
; CHECK: call __udivmodsi4
; CHECK-NEXT: ret
  %rem = urem i32 %a, %b
  ret i32 %rem
}

; Signed 32-bit remision
define i32 @srem32(i32 %a, i32 %b) {
; CHECK-LABEL: srem32:
; CHECK: call __divmodsi4
; CHECK-NEXT: ret
  %rem = srem i32 %a, %b
  ret i32 %rem
}

