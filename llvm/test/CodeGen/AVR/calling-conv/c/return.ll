; RUN: llc < %s -march=avr | FileCheck %s

; CHECK-LABEL: ret_i8
define i8 @ret_i8() {
  ; CHECK: ldi r24, 64
  ret i8 64
}

; CHECK-LABEL: ret_i16
define i16 @ret_i16() {
  ; CHECK:      ldi     r24, 0
  ; CHECK-NEXT: ldi     r25, 4
  ret i16 1024
}

; CHECK-LABEL: ret_i32
define i32 @ret_i32() {
  ; CHECK:      ldi     r22, 78
  ; CHECK-NEXT: ldi     r23, 97
  ; CHECK-NEXT: ldi     r24, 188
  ; CHECK-NEXT: ldi     r25, 0
  ret i32 12345678
}

; CHECK-LABEL: ret_i64
define i64 @ret_i64() {
  ; CHECK:      ldi     r18, 0
  ; CHECK-NEXT: ldi     r19, 255
  ; CHECK-NEXT: mov     r20, r18
  ; CHECK-NEXT: mov     r21, r19
  ; CHECK-NEXT: mov     r22, r18
  ; CHECK-NEXT: mov     r23, r19
  ; CHECK-NEXT: mov     r24, r18
  ; CHECK-NEXT: mov     r25, r19
  ret i64 18374966859414961920
}
