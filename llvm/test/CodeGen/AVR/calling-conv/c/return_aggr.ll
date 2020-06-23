; RUN: llc < %s -march=avr | FileCheck %s

; CHECK-LABEL: ret_struct_i8_i16_i8
define { i8, i16, i8 } @ret_struct_i8_i16_i8() {
start:
  ; for some reason the i16 is loaded to r24:r25
  ; and then moved to r23:r24
  ; CHECK: ldi r22, 64
  ; CHECK-NEXT: r23,
  ; CHECK-NEXT: r24,
  ; CHECK-NEXT: r25, 11
  %0 = insertvalue {i8, i16, i8} undef, i8 64, 0
  %1 = insertvalue {i8, i16, i8} %0, i16 1024, 1
  %2 = insertvalue {i8, i16, i8} %1, i8 11, 2
  ret {i8, i16, i8} %2
}

; CHECK-LABEL: ret_struct_i32_i16
define { i32, i16 } @ret_struct_i32_i16() {
start:
  ; CHECK: ldi r18, 4
  ; CHECK-NEXT: ldi r19, 3
  ; CHECK-NEXT: ldi r20, 2
  ; CHECK-NEXT: ldi r21, 1
  ; CHECK-NEXT: ldi r22, 0
  ; CHECK-NEXT: ldi r23, 8
  %0 = insertvalue { i32, i16 } undef, i32 16909060, 0
  %1 = insertvalue { i32, i16 } %0, i16 2048, 1
  ret { i32, i16} %1
}

