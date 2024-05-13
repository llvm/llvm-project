; RUN: llc -march=avr -verify-machineinstrs < %s | FileCheck %s

define i8 @sign_extended_1_to_8(i1) {
; CHECK-LABEL: sign_extended_1_to_8
entry-block:
  %1 = sext i1 %0 to i8
  ret i8 %1
}

define i16 @sign_extended_1_to_16(i1) {
; CHECK-LABEL: sign_extended_1_to_16
entry-block:
  %1 = sext i1 %0 to i16
  ret i16 %1
}

define i16 @sign_extended_8_to_16(i8) {
; CHECK-LABEL: sign_extended_8_to_16
entry-block:
  %1 = sext i8 %0 to i16
  ret i16 %1
}

define i32 @sign_extended_1_to_32(i1) {
; CHECK-LABEL: sign_extended_1_to_32
entry-block:
  %1 = sext i1 %0 to i32
  ret i32 %1
}

define i32 @sign_extended_8_to_32(i8) {
; CHECK-LABEL: sign_extended_8_to_32
entry-block:
  %1 = sext i8 %0 to i32
  ret i32 %1
}

define i32 @sign_extended_16_to_32(i16) {
; CHECK-LABEL: sign_extended_16_to_32
entry-block:
  %1 = sext i16 %0 to i32
  ret i32 %1
}

define i64 @sign_extended_1_to_64(i1) {
; CHECK-LABEL: sign_extended_1_to_64
entry-block:
  %1 = sext i1 %0 to i64
  ret i64 %1
}

define i64 @sign_extended_8_to_64(i8) {
; CHECK-LABEL: sign_extended_8_to_64
entry-block:
  %1 = sext i8 %0 to i64
  ret i64 %1
}

define i64 @sign_extended_16_to_64(i16) {
; CHECK-LABEL: sign_extended_16_to_64
entry-block:
  %1 = sext i16 %0 to i64
  ret i64 %1
}

define i64 @sign_extended_32_to_64(i32) {
; CHECK-LABEL: sign_extended_32_to_64
entry-block:
  %1 = sext i32 %0 to i64
  ret i64 %1
}
