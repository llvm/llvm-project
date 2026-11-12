; RUN: llc -fast-isel -fast-isel-abort=1 -verify-machineinstrs -mtriple=aarch64-apple-darwin < %s | FileCheck %s

define zeroext i8 @test_mul8(i8 %lhs, i8 %rhs) {
; CHECK-LABEL: test_mul8:
; CHECK:       mul {{w[0-9]+}}, w0, w1
  %1 = mul i8 %lhs, %rhs
  ret i8 %1
}

define zeroext i16 @test_mul16(i16 %lhs, i16 %rhs) {
; CHECK-LABEL: test_mul16:
; CHECK:       mul {{w[0-9]+}}, w0, w1
  %1 = mul i16 %lhs, %rhs
  ret i16 %1
}

define i32 @test_mul32(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: test_mul32:
; CHECK:       mul {{w[0-9]+}}, w0, w1
  %1 = mul i32 %lhs, %rhs
  ret i32 %1
}

define i64 @test_mul64(i64 %lhs, i64 %rhs) {
; CHECK-LABEL: test_mul64:
; CHECK:       mul {{x[0-9]+}}, x0, x1
  %1 = mul i64 %lhs, %rhs
  ret i64 %1
}

define i32 @test_mul2shift_i32(i32 %a) {
; CHECK-LABEL: test_mul2shift_i32:
; CHECK:       lsl {{w[0-9]+}}, w0, #2
  %1 = mul i32 %a, 4
  ret i32 %1
}

define i64 @test_mul2shift_i64(i64 %a) {
; CHECK-LABEL: test_mul2shift_i64:
; CHECK:       lsl {{x[0-9]+}}, x0, #3
  %1 = mul i64 %a, 8
  ret i64 %1
}

