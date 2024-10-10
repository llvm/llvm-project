; RUN: llc -mattr=+wide-arithmetic < %s | FileCheck %s

target triple = "wasm32-unknown-unknown"

define i128 @add_i128(i128 %a, i128 %b) {
; CHECK-LABEL:  add_i128:
; CHECK:        i64.add128
  %c = add i128 %a, %b
  ret i128 %c
}

define i128 @sub_i128(i128 %a, i128 %b) {
; CHECK-LABEL:  sub_i128:
; CHECK:        i64.sub128
  %c = sub i128 %a, %b
  ret i128 %c
}

define i128 @mul_i128(i128 %a, i128 %b) {
; CHECK-LABEL:  mul_i128:
; CHECK:        i64.mul_wide_u
  %c = mul i128 %a, %b
  ret i128 %c
}

define i128 @i64_mul_wide_s(i64 %a, i64 %b) {
; CHECK-LABEL: i64_mul_wide_s:
; CHECK:       i64.mul_wide_s
  %a128 = sext i64 %a to i128
  %b128 = sext i64 %b to i128
  %c = mul i128 %a128, %b128
  ret i128 %c
}

define i128 @i64_mul_wide_u(i64 %a, i64 %b) {
; CHECK-LABEL: i64_mul_wide_u:
; CHECK:       i64.mul_wide_u
  %a128 = zext i64 %a to i128
  %b128 = zext i64 %b to i128
  %c = mul i128 %a128, %b128
  ret i128 %c
}

define i64 @mul_i128_only_lo(i128 %a, i128 %b) {
; CHECK-LABEL:  mul_i128_only_lo:
; CHECK-NOT:    i64.mul128
; CHECK:        i64.mul
  %c = mul i128 %a, %b
  %d = trunc i128 %c to i64
  ret i64 %d
}
