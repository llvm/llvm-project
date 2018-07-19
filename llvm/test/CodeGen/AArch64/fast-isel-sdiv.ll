; RUN: llc -mtriple=aarch64-apple-darwin                             -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

define i32 @sdiv_i32_exact(i32 %a) {
; CHECK-LABEL: sdiv_i32_exact
; CHECK:       asr {{w[0-9]+}}, w0, #3
  %1 = sdiv exact i32 %a, 8
  ret i32 %1
}

define i32 @sdiv_i32_pos(i32 %a) {
; CHECK-LABEL: sdiv_i32_pos
; CHECK:       add [[REG1:w[0-9]+]], w0, #7
; CHECK-NEXT:  cmp w0, #0
; CHECK-NEXT:  csel [[REG2:w[0-9]+]], [[REG1]], w0, lt
; CHECK-NEXT:  asr {{w[0-9]+}}, [[REG2]], #3
  %1 = sdiv i32 %a, 8
  ret i32 %1
}

define i32 @sdiv_i32_neg(i32 %a) {
; CHECK-LABEL: sdiv_i32_neg
; CHECK:       add [[REG1:w[0-9]+]], w0, #7
; CHECK-NEXT:  cmp w0, #0
; CHECK-NEXT:  csel [[REG2:w[0-9]+]], [[REG1]], w0, lt
; CHECK-NEXT:  neg {{w[0-9]+}}, [[REG2]], asr #3
  %1 = sdiv i32 %a, -8
  ret i32 %1
}

define i64 @sdiv_i64_exact(i64 %a) {
; CHECK-LABEL: sdiv_i64_exact
; CHECK:       asr {{x[0-9]+}}, x0, #4
  %1 = sdiv exact i64 %a, 16
  ret i64 %1
}

define i64 @sdiv_i64_pos(i64 %a) {
; CHECK-LABEL: sdiv_i64_pos
; CHECK:       add [[REG1:x[0-9]+]], x0, #15
; CHECK-NEXT:  cmp x0, #0
; CHECK-NEXT:  csel [[REG2:x[0-9]+]], [[REG1]], x0, lt
; CHECK-NEXT:  asr {{x[0-9]+}}, [[REG2]], #4
  %1 = sdiv i64 %a, 16
  ret i64 %1
}

define i64 @sdiv_i64_neg(i64 %a) {
; CHECK-LABEL: sdiv_i64_neg
; CHECK:       add [[REG1:x[0-9]+]], x0, #15
; CHECK-NEXT:  cmp x0, #0
; CHECK-NEXT:  csel [[REG2:x[0-9]+]], [[REG1]], x0, lt
; CHECK-NEXT:  neg {{x[0-9]+}}, [[REG2]], asr #4
  %1 = sdiv i64 %a, -16
  ret i64 %1
}
