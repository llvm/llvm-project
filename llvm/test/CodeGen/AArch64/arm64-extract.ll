; RUN: llc < %s -mtriple=arm64-eabi -verify-machineinstrs | FileCheck %s

define i64 @ror_i64(i64 %in) {
; CHECK-LABEL: ror_i64:
    %left = shl i64 %in, 19
    %right = lshr i64 %in, 45
    %val5 = or i64 %left, %right
; CHECK: ror {{x[0-9]+}}, x0, #45
    ret i64 %val5
}

define i32 @ror_i32(i32 %in) {
; CHECK-LABEL: ror_i32:
    %left = shl i32 %in, 9
    %right = lshr i32 %in, 23
    %val5 = or i32 %left, %right
; CHECK: ror {{w[0-9]+}}, w0, #23
    ret i32 %val5
}

define i32 @extr_i32(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: extr_i32:
  %left = shl i32 %lhs, 6
  %right = lshr i32 %rhs, 26
  %val = or i32 %left, %right
  ; Order of lhs and rhs matters here. Regalloc would have to be very odd to use
  ; something other than w0 and w1.
; CHECK: extr {{w[0-9]+}}, w0, w1, #26

  ret i32 %val
}

define i64 @extr_i64(i64 %lhs, i64 %rhs) {
; CHECK-LABEL: extr_i64:
  %right = lshr i64 %rhs, 40
  %left = shl i64 %lhs, 24
  %val = or i64 %right, %left
  ; Order of lhs and rhs matters here. Regalloc would have to be very odd to use
  ; something other than w0 and w1.
; CHECK: extr {{x[0-9]+}}, x0, x1, #40

  ret i64 %val
}

; Regression test: a bad experimental pattern crept into git which optimised
; this pattern to a single EXTR.
define i32 @extr_regress(i32 %a, i32 %b) {
; CHECK-LABEL: extr_regress:

    %sh1 = shl i32 %a, 14
    %sh2 = lshr i32 %b, 14
    %val = or i32 %sh2, %sh1
; CHECK-NOT: extr {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, #{{[0-9]+}}

    ret i32 %val
; CHECK: ret
}
