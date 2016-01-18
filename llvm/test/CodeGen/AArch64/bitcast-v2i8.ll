; RUN: llc < %s -mtriple=aarch64-apple-ios | FileCheck %s

; Part of PR21549: going through the stack isn't ideal but is correct.

define i16 @test_bitcast_v2i8_to_i16(<2 x i8> %a) {
; CHECK-LABEL: test_bitcast_v2i8_to_i16
; CHECK:      mov.s   [[WREG_HI:w[0-9]+]], v0[1]
; CHECK-NEXT: fmov    [[WREG_LO:w[0-9]+]], s0
; CHECK-NEXT: strb    [[WREG_HI]], [sp, #15]
; CHECK-NEXT: strb    [[WREG_LO]], [sp, #14]
; CHECK-NEXT: ldrh    w0, [sp, #14]

  %aa = bitcast <2 x i8> %a to i16
  ret i16 %aa
}
