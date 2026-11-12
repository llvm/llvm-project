; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel=0                    -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel=1 -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

; AND
define zeroext i1 @and_rr_i1(i1 signext %a, i1 signext %b) {
; CHECK-LABEL: and_rr_i1
; CHECK:       and [[REG:w[0-9]+]], w0, w1
  %1 = and i1 %a, %b
  ret i1 %1
}

define zeroext i8 @and_rr_i8(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: and_rr_i8
; CHECK:       and [[REG:w[0-9]+]], w0, w1
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], #0xff
  %1 = and i8 %a, %b
  ret i8 %1
}

define zeroext i16 @and_rr_i16(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: and_rr_i16
; CHECK:       and [[REG:w[0-9]+]], w0, w1
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], #0xffff
  %1 = and i16 %a, %b
  ret i16 %1
}

define i32 @and_rr_i32(i32 %a, i32 %b) {
; CHECK-LABEL: and_rr_i32
; CHECK:       and w0, w0, w1
  %1 = and i32 %a, %b
  ret i32 %1
}

define i64 @and_rr_i64(i64 %a, i64 %b) {
; CHECK-LABEL: and_rr_i64
; CHECK:       and x0, x0, x1
  %1 = and i64 %a, %b
  ret i64 %1
}

define zeroext i1 @and_ri_i1(i1 signext %a) {
; CHECK-LABEL: and_ri_i1
; CHECK:       and {{w[0-9]+}}, w0, #0x1
  %1 = and i1 %a, 1
  ret i1 %1
}

define zeroext i8 @and_ri_i8(i8 signext %a) {
; CHECK-LABEL: and_ri_i8
; CHECK:       and {{w[0-9]+}}, w0, #0xf
  %1 = and i8 %a, 15
  ret i8 %1
}

define zeroext i16 @and_ri_i16(i16 signext %a) {
; CHECK-LABEL: and_ri_i16
; CHECK:       and {{w[0-9]+}}, w0, #0xff
  %1 = and i16 %a, 255
  ret i16 %1
}

define i32 @and_ri_i32(i32 %a) {
; CHECK-LABEL: and_ri_i32
; CHECK:       and w0, w0, #0xff
  %1 = and i32 %a, 255
  ret i32 %1
}

define i64 @and_ri_i64(i64 %a) {
; CHECK-LABEL: and_ri_i64
; CHECK:       and x0, x0, #0xff
  %1 = and i64 %a, 255
  ret i64 %1
}

define zeroext i8 @and_rs_i8(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: and_rs_i8
; CHECK:       and [[REG:w[0-9]+]], w0, w1, lsl #4
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], {{#0xff|#0xf0}}
  %1 = shl i8 %b, 4
  %2 = and i8 %a, %1
  ret i8 %2
}

define zeroext i16 @and_rs_i16(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: and_rs_i16
; CHECK:       and [[REG:w[0-9]+]], w0, w1, lsl #8
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], {{#0xffff|#0xff00}}
  %1 = shl i16 %b, 8
  %2 = and i16 %a, %1
  ret i16 %2
}

define i32 @and_rs_i32(i32 %a, i32 %b) {
; CHECK-LABEL: and_rs_i32
; CHECK:       and w0, w0, w1, lsl #8
  %1 = shl i32 %b, 8
  %2 = and i32 %a, %1
  ret i32 %2
}

define i64 @and_rs_i64(i64 %a, i64 %b) {
; CHECK-LABEL: and_rs_i64
; CHECK:       and x0, x0, x1, lsl #8
  %1 = shl i64 %b, 8
  %2 = and i64 %a, %1
  ret i64 %2
}

define i32 @and_mul_i32(i32 %a, i32 %b) {
; CHECK-LABEL: and_mul_i32
; CHECK:       and w0, w0, w1, lsl #2
  %1 = mul i32 %b, 4
  %2 = and i32 %a, %1
  ret i32 %2
}

define i64 @and_mul_i64(i64 %a, i64 %b) {
; CHECK-LABEL: and_mul_i64
; CHECK:       and x0, x0, x1, lsl #2
  %1 = mul i64 %b, 4
  %2 = and i64 %a, %1
  ret i64 %2
}

; OR
define zeroext i1 @or_rr_i1(i1 signext %a, i1 signext %b) {
; CHECK-LABEL: or_rr_i1
; CHECK:       orr [[REG:w[0-9]+]], w0, w1
  %1 = or i1 %a, %b
  ret i1 %1
}

define zeroext i8 @or_rr_i8(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: or_rr_i8
; CHECK:       orr [[REG:w[0-9]+]], w0, w1
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], #0xff
  %1 = or i8 %a, %b
  ret i8 %1
}

define zeroext i16 @or_rr_i16(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: or_rr_i16
; CHECK:       orr [[REG:w[0-9]+]], w0, w1
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], #0xffff
  %1 = or i16 %a, %b
  ret i16 %1
}

define i32 @or_rr_i32(i32 %a, i32 %b) {
; CHECK-LABEL: or_rr_i32
; CHECK:       orr w0, w0, w1
  %1 = or i32 %a, %b
  ret i32 %1
}

define i64 @or_rr_i64(i64 %a, i64 %b) {
; CHECK-LABEL: or_rr_i64
; CHECK:       orr x0, x0, x1
  %1 = or i64 %a, %b
  ret i64 %1
}

define zeroext i8 @or_ri_i8(i8 %a) {
; CHECK-LABEL: or_ri_i8
; CHECK:       orr [[REG:w[0-9]+]], w0, #0xf
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], #0xff
  %1 = or i8 %a, 15
  ret i8 %1
}

define zeroext i16 @or_ri_i16(i16 %a) {
; CHECK-LABEL: or_ri_i16
; CHECK:       orr [[REG:w[0-9]+]], w0, #0xff
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], #0xffff
  %1 = or i16 %a, 255
  ret i16 %1
}

define i32 @or_ri_i32(i32 %a) {
; CHECK-LABEL: or_ri_i32
; CHECK:       orr w0, w0, #0xff
  %1 = or i32 %a, 255
  ret i32 %1
}

define i64 @or_ri_i64(i64 %a) {
; CHECK-LABEL: or_ri_i64
; CHECK:       orr x0, x0, #0xff
  %1 = or i64 %a, 255
  ret i64 %1
}

define zeroext i8 @or_rs_i8(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: or_rs_i8
; CHECK:       orr [[REG:w[0-9]+]], w0, w1, lsl #4
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], {{#0xff|#0xf0}}
  %1 = shl i8 %b, 4
  %2 = or i8 %a, %1
  ret i8 %2
}

define zeroext i16 @or_rs_i16(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: or_rs_i16
; CHECK:       orr [[REG:w[0-9]+]], w0, w1, lsl #8
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], {{#0xffff|#0xff00}}
  %1 = shl i16 %b, 8
  %2 = or i16 %a, %1
  ret i16 %2
}

define i32 @or_rs_i32(i32 %a, i32 %b) {
; CHECK-LABEL: or_rs_i32
; CHECK:       orr w0, w0, w1, lsl #8
  %1 = shl i32 %b, 8
  %2 = or i32 %a, %1
  ret i32 %2
}

define i64 @or_rs_i64(i64 %a, i64 %b) {
; CHECK-LABEL: or_rs_i64
; CHECK:       orr x0, x0, x1, lsl #8
  %1 = shl i64 %b, 8
  %2 = or i64 %a, %1
  ret i64 %2
}

define i32 @or_mul_i32(i32 %a, i32 %b) {
; CHECK-LABEL: or_mul_i32
; CHECK:       orr w0, w0, w1, lsl #2
  %1 = mul i32 %b, 4
  %2 = or i32 %a, %1
  ret i32 %2
}

define i64 @or_mul_i64(i64 %a, i64 %b) {
; CHECK-LABEL: or_mul_i64
; CHECK:       orr x0, x0, x1, lsl #2
  %1 = mul i64 %b, 4
  %2 = or i64 %a, %1
  ret i64 %2
}

; XOR
define zeroext i1 @xor_rr_i1(i1 signext %a, i1 signext %b) {
; CHECK-LABEL: xor_rr_i1
; CHECK:       eor [[REG:w[0-9]+]], w0, w1
  %1 = xor i1 %a, %b
  ret i1 %1
}

define zeroext i8 @xor_rr_i8(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: xor_rr_i8
; CHECK:       eor [[REG:w[0-9]+]], w0, w1
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], #0xff
  %1 = xor i8 %a, %b
  ret i8 %1
}

define zeroext i16 @xor_rr_i16(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: xor_rr_i16
; CHECK:       eor [[REG:w[0-9]+]], w0, w1
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], #0xffff
  %1 = xor i16 %a, %b
  ret i16 %1
}

define i32 @xor_rr_i32(i32 %a, i32 %b) {
; CHECK-LABEL: xor_rr_i32
; CHECK:       eor w0, w0, w1
  %1 = xor i32 %a, %b
  ret i32 %1
}

define i64 @xor_rr_i64(i64 %a, i64 %b) {
; CHECK-LABEL: xor_rr_i64
; CHECK:       eor x0, x0, x1
  %1 = xor i64 %a, %b
  ret i64 %1
}

define zeroext i8 @xor_ri_i8(i8 signext %a) {
; CHECK-LABEL: xor_ri_i8
; CHECK:       eor [[REG:w[0-9]+]], w0, #0xf
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], #0xff
  %1 = xor i8 %a, 15
  ret i8 %1
}

define zeroext i16 @xor_ri_i16(i16 signext %a) {
; CHECK-LABEL: xor_ri_i16
; CHECK:       eor [[REG:w[0-9]+]], w0, #0xff
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], #0xffff
  %1 = xor i16 %a, 255
  ret i16 %1
}

define i32 @xor_ri_i32(i32 %a) {
; CHECK-LABEL: xor_ri_i32
; CHECK:       eor w0, w0, #0xff
  %1 = xor i32 %a, 255
  ret i32 %1
}

define i64 @xor_ri_i64(i64 %a) {
; CHECK-LABEL: xor_ri_i64
; CHECK:       eor x0, x0, #0xff
  %1 = xor i64 %a, 255
  ret i64 %1
}

define zeroext i8 @xor_rs_i8(i8 %a, i8 %b) {
; CHECK-LABEL: xor_rs_i8
; CHECK:       eor [[REG:w[0-9]+]], w0, w1, lsl #4
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], {{#0xff|#0xf0}}
  %1 = shl i8 %b, 4
  %2 = xor i8 %a, %1
  ret i8 %2
}

define zeroext i16 @xor_rs_i16(i16 %a, i16 %b) {
; CHECK-LABEL: xor_rs_i16
; CHECK:       eor [[REG:w[0-9]+]], w0, w1, lsl #8
; CHECK-NEXT:  and {{w[0-9]+}}, [[REG]], {{#0xffff|#0xff00}}
  %1 = shl i16 %b, 8
  %2 = xor i16 %a, %1
  ret i16 %2
}

define i32 @xor_rs_i32(i32 %a, i32 %b) {
; CHECK-LABEL: xor_rs_i32
; CHECK:       eor w0, w0, w1, lsl #8
  %1 = shl i32 %b, 8
  %2 = xor i32 %a, %1
  ret i32 %2
}

define i64 @xor_rs_i64(i64 %a, i64 %b) {
; CHECK-LABEL: xor_rs_i64
; CHECK:       eor x0, x0, x1, lsl #8
  %1 = shl i64 %b, 8
  %2 = xor i64 %a, %1
  ret i64 %2
}

define i32 @xor_mul_i32(i32 %a, i32 %b) {
; CHECK-LABEL: xor_mul_i32
; CHECK:       eor w0, w0, w1, lsl #2
  %1 = mul i32 %b, 4
  %2 = xor i32 %a, %1
  ret i32 %2
}

define i64 @xor_mul_i64(i64 %a, i64 %b) {
; CHECK-LABEL: xor_mul_i64
; CHECK:       eor x0, x0, x1, lsl #2
  %1 = mul i64 %b, 4
  %2 = xor i64 %a, %1
  ret i64 %2
}

