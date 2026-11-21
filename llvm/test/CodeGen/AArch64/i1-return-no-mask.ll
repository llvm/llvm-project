; RUN: llc -mtriple=aarch64-linux-gnu -o - %s | FileCheck %s

; AAPCS64 only requires the least significant byte of a bool return value to be
; 0 or 1; the upper bits of the result register are unspecified. When the
; returned value is already known to be a zero-extended bool, no mask has to be
; emitted at all.
;
; See https://github.com/llvm/llvm-project/issues/242721

define i1 @passthrough(i1 %a) {
; CHECK-LABEL: passthrough:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ret
  ret i1 %a
}

define i1 @not_arg(i1 %a) {
; CHECK-LABEL: not_arg:
; CHECK:       // %bb.0:
; CHECK-NEXT:    eor w0, w0, #0x1
; CHECK-NEXT:    ret
  %r = xor i1 %a, true
  ret i1 %r
}

define i1 @or_args(i1 %a, i1 %b) {
; CHECK-LABEL: or_args:
; CHECK:       // %bb.0:
; CHECK-NEXT:    orr w0, w0, w1
; CHECK-NEXT:    ret
  %r = or i1 %a, %b
  ret i1 %r
}

define i1 @and_args(i1 %a, i1 %b) {
; CHECK-LABEL: and_args:
; CHECK:       // %bb.0:
; CHECK-NEXT:    and w0, w0, w1
; CHECK-NEXT:    ret
  %r = and i1 %a, %b
  ret i1 %r
}

define i1 @xor_args(i1 %a, i1 %b) {
; CHECK-LABEL: xor_args:
; CHECK:       // %bb.0:
; CHECK-NEXT:    eor w0, w0, w1
; CHECK-NEXT:    ret
  %r = xor i1 %a, %b
  ret i1 %r
}

define i1 @and_not_arg(i1 %a, i1 %b) {
; CHECK-LABEL: and_not_arg:
; CHECK:       // %bb.0:
; CHECK-NEXT:    eor w8, w1, #0x1
; CHECK-NEXT:    and w0, w0, w8
; CHECK-NEXT:    ret
  %nb = xor i1 %b, true
  %r = and i1 %a, %nb
  ret i1 %r
}

; The upper bits of %v are unknown, so the low byte still has to be masked.
define i1 @unknown_high_bits(i32 %v) {
; CHECK-LABEL: unknown_high_bits:
; CHECK:       // %bb.0:
; CHECK-NEXT:    and w0, w0, #0x1
; CHECK-NEXT:    ret
  %t = trunc i32 %v to i1
  ret i1 %t
}

; An explicit "zeroext" return still requires the full 32-bit value to be
; zero-extended.
define zeroext i1 @zeroext_ret(i1 %a) {
; CHECK-LABEL: zeroext_ret:
; CHECK:       // %bb.0:
; CHECK-NEXT:    and w0, w0, #0x1
; CHECK-NEXT:    ret
  ret i1 %a
}
