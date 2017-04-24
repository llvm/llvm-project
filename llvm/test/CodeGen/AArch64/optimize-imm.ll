; RUN: llc -o - %s -mtriple=aarch64-- | FileCheck %s

; CHECK-LABEL: and1:
; CHECK: and {{w[0-9]+}}, w0, #0xfffffffd

define void @and1(i32 %a, i8* nocapture %p) {
entry:
  %and = and i32 %a, 253
  %conv = trunc i32 %and to i8
  store i8 %conv, i8* %p, align 1
  ret void
}

; (a & 0x3dfd) | 0xffffc000
;
; CHECK-LABEL: and2:
; CHECK: and {{w[0-9]+}}, w0, #0xfdfdfdfd

define i32 @and2(i32 %a) {
entry:
  %and = and i32 %a, 15869
  %or = or i32 %and, -16384
  ret i32 %or
}

; (a & 0x19) | 0xffffffc0
;
; CHECK-LABEL: and3:
; CHECK: and {{w[0-9]+}}, w0, #0x99999999

define i32 @and3(i32 %a) {
entry:
  %and = and i32 %a, 25
  %or = or i32 %and, -64
  ret i32 %or
}

; (a & 0xc5600) | 0xfff1f1ff
;
; CHECK-LABEL: and4:
; CHECK: and {{w[0-9]+}}, w0, #0xfffc07ff

define i32 @and4(i32 %a) {
entry:
  %and = and i32 %a, 787968
  %or = or i32 %and, -921089
  ret i32 %or
}

; Make sure we don't shrink or optimize an XOR's immediate operand if the
; immediate is -1. Instruction selection turns (and ((xor $mask, -1), $v0)) into
; a BIC.

; CHECK-LABEL: xor1:
; CHECK: orr [[R0:w[0-9]+]], wzr, #0x38
; CHECK: bic {{w[0-9]+}}, [[R0]], w0, lsl #3

define i32 @xor1(i32 %a) {
entry:
  %shl = shl i32 %a, 3
  %xor = and i32 %shl, 56
  %and = xor i32 %xor, 56
  ret i32 %and
}
