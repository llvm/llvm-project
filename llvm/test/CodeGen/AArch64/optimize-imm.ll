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

; Check that, when (and %t1, 129) is transformed to (and %t0, 0),
; (xor %arg, 129) doesn't get transformed to (xor %arg, 0).
;
; CHECK-LABEL: PR33100:
; CHECK: mov w[[R0:[0-9]+]], #129
; CHECK: eor {{x[0-9]+}}, {{x[0-9]+}}, x[[R0]]

define i64 @PR33100(i64 %arg) {
entry:
  %alloca0 = alloca i64
  store i64 8, i64* %alloca0, align 4
  %t0 = load i64, i64* %alloca0, align 4
  %t1 = shl i64 %arg, %t0
  %and0 = and i64 %t1, 129
  %xor0 = xor i64 %arg, 129
  %t2 = add i64 %and0, %xor0
  ret i64 %t2
}
