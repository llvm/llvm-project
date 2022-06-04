; Test XOR instructions.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 -verify-machineinstrs | FileCheck %s

; Check two register operands.
define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: jmp.n %r1
; CHECK: xor %r2, %r2, %r3
  %res = xor i32 %a, %b
  ret i32 %res
}

; Check two register operands, second operand inverted.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: jmp.n %r1
; CHECK: xor.c %r2, %r2, %r3
  %notb = xor i32 %b, -1
  %res = xor i32 %a, %notb
  ret i32 %res
}

; Check immediate in low 16 bits, high 16 bits clear.
define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: jmp.n %r1
; CHECK: xor %r2, %r2, 255
  %res = xor i32 %a, 255
  ret i32 %res
}

; Check immediate in high 16 bits, low 16 bits set.
define i32 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK: jmp.n %r1
; CHECK: xor.u %r2, %r2, 47806
  %res = xor i32 %a, 3133014016 ; = 0xBABE0000
  ret i32 %res
}

; Check 32-bit immediate.
define i32 @func_f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK: xor.u %r2, %r2, 61680
; CHECK: jmp.n %r1
; CHECK: xor %r2, %r2, 61680
  %res = xor i32 %a, 4042322160 ; = 0xF0F0F0F0
  ret i32 %res
}
