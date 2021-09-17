; Test AND instructions.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 | FileCheck %s

; Check two register operands.
define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: and %r2, %r2, %r3
; CHECK: jmp %r1
  %res = and i32 %a, %b
  ret i32 %res
}

; Check two register operands, second operand inverted.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: and.c %r2, %r2, %r3
; CHECK: jmp %r1
  %notb = xor i32 %b, -1
  %res = and i32 %a, %notb
  ret i32 %res
}

; Check immediate in low 16 bits, high 16 bits clear.
define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: mask %r2, %r2, 51966
; CHECK: jmp %r1
  %res = and i32 %a, 51966
  ret i32 %res
}

; Check immediate in low 16 bits, high 16 bits set.
define i32 @func_f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK: and %r2, %r2, 57005
; CHECK: jmp %r1
  %res = and i32 %a, 4294958765 ; = 0xFFFFDEAD
  ret i32 %res
}

; Check immediate in high 16 bits, low 16 bits clear.
define i32 @f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK: mask.u %r2, %r2, 47806
; CHECK: jmp %r1
  %res = and i32 %a, 3133014016 ; = 0xBABE0000
  ret i32 %res
}

; Check immediate in high 16 bits, low 16 bits set.
define i32 @func_f6(i32 %a) {
; CHECK-LABEL: f6:
; CHECK: and.u %r2, %r2, 47806
; CHECK: jmp %r1
  %res = and i32 %a, 3133079551 ; = 0xBABEFFFF
  ret i32 %res
}

; Check 32-bit immediate.
define i32 @func_f7(i32 %a) {
; CHECK-LABEL: f7:
; CHECK: and.u %r0, %r2, 61680
; CHECK: and %r2, %r0, 61680
; CHECK: jmp %r1
  %res = and i32 %a, 4042322160 ; = 0xF0F0F0F0
  ret i32 %res
}

; Check inverted mask.
define i32 @f8(i32 %a) {
; CHECK-LABEL: f8:
; CHECK: clr %r2, %r2, 16<8>
; CHECK: jmp %r1
  %res = and i32 %a, 4278190335 ; = 0xFF0000FF
  ret i32 %res
}
