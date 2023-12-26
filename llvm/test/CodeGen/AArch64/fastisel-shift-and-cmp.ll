; RUN: llc -fast-isel -mtriple=aarch64-none-none < %s | FileCheck %s

; Check that the shl instruction did not get folded in together with 
; the cmp instruction. It would create a miscompilation 

define i32 @icmp_i8_shift_and_cmp(i8 %a, i8 %b) {
  %op1 = xor i8 %a, -49
  %op2 = mul i8 %op1, %op1
; CHECK-NOT: cmp [[REGS:.*]] #[[SHIFT_VAL:[0-9]+]]
  %op3 = shl i8 %op2, 3
  %tmp3 = icmp eq i8 %b, %op3
  %conv = zext i1 %tmp3 to i32
  ret i32 %conv
}

