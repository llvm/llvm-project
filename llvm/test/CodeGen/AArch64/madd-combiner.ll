; RUN: llc -mtriple=aarch64-apple-darwin            -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -verify-machineinstrs < %s | FileCheck %s

; Test that we use the correct register class.
define i32 @mul_add_imm(i32 %a, i32 %b) {
; CHECK-LABEL: mul_add_imm
; CHECK:       orr [[REG:w[0-9]+]], wzr, #0x4
; CHECK-NEXT:  madd  {{w[0-9]+}}, w0, w1, [[REG]]
  %1 = mul i32 %a, %b
  %2 = add i32 %1, 4
  ret i32 %2
}

define i32 @mul_sub_imm1(i32 %a, i32 %b) {
; CHECK-LABEL: mul_sub_imm1
; CHECK:       mov [[REG:w[0-9]+]], #4
; CHECK-NEXT:  msub  {{w[0-9]+}}, w0, w1, [[REG]]
  %1 = mul i32 %a, %b
  %2 = sub i32 4, %1
  ret i32 %2
}

; bugpoint reduced test case. This only tests that we pass the MI verifier.
define void @mul_add_imm2() {
entry:
  br label %for.body
for.body:
  br i1 undef, label %for.body, label %for.body8
for.body8:
  %0 = mul i64 undef, -3
  %mul1971 = add i64 %0, -3
  %cmp7 = icmp slt i64 %mul1971, 1390451930000
  br i1 %cmp7, label %for.body8, label %for.end20
for.end20:
  ret void
}

