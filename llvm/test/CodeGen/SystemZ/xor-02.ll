; Test 32-bit XORs in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the lowest useful XILF value.
define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: xilf %r2, 1
; CHECK: br %r14
  %xor = xor i32 %a, 1
  ret i32 %xor
}

; Check the high end of the signed range.
define i32 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: xilf %r2, 2147483647
; CHECK: br %r14
  %xor = xor i32 %a, 2147483647
  ret i32 %xor
}

; Check the low end of the signed range, which should be treated
; as a positive value.
define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: xilf %r2, 2147483648
; CHECK: br %r14
  %xor = xor i32 %a, -2147483648
  ret i32 %xor
}

; Check the high end of the XILF range.
define i32 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK: xilf %r2, 4294967295
; CHECK: br %r14
  %xor = xor i32 %a, 4294967295
  ret i32 %xor
}
