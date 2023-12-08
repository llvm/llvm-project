; Test 64-bit XORs in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the lowest useful XILF value.
define i64 @f1(i64 %a) {
; CHECK-LABEL: f1:
; CHECK: xilf %r2, 1
; CHECK: br %r14
  %xor = xor i64 %a, 1
  ret i64 %xor
}

; Check the high end of the XILF range.
define i64 @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK: xilf %r2, 4294967295
; CHECK: br %r14
  %xor = xor i64 %a, 4294967295
  ret i64 %xor
}

; Check the lowest useful XIHF value, which is one up from the above.
define i64 @f3(i64 %a) {
; CHECK-LABEL: f3:
; CHECK: xihf %r2, 1
; CHECK: br %r14
  %xor = xor i64 %a, 4294967296
  ret i64 %xor
}

; Check the next value up again, which needs a combination of XIHF and XILF.
define i64 @f4(i64 %a) {
; CHECK-LABEL: f4:
; CHECK: xihf %r2, 1
; CHECK: xilf %r2, 4294967295
; CHECK: br %r14
  %xor = xor i64 %a, 8589934591
  ret i64 %xor
}

; Check the high end of the XIHF range.
define i64 @f5(i64 %a) {
; CHECK-LABEL: f5:
; CHECK: xihf %r2, 4294967295
; CHECK: br %r14
  %xor = xor i64 %a, -4294967296
  ret i64 %xor
}

; Check the next value up, which again must use XIHF and XILF.
define i64 @f6(i64 %a) {
; CHECK-LABEL: f6:
; CHECK: xihf %r2, 4294967295
; CHECK: xilf %r2, 1
; CHECK: br %r14
  %xor = xor i64 %a, -4294967295
  ret i64 %xor
}

; Check full bitwise negation
define i64 @f7(i64 %a) {
; CHECK-LABEL: f7:
; CHECK: xihf %r2, 4294967295
; CHECK: xilf %r2, 4294967295
; CHECK: br %r14
  %xor = xor i64 %a, -1
  ret i64 %xor
}
